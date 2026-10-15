"""Background processing, run as its own low-priority process (started and supervised by the server).
It only talks to the server through files, so an out-of-memory kill of the worker never touches a running recording.
Stages are checkpointed: transcription chunk by chunk (transcript.partial.jsonl), summary map step by map step."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import store  # noqa: E402

log = logging.getLogger("worker")
MAX_ATTEMPTS = 3
WIN_S = 15 * 60  # VAD window: bounds RAM for long meetings

_ctc = _whisper = None


def release_models():
    """Idle: give the speech models back (about 1.4 GB) so the box stays smooth for everything else."""
    global _ctc, _whisper
    if _ctc is None and _whisper is None:
        return
    if _whisper is not None and hasattr(_whisper, "close"):
        _whisper.close()  # stops the Qwen ASR child process (about 8 GB)
    _ctc = _whisper = None
    import gc

    gc.collect()
    log.info("idle: speech models released")


def release_post_models():
    global _embed
    if _embed is None:
        return
    _embed = None
    import gc

    gc.collect()
    log.info("idle: speaker model released")


recording_active = store.recording_active

PAUSED_PHASE = "paused: a meeting is being recorded · พักไว้ระหว่างบันทึกการประชุม"
_pause_prev: dict[str, tuple] = {}  # mid -> (phase before the pause, pause start)
_paused_s: dict[str, float] = {}    # mid -> seconds paused for recordings during the current stage (not processing time: stats subtract it)


def set_rec_pause(mid: str, on: bool):
    """Shows on the meeting that its processing waits for a recording (and counts the wait). Called by the stages' pause points and by the
    summary model's freeze guard; a second 'on' or an 'off' without 'on' is ignored."""
    if on:
        if mid in _pause_prev:
            return
        st = store.load_state(mid) or {}
        _pause_prev[mid] = (st.get("phase"), time.time())
        store.update_state(mid, phase=PAUSED_PHASE, eta_s=None)
        log.info("%s paused: a meeting is being recorded", mid)
        return
    if mid not in _pause_prev:
        return
    phase, t = _pause_prev.pop(mid)
    secs = time.time() - t
    _paused_s[mid] = _paused_s.get(mid, 0.0) + secs
    store.update_state(mid, phase=phase)
    store.log_event(mid, f"paused {secs:.0f}s while a meeting was recorded")
    log.info("%s goes on after %.0fs paused for a recording", mid, secs)


class Deferred(BaseException):
    """The processing window closed: the stage stops at this checkpoint and goes back to the queue (BaseException: no stage catches it)."""


def check_window(mid: str):
    """Raise Deferred outside the processing window (schedule.py) unless the meeting is a rush or its stage is about to finish."""
    import schedule

    st = store.load_state(mid) or {}
    if schedule.may_run(store.settings(), st):
        return
    eta = st.get("eta_s")
    if isinstance(eta, (int, float)) and eta - max(0.0, time.time() - float(st.get("eta_at") or time.time())) <= schedule.GRACE_S:
        return
    raise Deferred


def pause_while_recording(mid: str | None = None):
    if recording_active():
        if mid:
            set_rec_pause(mid, True)
        try:
            while recording_active():
                time.sleep(2)
        finally:
            if mid:
                set_rec_pause(mid, False)
    if mid:
        check_window(mid)


def wait_for_ram(min_gb: float, what: str):
    while store.mem_available_gb() < min_gb:
        log.warning("waiting for %.1f GB free RAM before %s (now %.1f)", min_gb, what, store.mem_available_gb())
        time.sleep(15)


_whisper_kind = ""
CT2_KINDS = {v[0]: v for v in config.WHISPER_CT2.values()}  # worker kind -> (kind, model dir, engine label, RAM, name)


def engines(accurate: bool, kind: str = "whisper"):
    """(fast Typhoon pass, accurate engine or None). kind 'qwen' = Typhoon ASR Qwen 1.7B with contextual biasing (slow, best), else Thonburian Whisper."""
    global _ctc, _whisper, _whisper_kind
    import asr

    if _ctc is None:
        _ctc = asr.ThaiCTC(threads=4, npu=bool(store.settings().get("asr_npu", True)))
    if accurate and (_whisper is None or _whisper_kind != kind):
        if _whisper is not None and hasattr(_whisper, "close"):
            _whisper.close()
        _whisper = None
        if kind == "wctc":
            import wctc

            try:
                _whisper = wctc.WhisperCTC()
            except Exception as e:  # noqa: BLE001 - the NPU is busy / out of memory: the default Whisper does the job
                log.warning("Whisper CTC could not load (%s): using the default Whisper", str(e)[:120])
                kind = "whisper"
                wait_for_ram(3.0, "loading Whisper")
                _whisper = asr.WhisperTH(threads=config.CORES)
        elif kind in CT2_KINDS:
            _, path, _, ram, name = CT2_KINDS[kind]
            wait_for_ram(ram, "loading " + name)
            _whisper = asr.WhisperTH(threads=config.CORES, model_dir=path)
        elif kind == "wen":
            wait_for_ram(2.5, "loading the English Whisper (large-v3-turbo)")
            _whisper = asr.WhisperTH(threads=config.CORES, model_dir=config.WHISPER_EN, language="en")
        elif kind == "rnnt":
            import rnnt

            wait_for_ram(2.0, "loading Pathumma FastConformer")
            _whisper = rnnt.PathummaRNNT(threads=config.CORES)
        elif kind == "vote":
            import vote_asr

            wait_for_ram(vote_asr.RAM_GB, "loading the three models of the Best (vote) mode")
            _whisper = vote_asr.VoteASR(threads=config.CORES)
        elif kind == "qwen":
            import qwen_asr

            wait_for_ram(10.0, "loading the Qwen ASR (about 8 GB)")
            _whisper = qwen_asr.QwenASR(threads=config.CORES, batch=12)
        else:
            wait_for_ram(3.0, "loading Whisper")
            _whisper = asr.WhisperTH(threads=config.CORES)
        _whisper_kind = kind
    return _ctc, (_whisper if accurate else None)


_embed = None


_embed_lock = threading.Lock()


def embedder():
    global _embed
    import speakers

    with _embed_lock:
        if _embed is None:
            _embed = speakers.Embedder(threads=4)
        return _embed


class Eta:
    """Progress callback that also tells the UI how long is left: eta_s, measured at eta_at (epoch seconds)."""

    def __init__(self, mid: str):
        self.mid, self.t0, self.last, self.checked = mid, time.time(), 0.0, time.time()

    def __call__(self, f: float):
        if recording_active():  # speakers / summary between model calls: wait for the recording to end (the time left excludes the wait)
            t = time.time()
            pause_while_recording(self.mid)
            self.t0 += time.time() - t
        now = time.time()
        if now - self.last < 1.0 and f < 0.999:
            return
        self.last = now
        el = now - self.t0
        eta = round(el * (1 - f) / f) if f >= 0.04 and el >= 8 else None  # None: not enough done yet to tell
        store.update_state(self.mid, progress=round(f, 3), eta_s=eta, eta_at=round(now, 1))
        if now - self.checked > 10 and f < 0.999:
            self.checked = now
            check_window(self.mid)


def plan_chunks(mid: str, audio, cfg: dict) -> list[list]:
    """[start, end, speaker-id | None] chunks for ASR, computed once and stored (with speakers.json), so a resumed run
    uses exactly the same chunks and never repeats the speaker analysis."""
    import asr
    import numpy as np
    import speakers

    d = store.mdir(mid)
    cached = store.read_json(d / "chunks.json")
    if cached and (d / "chunks.filled").exists():
        return [c if len(c) >= 3 else [c[0], c[1], None] for c in cached]
    if cached:  # planned by an older version (no pass for speech the VAD missed): add that pass, redo the recognised pieces
        chunks = [list(c) if len(c) >= 3 else [c[0], c[1], None] for c in cached]
        (d / "transcript.partial.jsonl").unlink(missing_ok=True)
        chunks = fill_missed(mid, audio, chunks)
        return chunks
    if cfg["identify_speakers"] and speakers.available() and not cfg["speakers_after"]:
        store.update_state(mid, phase="identifying speakers", progress=0.0)
        chunks, spk = speakers.analyse(audio, embedder(), cfg, Eta(mid))
        store.write_json(d / "speakers.json", spk)
        store.log_event(mid, f"found {len(spk)} speaker(s)")
    else:
        chunks, pos, n, win = [], 0, len(audio), WIN_S * config.SR
        (d / "speakers.json").unlink(missing_ok=True)
        while pos < n:
            end = min(n, pos + win)
            found = [[x + pos, y + pos, None] for x, y in asr.speech_chunks(audio[pos:end])]
            if end < n and found and end - found[-1][1] < config.SR:  # speech runs into the window edge: redo it in the next window
                end = found.pop()[0]
                if end <= pos:
                    end = min(n, pos + win)
            chunks += found
            pos = end
    return fill_missed(mid, audio, merge_live(mid, audio, chunks))


def merge_live(mid: str, audio, chunks: list[list]) -> list[list]:
    """Pieces already decoded during the meeting (whisper.live.jsonl) are kept as they are; the planned chunks only cover what is left."""
    import numpy as np

    d = store.mdir(mid)
    n = len(audio)
    recs = [r for r in read_jsonl(d / "whisper.live.jsonl") if 0 <= r["a"] < r["b"] <= n]
    if not recs:
        return chunks
    nf = n // 320 + 2
    covered = np.zeros(nf, bool)
    for r in recs:
        covered[r["a"] // 320: r["b"] // 320 + 1] = True
    out = [[r["a"], r["b"], None, "live"] for r in recs]
    for c in chunks:
        a, b = c[0], c[1]
        free = ~covered[a // 320: b // 320 + 1]
        i = 0
        while i < len(free):
            if not free[i]:
                i += 1
                continue
            j = i
            while j < len(free) and free[j]:
                j += 1
            ra, rb = a + i * 320, min(b, a + j * 320)
            if rb - ra >= 0.8 * config.SR:
                out.append([ra, rb, *c[2:]])
            i = j
    out.sort(key=lambda c: c[0])
    store.log_event(mid, f"{len(recs)} piece(s) ({sum(r['b'] - r['a'] for r in recs) / config.SR:.0f} s) were already transcribed during the meeting")
    return out


def fill_missed(mid: str, audio, chunks: list[list]) -> list[list]:
    import asr
    import numpy as np

    d = store.mdir(mid)
    extra = asr.fill_gaps(audio, chunks)  # speech the VAD missed
    if extra:
        # Every Whisper call costs a full 30 s window, so a missed piece next to a chunk (<= 2 s away, result <= 28 s) is simply added to it;
        # only isolated pieces become their own chunk, and those are kept only if the fast pass finds words in them.
        chunks.sort(key=lambda c: c[0])
        lone = []
        for a, b in extra:
            prev = max((c for c in chunks if c[1] <= a + 1), key=lambda c: c[1], default=None)
            nxt = min((c for c in chunks if c[0] >= b - 1), key=lambda c: c[0], default=None)
            if prev is not None and a - prev[1] <= 2 * config.SR and b - prev[0] <= 28 * config.SR:
                prev[1] = max(prev[1], b)
            elif nxt is not None and nxt[0] - b <= 2 * config.SR and nxt[1] - a <= 28 * config.SR:
                nxt[0] = min(nxt[0], a)
            else:
                lone.append((a, b))
        store.log_event(mid, f"{len(extra)} stretch(es) with sound but no detected speech found ({sum(b - a for a, b in extra) / config.SR:.0f} s): {len(extra) - len(lone)} joined to neighbouring speech, {len(lone)} checked on their own")
        centroids = {k: np.array(v["centroid"], np.float32) for k, v in (store.read_json(d / "speakers.json") or {}).items() if v.get("centroid")}
        for a, b in lone:
            who = None
            if centroids and b - a >= 0.8 * config.SR:  # nearest known voice
                e = embedder()(audio[a:b])
                who = max(centroids, key=lambda k: float(e @ centroids[k]))
            chunks.append([a, b, who, True])
        chunks.sort(key=lambda c: c[0])
    store.write_json(d / "chunks.json", chunks)
    (d / "chunks.filled").write_text("1")
    return chunks



def next_stage_after_transcript(cfg: dict, has_text: bool, mid: str):
    """Post tasks follow the transcript: who said what (when it was not done while chunking), then the summary."""
    import speakers

    if not has_text:
        return None
    if cfg["identify_speakers"] and cfg["speakers_after"] and speakers.available():
        return "speakers"
    return "summarize" if cfg["auto_summary"] else None


def transcribe_stats(mid, cfg, mode, kind, segs, chunks, done, work, live_done, audio, whisper, fixes, pack_ids) -> dict:
    """What the statistics keep about a transcription run (see stats.py)."""
    import statistics

    import textnorm
    from rapidfuzz.distance import Levenshtein

    srcs: dict = {}
    for s in segs:
        srcs[s["src"]] = srcs.get(s["src"], 0) + 1
    confs = [s["conf"] for s in segs if isinstance(s.get("conf"), (int, float))]
    dis_d = dis_n = 0
    for r in done.values():  # how far the final text is from what the fast Typhoon pass heard (a quality proxy: no ground truth exists)
        ty, fin = (r.get("typhoon") or ""), (r.get("text") or "")
        if ty and fin and not str(r.get("src", "")).startswith("typhoon"):
            a_, b_ = "".join(textnorm.polish(ty, False, False).split()), "".join(textnorm.polish(fin, False, False).split())
            dis_d += Levenshtein.distance(a_, b_)
            dis_n += max(len(a_), len(b_))
    live = sum(1 for c in chunks if len(c) > 3 and c[3] == "live" and (c[0], c[1]) in live_done)
    info = {"mode": mode, "engine": {"qwen": "typhoon-asr-qwen-1.7b-ctx", "rnnt": "pathumma-stt-fastconformer-rnnt", "vote": "vote3: pathumma-whisper-large + pathumma-whisper-medium + fastconformer", **{k: v[2] for k, v in CT2_KINDS.items()}, "wctc": "typhoon-whisper-medium-ctc (NPU)", "whisper": "thonburian-distil-whisper-medium"}[kind] if mode != "fast" else "typhoon-realtime-ctc (NPU)",
            "audio_s": round(len(audio) / config.SR, 1), "pieces": len(chunks), "pieces_from_live_pass": live, "pieces_computed": work["n"],
            "src_mix": srcs, "segments": len(segs), "chars": sum(len(s["text"]) for s in segs), "conf_mean": round(statistics.mean(confs), 3) if confs else None,
            "disagreement_pct": round(100 * dis_d / dis_n, 1) if dis_n else None, "term_corrections": len(fixes), "vocab_packs": pack_ids,
            "asr_npu": bool(cfg.get("asr_npu")), "live_accurate": bool(cfg.get("live_accurate"))}
    if hasattr(whisper, "stats"):
        info["npu_calls"] = whisper.stats()
    return info


def pick_engine(mid: str, cfg: dict) -> tuple[str, bool, str]:
    """(accuracy mode, accurate?, engine kind) as set in Settings; a model that is not installed falls back (noted in the meeting's log).
    The meeting's language (state "language", else the setting) comes first: English-only meetings use the English Whisper (the Thai models
    loop or write Thai for English speech); Thai + English meetings need a Whisper model (FastConformer / the vote / the CTC heads have no
    English), so a Thai-only choice becomes Typhoon Whisper Large v3 when installed, else the Thonburian Whisper."""
    mode = cfg["accuracy"]
    lang = meeting_language(mid, cfg)
    if lang == "en":
        if (config.WHISPER_EN / "model.bin").exists():
            return "english", True, "wen"
        store.log_event(mid, "English meeting, but the English Whisper model is not installed (models/whisper-en): the Thai models are used")
    elif lang == "th-en" and (mode in ("fast", "npu", "pathumma-rnnt", "vote3", "best") or mode not in ("accurate", *config.WHISPER_CT2)):
        if (config.WHISPER_LV3 / "model.bin").exists():
            store.log_event(mid, f"Thai + English meeting: {engines_names().get(mode, mode)} has no English, Typhoon Whisper Large v3 is used instead")
            mode = "large"
        else:
            store.log_event(mid, f"Thai + English meeting: {engines_names().get(mode, mode)} has no English, the Thonburian Whisper is used instead")
            mode = "accurate"
    accurate = mode in ("accurate", "best", "npu", "pathumma-rnnt", "vote3", *config.WHISPER_CT2)
    kind = "whisper"
    if mode == "vote3":
        import vote_asr

        if vote_asr.available():
            kind = "vote"
        else:
            store.log_event(mid, "the Best (vote) mode needs Pathumma Whisper Large + medium + FastConformer: using the default Whisper instead")
    if mode == "npu":
        import wctc

        if wctc.available():
            kind = "wctc"
        else:
            store.log_event(mid, "the Whisper CTC model is not installed (models/whisper-ctc-npu): using the default Whisper instead")
    if mode in config.WHISPER_CT2:
        k, path, _, _, name = config.WHISPER_CT2[mode]
        if (path / "model.bin").exists():
            kind = k
        else:
            store.log_event(mid, f"{name} is not installed ({path.relative_to(config.ROOT)}): using the default Whisper instead")
    if mode == "pathumma-rnnt":
        import rnnt

        if rnnt.available():
            kind = "rnnt"
        else:
            store.log_event(mid, "Pathumma FastConformer is not installed (model or NeMo environment missing): using the default Whisper instead")
    if mode == "best":
        import qwen_asr

        if qwen_asr.available():
            kind = "qwen"
        else:
            store.log_event(mid, "best accuracy needs the Qwen environment (.venv-qwen): using Whisper instead")
    return mode, accurate, kind


def meeting_language(mid: str, cfg: dict) -> str:
    st = store.load_state(mid) or {}
    lang = st.get("language") or cfg.get("language") or "th"
    return lang if lang in config.LANGUAGES else "th"


def engines_names() -> dict:
    import engines

    return engines.ACCURACY


def do_transcribe(mid: str):
    import json

    import asr
    import report

    cfg = store.settings()
    mode, accurate, kind = pick_engine(mid, cfg)
    if not store.wav_path(mid).exists():
        store.finalize_audio(mid)
    audio = store.load_audio(mid)
    store.update_state(mid, status="transcribing", phase="preparing", progress=0.0, eta_s=None, error=None, duration_s=len(audio) / config.SR)
    chunks = plan_chunks(mid, audio, cfg)
    store.update_state(mid, phase="transcribing", progress=0.0)
    ctc, whisper = engines(accurate, kind)
    part = store.mdir(mid) / "transcript.partial.jsonl"
    done = {}
    if part.exists():
        for line in part.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[r["i"]] = r
            except ValueError:
                pass  # torn last line after a crash: redo that chunk
    import learn

    import vocab

    meeting_state = store.load_state(mid) or {}
    pack_ids = vocab.active_ids(meeting_state)
    cfg = {**cfg, "vocab_rules": vocab.rules(pack_ids)}  # domain packs: Thai spellings -> terms (asr.polish below)
    glossary_terms = [x.strip() for x in cfg["glossary"].replace("\n", ",").split(",") if x.strip()] + vocab.terms(pack_ids, skip_thai_words=True)
    prompt = ""  # Whisper gets NO prompt (it ruins this Thai model); the terms are applied after recognition (correction) and in the summary
    if cfg["use_materials"]:
        learn.refresh_terms(mid)
    if kind == "qwen":  # contextual biasing: people, agenda words, names/acronyms from the documents and the glossary are spelled as written
        meeting = store.load_state(mid) or {}
        whisper.set_bias(list(meeting.get("participants") or []) + (learn.correction_terms(mid) if cfg["use_materials"] else []) + glossary_terms)
    live_done = {(r["a"], r["b"]): r for r in read_jsonl(store.mdir(mid) / "whisper.live.jsonl")} if accurate and kind == "whisper" else {}  # (the live lane's pieces are from the default Whisper)
    t_start = time.time()
    # progress by AUDIO LENGTH (a 25 s piece is worth more than a 2 s one); the time left by the NUMBER of pieces still to compute x the measured
    # time per piece (every Whisper call costs a full 30 s window whatever the length). Pieces decoded during the meeting or by an earlier run cost nothing.
    lens = [c[1] - c[0] for c in chunks]
    total_len = max(1, sum(lens))
    todo_n = sum(1 for i, c in enumerate(chunks) if i not in done and not (len(c) > 3 and c[3] == "live" and (c[0], c[1]) in live_done))
    work = {"n": 0, "secs": 0.0, "last": 0.0}

    def tick(force: bool = False):
        now = time.time()
        if not force and now - work["last"] < 1.5:
            return
        work["last"] = now
        got = sum(lens[j] for j in done)
        eta = round((todo_n - work["n"]) * work["secs"] / work["n"]) if work["n"] >= 3 and work["secs"] >= 20 else None
        store.update_state(mid, progress=round(got / total_len, 3), eta_s=max(0, eta) if eta is not None else None, eta_at=round(now, 1))
    BATCH = (12 if kind == "qwen" else 8) if whisper else 1  # Whisper decodes several chunks at once (see asr.WhisperTH.transcribe_many): ~2.5x faster, same text
    with open(part, "a", encoding="utf-8") as pf:
        pending: list = []  # (index, chunk, typhoon text, typhoon confidence, seconds spent so far) waiting for the Whisper batch

        def save(rec):
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.flush()

        def flush_batch(last: bool = False):
            if not pending:
                return
            t = time.time()
            ws = whisper.transcribe_many([audio[c[0]:c[1]] for _, c, _, _, _ in pending], BATCH) if whisper else [None] * len(pending)
            alts = list(getattr(whisper, "last_alts", None) or []) if kind == "vote" else []
            dt = time.time() - t
            for n_, ((i, c, tt, tc, spent), w) in enumerate(zip(pending, ws)):
                text, src, conf = asr.choose(w, (tt, tc))
                if kind == "vote" and src.startswith("whisper"):
                    src = "vote"
                if kind in ("qwen", "wctc") and src.startswith("whisper"):
                    src = src.replace("whisper", kind)
                rec = {"i": i, "t0": round(c[0] / config.SR, 2), "t1": round(c[1] / config.SR, 2), "text": text, "src": src, "spk": c[2],
                       "conf": round(conf, 3), "typhoon": tt, "whisper": w[0] if w else None}
                if n_ < len(alts):
                    rec["vote"] = alts[n_]
                save(rec)
                done[i] = rec
                work["secs"] += spent
            os.fsync(pf.fileno())
            work["n"] += len(pending)
            work["secs"] += dt
            pending.clear()
            tick(last)

        for i, ch in enumerate(chunks):
            a, b, who = ch[:3]
            if i in done:
                continue
            pause_while_recording(mid)
            if len(ch) > 3 and ch[3] == "live" and (a, b) in live_done:  # decoded during the meeting
                rec = {**live_done[(a, b)], "i": i, "t0": round(a / config.SR, 2), "t1": round(b / config.SR, 2), "spk": who}
                save(rec)
                done[i] = rec
                tick()
                continue
            t_chunk = time.time()
            tt, tc = ("", 0.0) if kind == "wen" else ctc.transcribe_long(audio[a:b])  # (the Thai realtime pass would only write Thai for English speech)
            if kind != "wen" and len(ch) > 3 and ch[3] is True and (len(tt.strip()) < 6 or tc < 0.6):  # a stretch the VAD skipped: keep it only if the fast pass really heard words
                save({"i": i, "t0": round(a / config.SR, 2), "t1": round(b / config.SR, 2), "text": "", "src": "silence", "spk": who, "conf": 0.0, "typhoon": tt, "whisper": None})
                done[i] = {"text": "", "src": "silence", "conf": 0.0, "t0": round(a / config.SR, 2), "t1": round(b / config.SR, 2), "spk": who}
                work["n"] += 1
                work["secs"] += time.time() - t_chunk
                tick()
                continue
            pending.append((i, ch, tt, tc, time.time() - t_chunk))
            if len(pending) >= BATCH:
                flush_batch()
        flush_batch(True)
    segs = []
    for i in range(len(chunks)):
        r = done[i]
        text = asr.polish(r["text"], cfg)
        if text.strip():
            segs.append({"t0": r["t0"], "t1": r["t1"], "text": text, "src": r["src"], "conf": r["conf"], "spk": r.get("spk")})
            v = r.get("vote") or {}
            others = {k: asr.polish(t, cfg) for k, t in (v.get("alts") or {}).items() if k != v.get("pick") and t.strip()}
            others = {k: t for k, t in others.items() if t.replace(" ", "") != text.replace(" ", "")}
            if others and v.get("agree", 1.0) < 0.9:  # the models disagree here: the other versions, for a person to choose from
                segs[-1]["alts"] = others
    d = store.mdir(mid)
    fixes = []
    if cfg["apply_term_corrections"] and cfg["use_materials"] and segs:
        fixes = learn.correct_segments(segs, learn.correction_terms(mid) + glossary_terms)  # near-miss spellings of known terms; originals stay in seg["orig"]
    store.write_json(d / "transcript.json", {"engine": ("whisper-large-v3-turbo-en" if kind == "wen" else "vote3: pathumma-whisper-large + pathumma-whisper-medium + fastconformer" if kind == "vote" else "typhoon-asr-qwen-1.7b-ctx+typhoon" if kind == "qwen" else "pathumma-stt-fastconformer-rnnt+typhoon" if kind == "rnnt" else CT2_KINDS[kind][2] + "+typhoon" if kind in CT2_KINDS else "typhoon-whisper-medium-ctc+typhoon" if kind == "wctc" else "whisper-th+typhoon") if accurate else "typhoon",
                                             "created": time.strftime("%F %T"), "segments": segs, "term_corrections": fixes})
    store.atomic_write(d / "transcript.md", store.transcript_md(store.load_state(mid), segs, store.speaker_names(mid)))
    if segs and (d / "speakers.json").exists() and not cfg.get("speakers_after"):  # voices known already: split overlapping speech now
        separate_overlaps(mid, audio, cfg)
        segs = (store.read_json(d / "transcript.json") or {}).get("segments", segs)
    if next_stage_after_transcript(cfg, bool(segs), mid) != "speakers":  # (otherwise the speakers step does it, after the voices are known)
        merge_mics(mid, audio, cfg)
        segs = (store.read_json(d / "transcript.json") or {}).get("segments", segs)
    report.refresh(mid)
    part.unlink(missing_ok=True)
    store.update_state(mid, status="transcribed", phase=None, progress=1.0, eta_s=0, queue=next_stage_after_transcript(cfg, bool(segs), mid),
                       attempts=0, words=sum(len(s["text"]) for s in segs))
    store.log_event(mid, f"transcribed {len(segs)} segments in {time.time() - t_start:.0f}s ({'best (3-model vote)' if kind == 'vote' else 'best' if kind == 'qwen' else mode if kind == 'rnnt' else mode if kind in CT2_KINDS else 'npu' if kind == 'wctc' else 'accurate' if accurate else 'fast'})" + (f"; {len(fixes)} term correction(s) from the documents/agenda" if fixes else ""))
    return transcribe_stats(mid, cfg, mode, kind, segs, chunks, done, work, live_done, audio, whisper, fixes, pack_ids)


def separate_overlaps(mid: str, audio, cfg: dict) -> dict:
    """Two people talking at once: the lines covering such a stretch are replaced by one line per voice, each transcribed from that voice
    alone (separation.py). Needs the speakers of the meeting (their voiceprints); lines a person corrected or checked are left alone."""
    import asr
    import numpy as np
    import separation
    import speakers

    if not cfg.get("separate_overlaps", True) or not separation.overlap_available():
        return {}
    d = store.mdir(mid)
    spk = store.read_json(d / "speakers.json") or {}
    cents = {k: speakers.unit(v["centroid"]) for k, v in spk.items() if v.get("centroid") and (v.get("talk_s") or 0) >= 3}
    t0 = time.time()
    store.update_state(mid, phase="finding overlapping speech")
    ovs = separation.find_overlaps(audio, float(cfg.get("overlap_threshold", 0.3)))
    stats = {"overlaps": len(ovs), "overlap_s": round(sum(b - a for a, b in ovs) / config.SR, 1), "separated": 0, "lines_in": 0, "lines_out": 0, "kept": 0}
    if not ovs or len(cents) < 2:
        stats["find_s"] = round(time.time() - t0, 1)
        if ovs:
            store.log_event(mid, f"{len(ovs)} stretch(es) with two voices at once ({stats['overlap_s']} s), but fewer than two known voices: left as they are")
        return stats
    if not separation.separation_available():
        store.log_event(mid, f"{len(ovs)} stretch(es) with two voices at once ({stats['overlap_s']} s): voice separation is not installed (models/separation)")
        return stats
    tr = store.read_json(d / "transcript.json") or {}
    segs = tr.get("segments") or []
    plan = separation.plan(segs, ovs)
    wins = [w for w in plan if not any(separation.protected(segs[i]) for i in range(w["lo"], w["hi"] + 1)) and not any(segs[i].get("inserted") for i in range(w["lo"], w["hi"] + 1))]
    stats.update(kept=len(plan) - len(wins), added=0)
    if not wins:
        return stats
    import vocab

    cfg = {**cfg, "vocab_rules": vocab.rules(vocab.active_ids(store.load_state(mid) or {}))}
    _, accurate, kind = pick_engine(mid, cfg)
    ctc, whisper = engines(accurate, kind)
    sep = separation.Separator(threads=8)
    done = []

    def transcribe(pcms):
        ws = whisper.transcribe_many(pcms, 8) if whisper else [None] * len(pcms)
        out = []
        for pcm, w in zip(pcms, ws):
            text, src, conf = asr.choose(w, ctc.transcribe_long(pcm))
            out.append((asr.polish(text, cfg), src, conf))
        return out

    try:
        for n, w in enumerate(wins):
            pause_while_recording(mid)
            store.update_state(mid, phase=f"separating overlapping voices ({n + 1}/{len(wins)})", progress=round(n / len(wins), 3))
            lo, hi = w["lo"], w["hi"]
            if w["mode"] == "replace":
                new = separation.resolve_window(audio, segs, lo, hi, sep, embedder(), cents, transcribe)
            else:
                new = separation.insert_window(audio, segs, lo, hi, w["spans"], sep, embedder(), cents, transcribe)
            if new:
                done.append((lo, hi, [segs[i]["text"] for i in range(lo, hi + 1)], w["mode"], new))
    finally:
        sep.close()
    with store.transcript_lock(mid):  # re-read: a line may have been corrected meanwhile; such a window is left alone
        tr = store.read_json(d / "transcript.json") or tr
        segs = tr.get("segments") or []
        for lo, hi, before, mode, new in sorted(done, key=lambda x: -x[0]):
            if hi < len(segs) and [segs[i]["text"] for i in range(lo, hi + 1)] == before and not any(separation.protected(segs[i]) for i in range(lo, hi + 1)):
                if mode == "replace":
                    segs[lo:hi + 1] = new
                    stats["separated"] += 1
                    stats["lines_in"] += hi - lo + 1
                    stats["lines_out"] += len(new)
                else:  # the lines stay; the other voice's words are added after the line they were said over
                    segs[hi + 1:hi + 1] = new
                    stats["added"] += len(new)
        tr["segments"] = segs
        tr["overlaps"] = {**stats, "secs": round(time.time() - t0, 1)}
        store.write_json(d / "transcript.json", tr)
    store.atomic_write(d / "transcript.md", store.transcript_md(store.load_state(mid), segs, store.speaker_names(mid)))
    stats["secs"] = round(time.time() - t0, 1)
    store.log_event(mid, f"{len(ovs)} stretch(es) with two voices at once ({stats['overlap_s']} s): {stats['separated']} separated into one line per voice "
                         f"({stats['lines_in']} line(s) -> {stats['lines_out']}), {stats['added']} short remark(s) added as their own line, in {stats['secs']:.0f}s"
                         + (f"; {stats['kept']} left alone (corrected by hand, or done before)" if stats["kept"] else ""))
    return stats


def merge_mics(mid: str, audio, cfg: dict) -> dict:
    """Participants who sent their phone's microphone (join link): their own words come from their phone (clean, even when someone talks
    over them), the room microphone's copy of those words goes, other people's words said at the same time stay. See mics.py."""
    import asr
    import mics
    import speakers
    import vocab

    tracks = mics.load_index(mid)
    if not tracks or not speakers.available():
        return {}
    d = store.mdir(mid)
    t0 = time.time()
    store.update_state(mid, phase="adding participants' own microphones")
    cfg = {**cfg, "vocab_rules": vocab.rules(vocab.active_ids(store.load_state(mid) or {}))}
    _, accurate, kind = pick_engine(mid, cfg)
    ctc, whisper = engines(accurate, kind)
    spk = store.read_json(d / "speakers.json") or {}
    cents = {k: speakers.unit(v["centroid"]) for k, v in spk.items() if v.get("centroid")}
    stats = {"tracks": len(tracks), "people": 0, "lines": 0, "dropped": 0, "skipped": []}
    by_name: dict[str, list[dict]] = {}
    for t in tracks:
        by_name.setdefault(t["name"], []).append(t)
    new_lines: dict[str, tuple[list[dict], str | None]] = {}
    for name, ts in by_name.items():
        lines, cvecs, talk = [], [], 0.0
        for t in ts:
            pause_while_recording(mid)
            mic = mics.load_track(mid, t)
            if len(mic) < 3 * config.SR:
                continue
            off, drift, q = mics.align(audio, mic, t["start_s"])
            units, cent = mics.own_units(mic, audio, off, drift, embedder())
            t.update(offset_s=round(off, 3), drift=drift, align_quality=round(q, 2), own_units=len(units))
            if cent is None or not units:
                continue
            cvecs.append(cent)
            pcms = [mic[a:b] for a, b in units]
            ws = whisper.transcribe_many(pcms, 8) if whisper else [None] * len(pcms)
            for (a, b), pcm, w in zip(units, pcms, ws):
                text, src, conf = asr.choose(w, ctc.transcribe_long(pcm))
                text = asr.polish(text, cfg)
                if len(text.replace(" ", "")) >= 2:
                    lines.append({"t0": round(mics.to_room(a / config.SR, off, drift), 2), "t1": round(mics.to_room(b / config.SR, off, drift), 2),
                                  "text": text, "src": src, "conf": round(conf, 3), "mic": t["id"]})
                    talk += (b - a) / config.SR
        if not lines:
            stats["skipped"].append(name)
            continue
        cent = speakers.centroid(cvecs)
        owner = max(cents, key=lambda k: float(cent @ cents[k]), default=None)
        if owner is not None and float(cent @ cents[owner]) < mics.OWN_SIM:
            owner = None
        if owner is None:  # the room never heard this person clearly: a new participant
            owner = f"S{max([int(k[1:]) for k in spk if k[1:].isdigit()] + [0]) + 1}"
            spk[owner] = {"name": name, "voice_id": None, "score": None, "suggest": None, "talk_s": round(talk, 1), "centroid": cent.tolist(), "samples": 0}
            cents[owner] = speakers.unit(cent)
        elif not spk[owner].get("name_locked"):
            spk[owner]["name"] = name  # the name the person typed on their phone
        for ln in lines:
            ln["spk"] = owner
        new_lines[name] = (lines, owner)
    if not new_lines:
        store.write_json(d / "mics.json", tracks)
        return stats
    store.write_json(d / "speakers.json", spk)
    import separation

    sr = config.SR
    ovs = separation.find_overlaps(audio, float(cfg.get("overlap_threshold", 0.3))) if separation.overlap_available() else []
    seps = []
    stats["rescued"] = 0

    def transcribe(pcms):
        ws = whisper.transcribe_many(pcms, 8) if whisper else [None] * len(pcms)
        return [(asr.polish(t, cfg), src, conf) for t, src, conf in (asr.choose(w, ctc.transcribe_long(p)) for p, w in zip(pcms, ws))]

    def rescue_for(owner):
        def rescue(s, free, covered):
            """Other people's words in a room line that a phone replaces: where the phone is silent, the room's own audio; where both
            talk at once, the other voice separated from the room audio."""
            pieces, others = [], {k: v for k, v in cents.items() if k != owner}
            for a, b in free:
                pcm = audio[int(a * sr):int(b * sr)]
                e = embedder()(pcm)
                if owner in cents and (not others or float(e @ cents[owner]) >= max(float(e @ v) for v in others.values())):
                    continue  # the phone owner's own voice (their phone missed it, the room line already had it): not a second copy
                who = s["spk"] if s.get("spk") != owner else max(others, key=lambda k: float(e @ others[k]), default=s.get("spk"))
                pieces.append({"a": int(a * sr), "b": int(b * sr), "spk": who, "pcm": pcm})
            if covered and others and separation.separation_available():
                if not seps:
                    seps.append(separation.Separator(threads=8))
                for a, b in covered:
                    for oa, ob in ovs:
                        if oa < b * sr and ob > a * sr:
                            pieces += separation.other_voice(audio, max(oa, int(a * sr)), min(ob, int(b * sr)), owner, seps[0], embedder(), cents)
            got = separation.to_lines(pieces, transcribe, from_room=True)
            stats["rescued"] += len(got)
            return got
        return rescue

    with store.transcript_lock(mid):
        tr = store.read_json(d / "transcript.json") or {"segments": []}
        segs = [s for s in tr.get("segments") or [] if not (s.get("mic") and not (s.get("src") == "edited" or s.get("reviewed") or s.get("history")))]  # done again
        try:
            for name, (lines, owner) in new_lines.items():
                segs, n = mics.merge(segs, lines, owner, rescue_for(owner))
                stats["people"] += 1
                stats["lines"] += len(lines)
                stats["dropped"] += n
        finally:
            for x in seps:
                x.close()
        tr["segments"] = segs
        tr["mics"] = {**stats, "secs": round(time.time() - t0, 1)}
        store.write_json(d / "transcript.json", tr)
    store.write_json(d / "mics.json", tracks)
    store.atomic_write(d / "transcript.md", store.transcript_md(store.load_state(mid), segs, store.speaker_names(mid)))
    stats["secs"] = round(time.time() - t0, 1)
    store.log_event(mid, f"{stats['people']} participant(s) with their own microphone: {stats['lines']} line(s) from their phones, "
                         f"{stats['dropped']} room line(s) replaced ({stats['rescued']} line(s) of other people kept from them), in {stats['secs']:.0f}s" + (f"; no clear own voice from {', '.join(stats['skipped'])}" if stats["skipped"] else ""))
    return stats


def do_speakers(mid: str):
    """Add speaker labels to an existing transcript without repeating the ASR (keeps manual text corrections)."""
    import report
    import speakers

    cfg = store.settings()
    d = store.mdir(mid)
    tr = store.read_json(d / "transcript.json")
    if not tr:
        raise RuntimeError("no transcript to label")
    if not speakers.available():
        raise RuntimeError("speaker model missing")
    if not store.wav_path(mid).exists():
        store.finalize_audio(mid)
    audio = store.load_audio(mid)
    store.update_state(mid, status="analysing", phase="identifying speakers", progress=0.0, eta_s=None, error=None)
    t0 = time.time()
    chunks, spk = speakers.analyse(audio, embedder(), cfg, Eta(mid))
    sr = config.SR
    last = None
    labels = []
    for seg in tr["segments"]:
        a, b = seg["t0"] * sr, seg["t1"] * sr
        best = max(chunks, key=lambda c: max(0, min(b, c[1]) - max(a, c[0])), default=None)
        ov = max(0, min(b, best[1]) - max(a, best[0])) if best else 0
        labels.append(best[2] if best and ov > 0 else last)
        last = labels[-1] or last
    store.write_json(d / "speakers.json", spk)
    with store.transcript_lock(mid):  # re-read: the text may have been corrected while the voices were analysed; only the labels change here
        tr = store.read_json(d / "transcript.json") or tr
        for seg, lab in zip(tr["segments"], labels):
            if not seg.get("mic"):  # a line from a participant's phone keeps its person
                seg["spk"] = lab
        store.write_json(d / "transcript.json", tr)
    store.atomic_write(d / "transcript.md", store.transcript_md(store.load_state(mid), tr["segments"], store.speaker_names(mid)))
    store.log_event(mid, f"identified {len(spk)} speaker(s) in {time.time() - t0:.0f}s")
    ovl = separate_overlaps(mid, audio, cfg)
    mic = merge_mics(mid, audio, cfg)
    report.refresh(mid)
    store.update_state(mid, status="transcribed", phase=None, progress=1.0, attempts=0, queue=("summarize" if cfg["auto_summary"] else None))
    return {"n": len(spk), "mode": "speakers-after" if cfg.get("speakers_after") else "speakers-first", "overlaps": ovl, "mics": mic}


def auto_email(mid: str):
    """Optional: e-mail the result once, right after the summary was written (Settings -> Email -> send automatically)."""
    import mailer

    cfg = mailer.load()
    if not (cfg["enabled"] and cfg["auto_send"]):
        return
    st = store.load_state(mid) or {}
    to, cc = mailer.recipients_for(st.get("owner"), cfg)   # the owner's own recipients, else the shared default ones
    if not to:
        return
    if any(e.get("auto") and e.get("ok") for e in st.get("email_log") or []):
        return  # already sent for this meeting: regenerating the summary does not mail again
    try:
        mailer.send_meeting(mid, to, auto=True, cc=cc)
    except Exception as e:  # noqa: BLE001 - a mail problem must never fail the processing job
        log.warning("automatic e-mail failed: %s", mailer.friendly_error(e, cfg))


def do_summarize(mid: str, only_report: bool = False):
    import report
    import summarize

    cfg = store.settings()
    d = store.mdir(mid)
    tr = store.read_json(d / "transcript.json")
    if not tr:
        raise RuntimeError("no transcript to summarize")
    store.update_state(mid, status="summarizing", phase="report" if only_report else "summary", progress=0.0, eta_s=None, error=None)
    import engines

    chosen = cfg.get("summary_model") or "typhoon-4b"
    need = 4.5  # Typhoon 4B (2.4 GB) plus room for its KV cache; a bigger model that is downloaded waits for its own measured need (30B: 18.6 GB, Gemma 4 E4B: 7.8 GB)
    if (config.SUMMARY_GGUF.get(chosen) or config.LLM_GGUF).exists():
        need = max(need, float(engines.SUMMARY_MEASURED.get(chosen, {}).get("ram_gb", 0)) + 0.5)
    wait_for_ram(need, "starting the summary model")
    t0 = time.time()
    names = store.speaker_names(mid)
    rep = report.refresh(mid)
    segs = tr["segments"]
    import learn

    st_now = store.load_state(mid) or {}
    context = store.context_text(st_now)
    agenda = st_now.get("agenda") or ((st_now.get("bot") or {}).get("agenda") if (st_now.get("bot") or {}).get("enabled") else None) or []
    coverage = learn.agenda_coverage(agenda, segs) if agenda else []
    if coverage:
        store.write_json(d / "agenda_coverage.json", coverage)
    import vocab

    md = None
    pack_ids = vocab.active_ids(store.load_state(mid) or {})
    terms = (learn.trusted_terms(mid) if cfg["use_materials"] else []) + vocab.terms(pack_ids, 40)
    if vocab.contexts(pack_ids):  # the domain descriptions of the meeting's vocabulary packs: background only, never "said in the meeting"
        context = (context + "\nบริบทของสาขาธุรกิจ:\n" + vocab.contexts(pack_ids)).strip()
    with summarize.LlamaServer(on_pause=lambda on: set_rec_pause(mid, on)) as llm:
        store.log_event(mid, f"summary model running on {llm.backend}")
        llm_name, llm_backend = llm.gguf.name.replace(".gguf", ""), llm.backend
        md = None
        if not only_report:
            md = summarize.summarize(llm, mid, segs, names, cfg["glossary"], Eta(mid), context,
                                     terms=terms, coverage=coverage, agenda=agenda, use_refs=bool(cfg["use_materials"] and cfg["summary_use_passages"]))
            if coverage:
                md += "\n\n" + summarize.by_agenda(llm, mid, segs, names, coverage, terms)
            import textnorm

            md = textnorm.polish(md, numbers=bool(cfg.get("numbers_arabic", True)), english=bool(cfg.get("english_words", True)))  # the model may still spell out numbers
            before = (d / "summary.md").read_text(encoding="utf-8") if (d / "summary.md").exists() else None
            store.atomic_write(d / "summary.md", f"# {store.load_state(mid)['title']}\n\n{md}\n")
            if before is not None:  # a regenerated summary replaces the previous text (maybe hand-edited): keep it in the history
                store.add_summary_history(mid, "ai", "AI", before, (d / "summary.md").read_text(encoding="utf-8"))
            (d / "summary.ai.md").unlink(missing_ok=True)  # a new summary replaces a hand-edited one (the UI asks first)
            store.update_state(mid, summary_names_key=report.names_key(mid), summary_edited=False, summary_edited_at=None)
        if rep and rep["overview"]["identified"]:
            store.update_state(mid, phase="report")
            nar = summarize.narratives(llm, mid, segs, names, rep, cfg["glossary"], Eta(mid))  # no meeting context: a person's notes come only from their own lines
            rep["narrative"], rep["narrative_names_key"] = nar, report.names_key(mid)
            rep.pop("narrative_ai", None)  # new notes replace hand-edited ones (the UI asks first)
            store.write_json(d / "report.json", rep)
            store.atomic_write(d / "report.md", report.to_md(mid, rep))
        llm_metrics = llm.summary_metrics()
    (d / "summary.partial.json").unlink(missing_ok=True)
    store.update_state(mid, status="done", phase=None, progress=1.0, queue=None, attempts=0)
    store.log_event(mid, f"{'report' if only_report else 'summary and report'} written in {time.time() - t0:.0f}s")
    if not only_report:
        auto_email(mid)
    info = {"model": llm_name, "backend": llm_backend, "only_report": only_report, **llm_metrics}
    if md:
        info["summary_chars"] = len(md)
        info["decisions_found"], info["tasks_found"] = summary_counts(md)
    return info


def summary_counts(md: str) -> tuple[int, int]:
    """Items under 'มติ / ข้อตัดสินใจ' and 'งานที่ต้องทำ' of the written summary (0 when the section says nothing was found)."""
    import re

    def items(title):
        m = re.search(r"## " + re.escape(title) + r"[^\n]*\n(.*?)(?=\n## |\Z)", md, re.S)
        if not m:
            return 0
        return sum(1 for ln in m.group(1).splitlines() if re.match(r"^\s*(?:\d+[.)]|[-*•])\s+\S", ln))

    return items("มติ") or items("Decisions"), items("งานที่ต้องทำ") or items("Action items")


def recover():
    """After a crash/reboot: jobs that were mid-way go back to their queue and resume from their checkpoints."""
    flag = config.DATA / ".worker_stopped"
    graceful = flag.exists()
    flag.unlink(missing_ok=True)
    for st in store.list_meetings():
        mid = st["id"]
        if graceful and st["status"] in ("transcribing", "analysing", "summarizing"):
            store.update_state(mid, attempts=0)
        if st["status"] == "transcribing":
            store.update_state(mid, status="recorded", queue="transcribe")
        elif st["status"] == "analysing":
            store.update_state(mid, status="transcribed", queue="speakers")
        elif st["status"] == "summarizing":
            store.update_state(mid, status="transcribed", queue="summarize")


LIVE_MAX_S = 25.0  # one Whisper call = one 30 s window, so several short utterances are decoded together


def read_jsonl(path) -> list[dict]:
    import json

    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # torn last line
    return out


def live_step(mid: str, ctc, whisper) -> bool:
    """Accurate-pass one group of finished utterances of a meeting that is still being recorded. True if it did work."""
    import json

    import numpy as np

    import asr

    d = store.mdir(mid)
    live = [c for c in read_jsonl(d / "live.jsonl") if not c.get("mic")]  # (participants' phones are separate tracks: merge_mics)
    done = read_jsonl(d / "whisper.live.jsonl")
    start_t = done[-1]["t1"] if done else 0.0
    cands = [c for c in live if c["t0"] >= start_t - 0.05]
    if not cands:
        return False
    group, closed = [], False
    for c in cands:
        if group and (c["t0"] - group[-1]["t1"] > 1.7 or c["t1"] - group[0]["t0"] > LIVE_MAX_S):
            closed = True
            break
        group.append(c)
    head = store.pcm_seconds(mid)
    span = group[-1]["t1"] - group[0]["t0"]
    # not finished yet: more speech may still join this group (unless it is long enough, or the speaker has stopped for a while)
    if not closed and not (span >= 18 and group[-1].get("end") in ("pause", "dip")) and head - group[-1]["t1"] < 6.0:
        return False
    a, b = max(0, int(group[0]["t0"] * config.SR) - 1600), min(int(head * config.SR), int(group[-1]["t1"] * config.SR) + 1600)
    if b - a < config.SR // 2:
        return False
    with open(store.pcm_path(mid), "rb") as f:
        f.seek(a * 2)
        seg = np.frombuffer(f.read((b - a) * 2), np.int16)
    tt, tc = ctc.transcribe_long(seg)
    w = whisper.transcribe(seg, "")
    text, src, conf = asr.choose(w, (tt, tc))
    rec = {"a": a, "b": b, "t0": round(a / config.SR, 2), "t1": round(b / config.SR, 2), "text": text, "src": src, "spk": None,
           "conf": round(conf, 3), "typhoon": tt, "whisper": w[0] if w else None}
    with open(d / "whisper.live.jsonl", "a", encoding="utf-8") as pf:
        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pf.flush()
        os.fsync(pf.fileno())
    return True


def live_lane():
    """While a meeting is being recorded: the accurate pass runs on utterances that are already finished (cores 4-7, nice 15,
    4 threads, own model copy), so after 'stop' only the last minutes and the gaps remain. Idle (and model freed) otherwise."""
    import asr

    try:
        os.nice(5)  # on top of the worker's 10 (per thread on Linux)
        os.sched_setaffinity(0, range(config.CORES // 2, config.CORES))
    except OSError:
        pass
    ctc = whisper = None
    idle_since = time.time()
    while True:
        cfg = store.settings()
        worked = False
        if cfg["live_accurate"] and cfg["accuracy"] == "accurate" and recording_active():
            for st in store.list_meetings():
                if st["status"] != "recording" or st.get("language") == "en":  # (the live lane's Whisper is the Thai one: an English meeting is done after it ends)
                    continue
                try:
                    if whisper is None:
                        wait_for_ram(3.0, "loading the live Whisper copy")
                        ctc = asr.ThaiCTC(threads=2, npu=bool(cfg.get("asr_npu", True)))
                        whisper = asr.WhisperTH(threads=config.CORES // 2)
                    worked = live_step(st["id"], ctc, whisper) or worked
                except Exception:  # noqa: BLE001 - this is a bonus: the normal pass after the meeting redoes whatever is missing
                    log.warning("live accurate pass: %s", traceback.format_exc().splitlines()[-1])
                    time.sleep(5)
        if worked:
            idle_since = time.time()
            continue
        if whisper is not None and time.time() - idle_since > 60:
            ctc = whisper = None
            import gc

            gc.collect()
            log.info("live accurate pass: model released")
        time.sleep(2)


LANES = {"transcribe": ("transcribe",), "post": ("speakers", "summarize", "report")}


def next_job(stages):
    """The oldest queued meeting of these stages; outside the processing window only a rush meeting (Process now)."""
    import schedule

    cfg = store.settings()
    for st in reversed(store.list_meetings()):  # oldest first
        if st.get("queue") in stages and schedule.may_run(cfg, st):
            return st
    return None


def run_job(mid: str, stage: str, job: dict):
    attempts = job.get("attempts", 0) + 1
    if attempts > MAX_ATTEMPTS:
        store.update_state(mid, status="error", queue=None, error=f"{stage} failed {MAX_ATTEMPTS} times (see history)",
                           failed_stage=stage, attempts=0)
        return
    store.update_state(mid, attempts=attempts)
    log.info("%s %s (attempt %d)", stage, mid, attempts)
    try:
        import stats

        with stats.stage(mid, "transcribe" if stage == "transcribe" else "speakers" if stage == "speakers" else "summary", started_ts=round(time.time(), 1)) as st_:
            try:
                if stage == "transcribe":
                    st_.update(do_transcribe(mid) or {})
                elif stage == "speakers":
                    st_.update(do_speakers(mid) or {})
                else:
                    st_.update(do_summarize(mid, only_report=(stage == "report")) or {})
            except Deferred:
                st_["split"] = True  # done over more than one window: its seconds do not measure the speed (schedule._ratio skips it)
                raise
            finally:
                _pause_prev.pop(mid, None)
                if _paused_s.get(mid):
                    st_["paused_rec_s"] = round(_paused_s.pop(mid), 1)  # stats.stage leaves it out of 'secs'
        if stage in ("summarize", "report"):
            rec = stats.load(mid).get("recording") or {}
            if rec.get("ended_ts"):
                stats.record(mid, "done", finished_at=time.strftime("%F %T"), time_to_result_s=round(time.time() - rec["ended_ts"]),
                             queue_wait_s=round(((stats.load(mid).get("transcribe") or {}).get("started_ts") or rec["ended_ts"]) - rec["ended_ts"]))
    except Deferred:
        import schedule

        cfg = store.settings()
        store.update_state(mid, status="recorded" if stage == "transcribe" else "transcribed", queue=stage, phase=None, eta_s=None, attempts=attempts - 1)
        store.log_event(mid, f"{stage} paused: the processing window closed, it goes on from here at "
                             + time.strftime("%a %H:%M", time.localtime(schedule.next_open(cfg))))
        log.info("%s %s paused until the processing window opens", stage, mid)
    except Exception as e:  # noqa: BLE001 - keep the worker alive, record why
        log.error("job failed: %s", traceback.format_exc())
        store.log_event(mid, f"{stage} error: {e}")
        if stage == "transcribe" and attempts >= MAX_ATTEMPTS or stage != "transcribe":
            store.update_state(mid, status="error", queue=None, error=f"{stage}: {e}", failed_stage=stage)
        time.sleep(2)


def lane(name: str):
    """One worker thread per lane: the transcript is produced by one, everything that follows it (speakers, summary, report)
    by the other, so a long summary never delays the transcription of the next meeting."""
    stages, idle_since = LANES[name], time.time()
    while True:
        job = next_job(stages) if not recording_active() else None
        if not job:
            if time.time() - idle_since > 45:
                (release_models if name == "transcribe" else release_post_models)()
            time.sleep(3)
            continue
        run_job(job["id"], job["queue"], job)
        idle_since = time.time()  # the grace period before releasing models starts when a job ENDS


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(levelname)s %(message)s")
    try:
        os.nice(10)
        open("/proc/self/oom_score_adj", "w").write("800")  # if memory runs out, the worker dies first, not the recorder
        cores = os.environ.get("ZECRET_WORKER_CORES")  # tests: e.g. "4-7"
        lo, _, hi = (cores or f"0-{config.CORES - 1}").partition("-")
        os.sched_setaffinity(0, range(int(lo), int(hi or lo) + 1))  # the worker pauses while a meeting is being recorded and is nice 10, so it may use every core
    except OSError:
        pass
    import signal

    def _term(signum, frame):  # lanes are daemon threads; the llama-server child is stopped by its atexit hook
        (config.DATA / ".worker_stopped").touch()  # a deliberate stop (restart / deploy): the next start must not count it as a failed attempt
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _term)
    parent = os.getppid()
    store.init()
    recover()
    import summarize

    summarize.kill_orphan()
    log.info("worker ready (pid %d, lanes: %s)", os.getpid(), ", ".join(LANES))
    for name in LANES:
        threading.Thread(target=lane, args=(name,), name=f"lane-{name}", daemon=True).start()
    threading.Thread(target=live_lane, name="lane-live", daemon=True).start()
    while True:
        if os.getppid() != parent:
            log.info("server gone, exiting")
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
