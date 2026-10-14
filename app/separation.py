"""Two people talking at the same time.

  1. find_overlaps(): pyannote segmentation-3.0 (ONNX, CPU, already in models/speaker) marks every frame where two voices are active
     at once (its powerset classes "speaker i + speaker j"). 10-s windows with 50% hop, averaged; stretches >= OVL_MIN_S are kept.
  2. Separator: Asteroid Conv-TasNet (Libri2Mix with noise, 16 kHz, exported to ONNX) splits such a stretch into two
     streams. Each stream is then given to the nearest voice of the meeting and
     transcribed on its own, so both people's words are kept instead of one person's words garbled with the other's.
Measured numbers are in PROGRESS.md (Round 41)."""
from __future__ import annotations

import threading
import time

import numpy as np

import config

SEG_MODEL = config.ROOT / "models/speaker/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
SEP_MODEL = config.ROOT / "models/separation/ConvTasNet_Libri2Mix_sepnoisy_16k"
WIN, HOP, SHIFT, FIELD = 160000, 80000, 270, 991
OVL_THR = 0.5      # overlap probability of a frame
OVL_MIN_S = 0.3    # shorter overlaps (a cough, a breath) are not separated
SEP_IDLE_S = 300


def overlap_available() -> bool:
    return SEG_MODEL.exists()


def separation_available() -> bool:
    return (SEP_MODEL / "model.onnx").exists()


_seg = None
_seg_lock = threading.Lock()


def _session():
    global _seg
    with _seg_lock:
        if _seg is None:
            import onnxruntime as ort

            o = ort.SessionOptions()
            o.intra_op_num_threads, o.inter_op_num_threads = 4, 1
            _seg = ort.InferenceSession(str(SEG_MODEL), o, providers=["CPUExecutionProvider"])
        return _seg


def overlap_prob(audio) -> tuple[np.ndarray, float]:
    """Per-frame probability that two voices are active together -> (probs, seconds per frame)."""
    x = np.asarray(audio)
    x = (x.astype(np.float32) / 32768.0) if x.dtype == np.int16 else x.astype(np.float32)
    n = len(x)
    if n == 0:
        return np.zeros(0, np.float32), SHIFT / config.SR
    nfr = n // SHIFT + 1
    acc, cnt = np.zeros(nfr, np.float32), np.zeros(nfr, np.float32)
    s = _session()
    starts = list(range(0, max(1, n - WIN + HOP), HOP))
    for i in range(0, len(starts), 16):
        batch = starts[i:i + 16]
        xs = np.zeros((len(batch), 1, WIN), np.float32)
        for j, a in enumerate(batch):
            seg = x[a:a + WIN]
            xs[j, 0, :len(seg)] = seg
        y = s.run(None, {"x": xs})[0]  # log-probabilities of the 7 powerset classes
        p = np.exp(y[:, :, 4:7]).sum(-1)  # classes 4-6: two speakers at once
        for j, a in enumerate(batch):
            f0 = a // SHIFT
            valid = min(p.shape[1], max(0, (min(n, a + WIN) - a - FIELD) // SHIFT + 1))
            k = min(valid, nfr - f0)
            acc[f0:f0 + k] += p[j, :k]
            cnt[f0:f0 + k] += 1
    return acc / np.maximum(cnt, 1), SHIFT / config.SR


def find_overlaps(audio, thr: float = OVL_THR, min_s: float = OVL_MIN_S) -> list[tuple[int, int]]:
    """[(start, end)] in samples where two people talk at the same time (gaps under 0.3 s are bridged)."""
    p, fs = overlap_prob(audio)
    on = p >= thr
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            runs.append([i, j + 1])
            i = j + 1
        else:
            i += 1
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) * fs < 0.3:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    sr = config.SR
    return [(int(a * fs * sr + FIELD // 2), min(len(audio), int(b * fs * sr + FIELD // 2))) for a, b in merged if (b - a) * fs >= min_s]


class Separator:
    """Conv-TasNet as ONNX (exported here from the Asteroid weights: same output, twice as fast as torch on this CPU, no helper venv).
    Loaded on first use, dropped by close() or after SEP_IDLE_S unused."""

    def __init__(self, threads: int = 4):
        if not separation_available():
            raise RuntimeError(f"voice separation needs {SEP_MODEL / 'model.onnx'}")
        self.threads, self.sess, self.used = threads, None, 0.0
        self.lock = threading.Lock()

    def separate(self, pcm) -> tuple[np.ndarray, np.ndarray]:
        """int16 mono 16 kHz -> two int16 streams of the same length, on ONE scale (a near-empty stream stays quiet)."""
        x = np.asarray(pcm)
        x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
        with self.lock:
            if self.sess is None:
                import onnxruntime as ort

                o = ort.SessionOptions()
                o.intra_op_num_threads, o.inter_op_num_threads = self.threads, 1
                self.sess = ort.InferenceSession(str(SEP_MODEL / "model.onnx"), o, providers=["CPUExecutionProvider"])
            self.used = time.time()
            y = self.sess.run(None, {"x": x[None, :]})[0][0, :, : len(x)]  # [2, time]; the model's output level is arbitrary
            self.used = time.time()
        peak = max(1e-6, float(np.abs(x).max()))
        y = y / max(1e-6, float(np.abs(y).max())) * min(0.9, peak)
        y = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
        return y[0], y[1]

    def close(self):
        with self.lock:
            self.sess = None

    def release_idle(self):
        if self.sess is not None and time.time() - self.used > SEP_IDLE_S:
            self.close()


# ------------------------------------------------------------------ the pass over a finished transcript
DEBUG = False
PIECE_S = 24.0     # a window is separated in pieces of at most this length
QUIET = 0.05       # a separated stream under 5% of the other one's loudness is the empty side (one person talking)
LOUD2 = 0.30       # a second stream at least this loud is a second voice even when its voiceprint is unclear
OTHER_MIN = 0.15   # a quieter stream is kept only when it sounds like ANOTHER voice of the meeting than the louder one (cosine)
SPAN_MIN_S = 0.25  # shortest piece of speech kept from a stream (a quick "ใช่" is about 0.3 s)
INSERT_SHARE = 0.35  # a window where people overlap for less than this share is kept, and the other voice's words are ADDED as their own line
INSERT_MAX_S = 2.5  # ... and only when every overlap in it is at most this long (a remark, not two people talking in parallel)
PAD_S = 0.5        # sound kept around a short overlap when it is separated on its own
ECHO_MARGIN = 0.15  # an added line whose voice is this much closer to its own person than to the main speaker is always kept;
ECHO_MIN = 65      # otherwise it is the main voice leaking into the other stream when its words are already in the line (fuzzy, 0-100) ...
ECHO_FLOOR = -0.05  # ... or when its voice is closer to the main speaker's
LEFT_MIN = 6       # characters of the mixed text missing from all separated lines before they are shown as "also heard"


def protected(seg: dict) -> bool:
    """Lines a person corrected or checked are never replaced by automatic passes."""
    return bool(seg.get("reviewed") or seg.get("history") or seg.get("src") == "edited")


def windows(segs: list[dict], overlaps: list[tuple[int, int]]) -> list[tuple[float, float, list[int]]]:
    """Transcript lines touched by each overlap, grouped: [(t0, t1, [segment indexes])]. Groups sharing a line are joined."""
    sr = config.SR
    groups: list[list[int]] = []
    for a, b in overlaps:
        ta, tb = a / sr, b / sr
        idx = [i for i, s in enumerate(segs) if s["t0"] < tb and s["t1"] > ta]
        if not idx:
            continue
        if groups and set(idx) & set(groups[-1]):
            groups[-1] = sorted(set(groups[-1]) | set(idx))
        else:
            groups.append(idx)
    return [(min(segs[i]["t0"] for i in g), max(segs[i]["t1"] for i in g), g) for g in groups]


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float32) ** 2))) if len(x) else 0.0


def speech_span(pcm, frame: int = 320) -> tuple[int, int] | None:
    """First and last loud frame of a stream (relative samples), or None."""
    n = len(pcm) // frame
    if n == 0:
        return None
    e = np.sqrt((np.asarray(pcm[: n * frame], np.float32).reshape(n, frame) ** 2).mean(1))
    on = np.nonzero(e > max(150.0, 0.15 * float(e.max())))[0]
    return (int(on[0]) * frame, int(on[-1] + 1) * frame) if len(on) else None


def assign(e1, e2, cents: dict[str, np.ndarray]) -> tuple[str, str]:
    """The pair of different voices that fits the two streams best."""
    ids = list(cents)
    best, pair = -9.0, (ids[0], ids[min(1, len(ids) - 1)])
    for i in ids:
        for j in ids:
            if i != j:
                s = float(e1 @ cents[i]) + float(e2 @ cents[j])
                if s > best:
                    best, pair = s, (i, j)
    return pair


def _voice(e, cents: dict[str, np.ndarray]) -> tuple[str, float]:
    k = max(cents, key=lambda c: float(e @ cents[c]))
    return k, float(e @ cents[k])


def second_voice(s1, s2, embed, cents: dict[str, np.ndarray], ratio_min: float = QUIET, other_min: float = OTHER_MIN):
    """Two separated streams -> [(stream, voice id)] for the streams that hold a real voice (1 or 2 items, [] for silence).
    A quiet stream is a second person only when it sounds like a different voice of the meeting than the loud one: separating one
    voice leaves a faint copy of that same voice in the other stream."""
    r1, r2 = _rms(s1), _rms(s2)
    if max(r1, r2) <= 0:
        return []
    (loud, rl), (quiet, rq) = ((s1, r1), (s2, r2)) if r1 >= r2 else ((s2, r2), (s1, r1))
    el = embed(loud)
    if rq >= LOUD2 * rl:
        a, b = assign(el, embed(quiet), cents)
        return [(loud, a), (quiet, b)]
    vl, _ = _voice(el, cents)
    if rq >= ratio_min * rl:
        eq = embed(quiet)
        others = {k: v for k, v in cents.items() if k != vl}
        if others:
            vq, sim = _voice(eq, others)
            if sim >= other_min and sim > float(eq @ cents[vl]):
                return [(loud, vl), (quiet, vq)]
    return [(loud, vl)]


def split_streams(audio, t0: float, t1: float, sep: "Separator", embed, cents: dict[str, np.ndarray]) -> list[dict]:
    """Separate [t0, t1] (in pieces <= PIECE_S) -> pieces of one voice each: [{"a", "b" (samples, absolute), "spk", "pcm"}]."""
    sr = config.SR
    a0, a1 = int(t0 * sr), min(len(audio), int(t1 * sr))
    n = max(1, int(np.ceil((a1 - a0) / (PIECE_S * sr))))
    step = (a1 - a0) // n
    out = []
    for k in range(n):
        pa, pb = a0 + k * step, (a1 if k == n - 1 else a0 + (k + 1) * step)
        s1, s2 = sep.separate(audio[pa:pb])
        for s, sid in second_voice(s1, s2, embed, cents):
            span = speech_span(s)
            if span and span[1] - span[0] >= SPAN_MIN_S * sr:
                out.append({"a": pa + span[0], "b": pa + span[1], "spk": sid, "pcm": s[span[0]:span[1]]})
    return out


def other_voice(audio, a: int, b: int, main: str | None, sep: "Separator", embed, cents: dict[str, np.ndarray],
                ratio_min: float = QUIET, other_min: float = OTHER_MIN) -> list[dict]:
    """A short stretch [a, b) (samples) inside a line of voice `main`: separate it (with PAD_S around) -> the pieces of OTHER voices only."""
    sr = config.SR
    pa, pb = max(0, a - int(PAD_S * sr)), min(len(audio), b + int(PAD_S * sr))
    s1, s2 = sep.separate(audio[pa:pb])
    out = []
    for s, sid in second_voice(s1, s2, embed, cents, ratio_min, other_min):
        span = speech_span(s)
        if sid != main and span and span[1] - span[0] >= SPAN_MIN_S * sr:
            out.append({"a": pa + span[0], "b": pa + span[1], "spk": sid, "pcm": s[span[0]:span[1]]})
    return out


def to_lines(pieces: list[dict], transcribe, **extra) -> list[dict]:
    """Separated pieces -> transcript lines (pieces that give no text are dropped)."""
    sr = config.SR
    out = []
    for p, (text, src, conf) in zip(pieces, transcribe([p["pcm"] for p in pieces]) if pieces else []):
        if len(text.replace(" ", "")) >= 2:
            out.append({"t0": round(p["a"] / sr, 2), "t1": round(p["b"] / sr, 2), "text": text, "src": src, "conf": round(conf, 3), "spk": p["spk"], "overlap": True, **extra})
    return sorted(out, key=lambda x: x["t0"])


def plan(segs: list[dict], overlaps: list[tuple[int, int]]) -> list[dict]:
    """What to do with each window: {"lo", "hi", "mode": "replace" | "insert", "spans": [(a, b) samples]}.
    replace = the people overlap for much of it: the lines are rebuilt, one per voice; insert = a short overlap inside a longer line:
    the line is kept and the other voice's words are added."""
    sr = config.SR
    out = []
    for t0, t1, g in windows(segs, overlaps):
        spans = [(a, b) for a, b in overlaps if a / sr < t1 and b / sr > t0]
        share = sum(min(b / sr, t1) - max(a / sr, t0) for a, b in spans) / max(0.1, t1 - t0)
        short = share < INSERT_SHARE and all((b - a) / sr <= INSERT_MAX_S for a, b in spans)
        out.append({"lo": min(g), "hi": max(g), "mode": "insert" if short else "replace", "spans": spans})
    return out


def leftover(mixed: list[dict], new: list[dict]) -> str:
    """Words recognised in the mixed sound that none of the separated lines has (so they are not lost silently)."""
    import difflib

    m = "".join(" ".join(x.get("text") or "" for x in mixed).split())
    t = "".join(" ".join(x["text"] for x in new).split())
    if not m:
        return ""
    sm = difflib.SequenceMatcher(None, m, t, autojunk=False)
    runs, pos = [], 0
    for blk in sm.get_matching_blocks():
        if blk.a - pos >= LEFT_MIN:
            runs.append(m[pos:blk.a].lstrip("\u0e31\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e3a\u0e47\u0e48\u0e49\u0e4a\u0e4b\u0e4c\u0e4d\u0e4e"))
        pos = blk.a + blk.size
    return " … ".join(runs)


def resolve_window(audio, segs: list[dict], lo: int, hi: int, sep: "Separator", embed, cents: dict[str, np.ndarray], transcribe) -> list[dict]:
    """"replace": lines lo..hi -> new lines, one per voice (empty list = leave the window as it is).
    transcribe(list of int16 pieces) -> [(text, src, conf)]."""
    wa, wb = min(segs[i]["t0"] for i in range(lo, hi + 1)), max(segs[i]["t1"] for i in range(lo, hi + 1))
    new = to_lines(split_streams(audio, wa, wb, sep, embed, cents), transcribe)
    if not new:
        return []
    mixed = []  # what was recognised from the mixed sound (kept from an earlier pass when the voices are analysed again)
    for i in range(lo, hi + 1):
        mixed += segs[i].get("mixed") or [] if segs[i].get("overlap") else [{k: segs[i].get(k) for k in ("t0", "t1", "text", "spk")}]
    new[0]["mixed"] = mixed
    if left := leftover(mixed, new):
        new[0]["leftover"] = left
    return new


def insert_window(audio, segs: list[dict], lo: int, hi: int, spans: list[tuple[int, int]], sep: "Separator", embed, cents: dict[str, np.ndarray], transcribe) -> list[dict]:
    """"insert": the other voices' words in short overlaps inside lines lo..hi -> new lines to ADD (the lines stay).
    Separation leaves a faint copy of the main voice in the other stream; such an "echo" is recognised by its voice (not clearly the
    other person's) together with its words (already in the line, or the voice is closer to the main speaker's)."""
    from rapidfuzz import fuzz

    import textnorm

    sr = config.SR
    norm = lambda t: "".join(textnorm.polish(t, False, False).split())  # noqa: E731
    pieces = []
    for a, b in spans:
        line = max(range(lo, hi + 1), key=lambda i: min(b / sr, segs[i]["t1"]) - max(a / sr, segs[i]["t0"]))
        pieces += [{**p, "main": segs[line].get("spk")} for p in other_voice(audio, a, b, segs[line].get("spk"), sep, embed, cents)]
    out = []
    for p, ln in zip(pieces, [to_lines([p], transcribe, inserted=True) for p in pieces]):
        if not ln:
            continue
        ln = ln[0]
        e = embed(p["pcm"])
        margin = float(e @ cents[p["spk"]]) - (float(e @ cents[p["main"]]) if p["main"] in cents else 0.0)
        under = " ".join(segs[i]["text"] for i in range(lo, hi + 1) if segs[i]["t0"] < ln["t1"] + 1 and segs[i]["t1"] > ln["t0"] - 1)
        echo = fuzz.partial_ratio(norm(ln["text"]), norm(under))
        if DEBUG:
            ln["_echo"], ln["_margin"] = round(echo), round(margin, 2)
        if margin >= ECHO_MARGIN or (echo < ECHO_MIN and margin >= ECHO_FLOOR):
            out.append(ln)
    return out
