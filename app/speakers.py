"""Who is speaking. Offline, on the CPU.

Method (measured on a 3-voice test conversation: every utterance went to the right speaker, same-voice cosine 0.82-0.93
vs <= 0.38 between voices, while sherpa's own pyannote clustering split it into 4-5 clusters and mislabelled turn edges):
  1. pause-delimited utterances (Silero VAD, 400 ms)  2. one CAM++ voice embedding per utterance
  3. average-linkage clustering of those embeddings (cosine)  4. long utterances are split where the voice clearly changes
  5. clusters are matched to enrolled voice profiles (data/voices/*.json)
Voice matching of Thai speech has not been validated on real recordings: the UI shows the score and lets the user confirm.
"""
from __future__ import annotations

import json
import threading
import time
import uuid

import numpy as np

import config
import store

SR = config.SR
MIN_EMB_S = 0.8           # shorter utterances are labelled by their nearest cluster instead of being clustered
WIN_S, HOP_S = 2.5, 1.25  # windows used to look for a speaker change inside long utterances
VOICE_MAX_EMB = 20


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class Embedder:
    def __init__(self, threads: int = 2):
        import sherpa_onnx

        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(config.SPEAKER_EMB), num_threads=threads))
        self.lock = threading.Lock()

    def __call__(self, pcm) -> np.ndarray:
        x = np.asarray(pcm, np.float32)
        if x.dtype != np.float32 or np.abs(x).max(initial=0) > 1.5:
            x = x / 32768.0
        with self.lock:
            s = self.ex.create_stream()
            s.accept_waveform(SR, x)
            s.input_finished()
            return unit(self.ex.compute(s))


def available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return config.SPEAKER_EMB.exists()


# ------------------------------------------------------------------ utterances
def vad_units(audio, win_s: int = 15 * 60) -> list[tuple[int, int]]:
    """Pause-delimited speech units [(start_sample, end_sample)], computed in windows to bound RAM."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=0.45, min_speech_duration_ms=250, min_silence_duration_ms=400, speech_pad_ms=120,
                      max_speech_duration_s=40)
    n, pos, out = len(audio), 0, []
    win = win_s * SR
    while pos < n:
        end = min(n, pos + win)
        found = [(pos + s["start"], pos + s["end"]) for s in get_speech_timestamps(np.asarray(audio[pos:end], np.float32) / 32768.0, opts)]
        if end < n and found and end - found[-1][1] < SR:  # speech runs into the window edge: redo it in the next window
            end = found.pop()[0]
            if end <= pos:
                end = min(n, pos + win)
        out += found
        pos = end
    return out


# ------------------------------------------------------------------ clustering
def cluster(E: np.ndarray, threshold: float = 0.5, k: int = 0) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine similarity. Stops when the best pair is below `threshold`
    (or at k clusters when k > 0)."""
    n = len(E)
    if n == 0:
        return np.zeros(0, int)
    S = E @ E.T
    np.fill_diagonal(S, -9)
    size = np.ones(n)
    members = {i: [i] for i in range(n)}
    alive = np.ones(n, bool)
    while alive.sum() > 1:
        masked = np.where(alive[:, None] & alive[None, :], S, -9)
        i, j = np.unravel_index(int(masked.argmax()), masked.shape)
        best = masked[i, j]
        if k > 0:
            if alive.sum() <= k:
                break
        elif best < threshold:
            break
        # merge j into i: UPGMA update
        new = (S[i] * size[i] + S[j] * size[j]) / (size[i] + size[j])
        S[i, :] = new
        S[:, i] = new
        S[i, i] = -9
        size[i] += size[j]
        members[i] += members.pop(j)
        alive[j] = False
        S[j, :] = -9
        S[:, j] = -9
    labels = np.zeros(n, int)
    for c, i in enumerate(sorted(members, key=lambda m: min(members[m]))):  # numbered by first appearance
        for m in members[i]:
            labels[m] = c
    return labels


def centroid(vs) -> np.ndarray:
    return unit(np.mean(np.asarray(vs, np.float32), axis=0))


# ------------------------------------------------------------------ voice library
def _vfile(vid: str):
    return config.VOICES / f"{vid}.json"


def load_voices() -> list[dict]:
    config.VOICES.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(config.VOICES.glob("*.json")):
        v = store.read_json(f)
        if v and v.get("embeddings"):
            out.append(v)
    return out


def public_voice(v: dict) -> dict:
    return {"id": v["id"], "name": v["name"], "samples": len(v["embeddings"]), "created": v.get("created"), "updated": v.get("updated")}


def add_voice_samples(name: str, embs, voice_id: str | None = None, source: str = "") -> dict:
    """Add embeddings to a profile (by id, else by case-insensitive name, else a new profile)."""
    name = name.strip()
    _private(config.VOICES)
    voices = load_voices()
    v = next((x for x in voices if voice_id and x["id"] == voice_id), None) or next((x for x in voices if x["name"].casefold() == name.casefold()), None)
    now = time.strftime("%F %T")
    if v is None:
        v = {"id": "v" + uuid.uuid4().hex[:8], "name": name, "embeddings": [], "created": now}
    if name:
        v["name"] = name
    v["embeddings"] = (v["embeddings"] + [np.asarray(e, np.float32).round(5).tolist() for e in embs])[-VOICE_MAX_EMB:]
    v["updated"] = now
    v.setdefault("sources", []).append(source)
    v["sources"] = v["sources"][-10:]
    store.write_json(_vfile(v["id"]), v)
    return v


# ------------------------------------------------------------------ voices sent by participants (wait for approval)
def _private(d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)   # voiceprints are biometric: readable by this user only
    except OSError:
        pass


def add_pending(name: str, embs, meta: dict) -> dict:
    _private(config.PENDING)
    p = {"id": "p" + uuid.uuid4().hex[:8], "name": name.strip()[:40], "embeddings": [np.asarray(e, np.float32).round(5).tolist() for e in embs],
         "time": time.strftime("%F %T"), **meta}
    store.write_json(config.PENDING / f"{p['id']}.json", p)
    return p


def load_pending() -> list[dict]:
    out = []
    for f in sorted(config.PENDING.glob("p*.json")) if config.PENDING.exists() else []:
        p = store.read_json(f)
        if p and p.get("embeddings"):
            out.append(p)
    return out


def get_pending(pid: str) -> dict | None:
    import re

    return store.read_json(config.PENDING / f"{pid}.json") if re.fullmatch(r"p[0-9a-f]{8}", pid) else None


def delete_pending(pid: str) -> bool:
    import re

    f = config.PENDING / f"{pid}.json"
    if re.fullmatch(r"p[0-9a-f]{8}", pid) and f.exists():
        f.unlink()
        return True
    return False


def best_match(embs, exclude_name: str = "") -> dict | None:
    """Which enrolled voice a new voiceprint sounds most like (helps the host spot duplicates and impersonation)."""
    voices = load_voices()
    if not voices:
        return None
    c = centroid(embs)
    scored = [(float(voice_centroid(v) @ c), v["name"]) for v in voices]
    s, n = max(scored)
    return {"name": n, "score": round(s, 3)}


def approve_pending(pid: str) -> dict | None:
    """The host accepted this voice: it joins the voice of the same name (or becomes a new one). Consent is kept in the profile."""
    p = get_pending(pid)
    if not p:
        return None
    v = add_voice_samples(p["name"], p["embeddings"], source=f"participant link {p['time']}")
    v.setdefault("consents", []).append({"time": p["time"], "approved": time.strftime("%F %T"), "by": "participant link", "consent": bool(p.get("consent"))})
    v["consents"] = v["consents"][-10:]
    store.write_json(_vfile(v["id"]), v)
    delete_pending(pid)
    return v


def voice_centroid(v: dict) -> np.ndarray:
    return centroid(v["embeddings"])


def match_voices(cents: dict[str, np.ndarray], threshold: float) -> dict[str, dict]:
    """cluster id -> {voice_id, name, score, confident}. Greedy one-to-one by score; below the threshold it is only a suggestion."""
    voices = load_voices()
    if not voices or not cents:
        return {}
    pairs = []
    for sid, c in cents.items():
        for v in voices:
            sims = np.asarray(v["embeddings"], np.float32) @ c
            # centroid similarity, lifted slightly by the best single sample (voices vary with mood and microphone)
            score = float(0.7 * (voice_centroid(v) @ c) + 0.3 * sims.max())
            pairs.append((score, sid, v))
    pairs.sort(key=lambda p: -p[0])
    used_s, used_v, out = set(), set(), {}
    for score, sid, v in pairs:
        if sid in used_s or v["id"] in used_v or score < threshold - 0.15:
            continue
        used_s.add(sid)
        used_v.add(v["id"])
        out[sid] = {"voice_id": v["id"], "name": v["name"], "score": round(score, 3), "confident": score >= threshold}
    return out


# ------------------------------------------------------------------ offline analysis of one meeting
def analyse(audio, embed: Embedder, cfg: dict, progress=lambda f: None) -> tuple[list[list], dict]:
    """-> (chunks [[start, end, "S1"], ...] ready for ASR, speakers {"S1": {...}})"""
    units = vad_units(audio)
    if not units:
        return [], {}
    progress(0.05)
    ovl = overlap_mask(audio, units, cfg)
    embs, ok = [], []
    for i, (a, b) in enumerate(units):
        if (b - a) / SR >= MIN_EMB_S and not ovl[i]:
            embs.append(embed(audio[a:b]))
            ok.append(i)
        if i % 20 == 0:
            progress(0.05 + 0.5 * i / len(units))
    thr = float(cfg.get("speaker_threshold", 0.5))
    k = int(cfg.get("expected_speakers", 0) or 0)
    if not embs:
        labels = {}
    else:
        lab = cluster(np.array(embs), thr, k)
        labels = {u: int(l) for u, l in zip(ok, lab)}
    if embs and not k:
        labels = absorb_small(labels, units, ok, embs)
    ncl = (max(labels.values()) + 1) if labels else 1
    cents = [centroid([e for e, u in zip(embs, ok) if labels[u] == c]) for c in range(ncl)] if embs else [np.zeros(192, np.float32)]
    # split long utterances where the voice clearly changes; label short ones by the nearest centroid
    pieces: list[tuple[int, int, int, np.ndarray | None]] = []
    for i, (a, b) in enumerate(units):
        dur = (b - a) / SR
        if i not in labels:
            if embs and dur >= 0.3:
                e = embed(audio[a:b])
                near = int(np.argmax([e @ c for c in cents]))
                if ovl[i] and dur >= 6.0 and len(cents) > 1:  # two voices in a long utterance: cut it where the voice changes
                    pieces += [(pa, pb, lbl, None) for pa, pb, lbl in split_unit(audio, a, b, near, cents, embed)]
                else:
                    pieces.append((a, b, near, e))
            else:
                pieces.append((a, b, -1, None))
            continue
        main = labels[i]
        for pa, pb, lbl in split_unit(audio, a, b, main, cents, embed) if dur >= 6.0 and len(cents) > 1 else [(a, b, main)]:
            pieces.append((pa, pb, lbl, None))
        progress(0.55 + 0.35 * (i + 1) / len(units))
    # unlabeled fragments take the neighbour's label (sub-0.3 s blips)
    last = next((p[2] for p in pieces if p[2] >= 0), 0)
    fixed = []
    for a, b, lbl, e in pieces:
        last = lbl if lbl >= 0 else last
        fixed.append([a, b, last])
    # speaker ids by first appearance: S1, S2 ...
    order = {}
    for a, b, l in fixed:
        order.setdefault(l, len(order) + 1)
    chunks = merge_for_asr([[a, b, f"S{order[l]}"] for a, b, l in fixed])
    ovl_at = lambda a, b: any(units[i][0] < b and units[i][1] > a for i in range(len(units)) if ovl[i])  # noqa: E731
    spk = {}
    for l, n in order.items():
        sid = f"S{n}"
        mine = [(b - a) for a, b, ll in fixed if ll == l]
        # keep a few of this speaker's longest utterance embeddings: they are what "remember this voice" stores
        longest = sorted([(a, b) for a, b, ll in fixed if ll == l and not ovl_at(a, b)], key=lambda p: p[0] - p[1])[:5]  # never a piece with two voices
        samples = [embed(audio[a:min(b, a + 12 * SR)]).round(5).tolist() for a, b in longest if (b - a) / SR >= 1.5]
        c = centroid(samples) if samples else (cents[l] if l < len(cents) else cents[0])
        spk[sid] = {"name": None, "voice_id": None, "score": None, "suggest": None, "talk_s": round(sum(mine) / SR, 1),
                    "centroid": c.round(5).tolist(), "samples": samples}
    apply_matches(spk, float(cfg.get("voice_match_threshold", 0.5)))
    progress(1.0)
    return chunks, spk


def overlap_mask(audio, units, cfg: dict, share: float = 0.10, min_s: float = 1.0) -> list[bool]:
    """Utterances where two people talk at once (>= 1 s or >= 10% of it: the detector finds only the core of an overlap, about half). Their voiceprint sounds like both people, so it would chain two
    speakers into one cluster: they are left out of the clustering and labelled afterwards (measured: 2 of 12 test conversations had
    their two people merged into one before this). All False when the overlap model is missing or every utterance overlaps."""
    if not cfg.get("separate_overlaps", True):
        return [False] * len(units)
    try:
        import separation

        if not separation.overlap_available():
            return [False] * len(units)
        ovs = separation.find_overlaps(audio, float(cfg.get("overlap_threshold", 0.3)), min_s=0.3)
    except Exception:  # noqa: BLE001 - speaker labels must not depend on the overlap model
        return [False] * len(units)
    mask = []
    for a, b in units:
        inter = sum(max(0, min(b, y) - max(a, x)) for x, y in ovs)
        mask.append(inter >= min(share * (b - a), min_s * SR))
    long_ok = [((b - a) / SR >= MIN_EMB_S) and not m for (a, b), m in zip(units, mask)]
    return mask if any(long_ok) else [False] * len(units)


def absorb_small(labels: dict[int, int], units, ok, embs, min_s: float = 5.0, min_share: float = 0.04) -> dict[int, int]:
    """A "speaker" with under 5 s (or 4%) of the speech is almost always a fragment, noise or a cough: fold its
    utterances into the closest real speaker instead of listing a phantom person."""
    dur: dict[int, float] = {}
    for u, l in labels.items():
        dur[l] = dur.get(l, 0.0) + (units[u][1] - units[u][0]) / SR
    total = sum(dur.values()) or 1.0
    big = [l for l, v in dur.items() if v >= max(min_s, min_share * total)]
    if not big or len(big) == len(dur):
        return labels
    cents = {l: centroid([e for e, u in zip(embs, ok) if labels[u] == l]) for l in big}
    out = dict(labels)
    for e, u in zip(embs, ok):
        if labels[u] not in cents:
            out[u] = max(cents, key=lambda l: float(e @ cents[l]))
    remap = {l: i for i, l in enumerate(sorted(set(out.values())))}
    return {u: remap[l] for u, l in out.items()}


def split_unit(audio, a: int, b: int, main: int, cents: list[np.ndarray], embed: Embedder) -> list[tuple[int, int, int]]:
    """Cut a long utterance where >= 2 consecutive windows are clearly closer to another speaker than to the main one."""
    w, h = int(WIN_S * SR), int(HOP_S * SR)
    starts = list(range(a, max(a + 1, b - w + 1), h))
    lbl = []
    for s in starts:
        e = embed(audio[s:s + w])
        sims = [float(e @ c) for c in cents]
        best = int(np.argmax(sims))
        lbl.append(best if best != main and sims[best] - sims[main] > 0.15 else main)
    cuts, i = [], 0
    while i < len(lbl):
        if lbl[i] != main:
            j = i
            while j + 1 < len(lbl) and lbl[j + 1] == lbl[i]:
                j += 1
            if j > i:  # at least two windows in a row
                cuts.append((starts[i] + int(0.4 * SR), starts[j] + w - int(0.2 * SR), lbl[i]))
            i = j + 1
        else:
            i += 1
    if not cuts:
        return [(a, b, main)]
    out, pos = [], a
    for ca, cb, l in cuts:
        if ca - pos > 0.6 * SR:
            out.append((pos, ca, main))
        out.append((max(ca, pos), min(cb, b), l))
        pos = min(cb, b)
    if b - pos > 0.6 * SR:
        out.append((pos, b, main))
    return out


def merge_for_asr(parts: list[list], max_s: float = 28.0, gap_s: float = 1.2) -> list[list]:
    """Neighbouring pieces of the same speaker become one ASR chunk (<= 28 s, Whisper's window)."""
    out: list[list] = []
    for a, b, s in parts:
        if out and out[-1][2] == s and a - out[-1][1] < gap_s * SR and b - out[-1][0] <= max_s * SR:
            out[-1][1] = b
        else:
            out.append([a, b, s])
    return out


def apply_matches(spk: dict, threshold: float):
    """Fill name/voice_id (confident match) or suggest (weaker) from the enrolled voices. Never overwrites a user's choice."""
    cents = {sid: unit(v["centroid"]) for sid, v in spk.items() if not v.get("name_locked")}
    for sid, m in match_voices(cents, threshold).items():
        v = spk[sid]
        if m["confident"] and not v.get("name_locked"):
            v.update(name=m["name"], voice_id=m["voice_id"], score=m["score"], suggest=None)
        else:
            v.update(suggest={"name": m["name"], "voice_id": m["voice_id"], "score": m["score"]})


# ------------------------------------------------------------------ live (provisional) labels
class LiveSpeakers:
    """Online labelling for live captions: enrolled voices first, otherwise "ผู้เข้าร่วม N" by running centroids."""

    def __init__(self, embed: Embedder, threshold: float = 0.5, match_threshold: float = 0.5):
        self.embed, self.thr, self.mthr = embed, threshold, match_threshold
        self.voices = [(v["name"], voice_centroid(v)) for v in load_voices()]
        self.cents: list[list] = []   # [sum_vector, count]
        self.last = None

    def label(self, pcm) -> str | None:
        if len(pcm) < MIN_EMB_S * SR:
            return self.last
        e = self.embed(pcm)
        if self.voices:
            sims = [float(e @ c) for _, c in self.voices]
            i = int(np.argmax(sims))
            if sims[i] >= self.mthr:
                self.last = self.voices[i][0]
                return self.last
        sims = [float(e @ unit(s / n)) for s, n in self.cents]
        if sims and max(sims) >= self.thr:
            i = int(np.argmax(sims))
            self.cents[i][0] = self.cents[i][0] + e
            self.cents[i][1] += 1
        else:
            self.cents.append([e.copy(), 1])
            i = len(self.cents) - 1
        self.last = f"ผู้เข้าร่วม {i + 1}"
        return self.last
