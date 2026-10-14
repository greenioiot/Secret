"""Participants' own microphones (their phones, through the join link): one clean track per person.

During the meeting (server.py): Track writes the phone's audio to meetings/<id>/mics/<track>.pcm, kept on the meeting's timeline (a gap
in the phone's audio, e.g. a reconnect, is filled with silence), and listed in mics.json. LiveGate decides which live sentences from a
phone are its owner's (the owner is much louder on their own phone than on the room microphone; other people are not).
LiveMerge drops the room caption of words that a phone already captioned (or replaces it).

After the meeting (worker.merge_mics): align() finds the exact offset (and clock drift) of each track against the room recording,
own_units() keeps the owner's voice only, merge() puts the owner's lines from their phone into the transcript and removes the room
microphone's copy of the same words, while other people's words said at the same time stay (from the room microphone)."""
from __future__ import annotations

import re
import threading
import time

import numpy as np

import config

SR = config.SR
MAX_MICS = 8          # phones sending audio at once in one meeting (each adds live captions on the CPU)
AHEAD_MAX_S = 5.0     # a phone may run this far ahead of the meeting's clock (buffering); more is refused
GAP_FILL_S = 1.0      # a phone more than this far behind the meeting's clock gets silence up to "now" (reconnects, paused tabs)
OWN_DB = 8.0          # live: a sentence quieter (relative to the room) than the owner's usual level by more than this is someone else
OTHER_RUN_S = 1.5     # after: a phone unit is cut only where someone else talks this long (shorter dips stay: cutting words hurts more)
OWN_SIM = 0.45        # after: a phone unit whose voice is at least this close to the owner's voiceprint is the owner's
ALIGN_MAX_S = 3.0     # after: the phone's track may be off by this much (network, buffering)
DUP_SIM = 70          # a room line this similar (fuzzy 0-100) to a phone line at the same time is a copy of the phone owner's words
COVER = 0.5           # a room line of the owner's voice covered this much (time) by their phone lines is replaced by them
COVER_OVL = 0.2       # ... a line the overlap pass separated out of a mix (less clean): already at this much


# ------------------------------------------------------------------ during the meeting
def mdir(mid: str):
    import store

    return store.mdir(mid) / "mics"


def load_index(mid: str) -> list[dict]:
    import store

    return store.read_json(store.mdir(mid) / "mics.json") or []


class Track:
    """One phone's audio for one connection. pos_s = the meeting's recording position now (seconds)."""

    _lock = threading.Lock()

    def __init__(self, mid: str, tid: str, name: str, pos_s: float):
        import store

        self.mid, self.id, self.name = mid, tid, name
        mdir(mid).mkdir(exist_ok=True)
        self.path = mdir(mid) / f"{tid}.pcm"
        self.f = open(self.path, "ab")
        self.start_s = round(max(0.0, pos_s), 3)
        self.n = 0  # samples written
        with Track._lock:
            idx = load_index(mid)
            idx.append({"id": tid, "name": name, "file": self.path.name, "start_s": self.start_s, "opened": time.strftime("%F %T")})
            store.write_json(store.mdir(mid) / "mics.json", idx)

    def write(self, data: bytes, pos_s: float) -> int | None:
        """Append a chunk received now -> samples of silence put before it (a gap), or None when the chunk is refused (the phone sends
        faster than real time)."""
        x = data[: len(data) & ~1]
        due = int((pos_s - self.start_s) * SR)
        if self.n + len(x) // 2 > due + AHEAD_MAX_S * SR:
            return None
        behind = due - len(x) // 2 - self.n
        pad = behind if behind > GAP_FILL_S * SR else 0
        if pad:
            self.f.write(b"\0\0" * pad)
            self.n += pad
        self.f.write(x)
        self.n += len(x) // 2
        return pad

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass


class Ring:
    """The last seconds of the room microphone, to compare a phone sentence with what the room heard at the same time."""

    def __init__(self, seconds: float = 90.0):
        self.buf = np.zeros(int(seconds * SR), np.int16)
        self.end_s = 0.0  # meeting time of the sample after the last one

    def add(self, data: bytes, end_s: float):
        x = np.frombuffer(data[: len(data) & ~1], np.int16)
        if len(x) >= len(self.buf):
            self.buf[:] = x[-len(self.buf):]
        else:
            self.buf = np.roll(self.buf, -len(x))
            self.buf[-len(x):] = x
        self.end_s = end_s

    def rms(self, t0: float, t1: float) -> float | None:
        a = len(self.buf) - int((self.end_s - t0) * SR)
        b = len(self.buf) - int((self.end_s - t1) * SR)
        if a < 0 or b <= a:
            return None
        return float(np.sqrt(np.mean(self.buf[a:b].astype(np.float32) ** 2))) + 1e-3


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float32) ** 2))) + 1e-3 if len(x) else 1e-3


class LiveGate:
    """Is a sentence from a phone its owner's? The owner speaks into it from close by: relative to the room microphone it is much louder
    than other people's voices that the phone also picks up. The owner's usual level is learnt from the loudest sentences so far."""

    def __init__(self):
        self.levels: list[float] = []

    def own(self, pcm, room_rms: float | None) -> bool:
        if room_rms is None:
            return True
        db = 20 * np.log10(_rms(pcm) / room_rms)
        self.levels.append(db)
        self.levels = self.levels[-60:]
        ref = float(np.percentile(self.levels, 80)) if len(self.levels) >= 3 else max(self.levels)
        return db >= ref - OWN_DB


_norm_re = re.compile(r"\s+")


def norm(t: str) -> str:
    return _norm_re.sub("", t or "").lower()


def similar(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return float(fuzz.partial_ratio(a, b)) if min(len(a), len(b)) >= 4 else (100.0 if a == b else 0.0)


class LiveMerge:
    """Recent live captions of one meeting, by source: a phone caption and a room caption of the same words -> keep the phone's."""

    def __init__(self, keep_s: float = 40.0):
        self.items: list[dict] = []
        self.keep_s = keep_s

    def _match(self, rec: dict, phone: bool) -> dict | None:
        for it in reversed(self.items):
            if bool(it.get("mic")) == phone or it.get("dup"):
                continue
            ov = min(rec["t1"], it["t1"]) - max(rec["t0"], it["t0"])
            if ov > 0.3 * min(rec["t1"] - rec["t0"], it["t1"] - it["t0"]) and similar(rec["text"], it["text"]) >= DUP_SIM:
                return it
        return None

    def add(self, rec: dict) -> dict:
        """-> rec with "dup": True (a room copy of a phone caption: not shown) or "replaces": id (a phone caption replacing a room one)."""
        self.items = [it for it in self.items if it["t1"] > rec["t0"] - self.keep_s]
        hit = self._match(rec, bool(rec.get("mic")))
        if hit and rec.get("mic"):
            rec["replaces"] = hit["id"]
            hit["dup"] = True
        elif hit:
            rec["dup"] = True
        self.items.append(rec)
        return rec


# ------------------------------------------------------------------ after the meeting
def load_track(mid: str, t: dict) -> np.ndarray:
    p = mdir(mid) / t["file"]
    return np.fromfile(p, np.int16) if p.exists() else np.zeros(0, np.int16)


def _env(x, hop: int = 320) -> np.ndarray:
    """Log energy per 20 ms, centred: what two recordings of the same room share even when their microphones differ."""
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, np.float32)
    e = np.log(np.mean(np.asarray(x[: n * hop], np.float32).reshape(n, hop) ** 2, axis=1) + 1.0)
    return (e - e.mean()) / (e.std() + 1e-6)


def _lag(room_env: np.ndarray, mic_env: np.ndarray, at: int, max_lag: int) -> tuple[int, float]:
    """Best shift (frames) of mic_env placed at frame `at` of room_env, within +-max_lag -> (lag, correlation)."""
    best, best_c = 0, -1.0
    n = len(mic_env)
    for lag in range(-max_lag, max_lag + 1):
        a = at + lag
        lo, hi = max(0, -a), min(n, len(room_env) - a)
        if hi - lo < 50:
            continue
        r, m = room_env[a + lo:a + hi], mic_env[lo:hi]
        c = float(np.dot(r - r.mean(), m - m.mean()) / ((np.linalg.norm(r - r.mean()) * np.linalg.norm(m - m.mean())) + 1e-6))
        if c > best_c:
            best, best_c = lag, c
    return best, best_c


def align(room, mic, start_s: float, block_s: float = 60.0) -> tuple[float, float, float]:
    """Room time of a phone track's sample 0 and its clock drift: room_t = offset + mic_t * (1 + drift). -> (offset_s, drift, quality)."""
    hop = 320
    re_, me_ = _env(room, hop), _env(mic, hop)
    max_lag, at0 = int(ALIGN_MAX_S * SR / hop), int(start_s * SR / hop)
    blk = int(block_s * SR / hop)
    pts = []
    for b0 in range(0, max(1, len(me_) - blk // 3), blk):
        seg = me_[b0:b0 + blk]
        if len(seg) < 100 or seg.std() < 0.3:
            continue
        lag, c = _lag(re_, seg, at0 + b0, max_lag)
        if c >= 0.3:
            pts.append((b0, lag, c))
    if not pts:
        lag, c = _lag(re_, me_, at0, max_lag)
        return start_s + lag * hop / SR, 0.0, c
    if len(pts) >= 3:
        x = np.array([p[0] for p in pts], np.float64)
        y = np.array([p[1] for p in pts], np.float64)
        k, b = np.polyfit(x, y, 1)
        if abs(k) < 1e-3:  # more than 0.1 % is not a clock: keep a constant offset
            return start_s + b * hop / SR, float(k), float(np.mean([p[2] for p in pts]))
    lag = float(np.median([p[1] for p in pts]))
    return start_s + lag * hop / SR, 0.0, float(np.mean([p[2] for p in pts]))


def to_room(t: float, offset: float, drift: float) -> float:
    return offset + t * (1 + drift)


def _rel_db(mic, room, a: int, b: int, offset: float, drift: float) -> float:
    ra, rb = int(to_room(a / SR, offset, drift) * SR), int(to_room(b / SR, offset, drift) * SR)
    return 20 * np.log10(_rms(mic[a:b]) / _rms(room[max(0, ra):max(0, rb)]))


def own_units(mic, room, offset: float, drift: float, embed, min_s: float = 0.6, step_s: float = 0.5) -> tuple[list[tuple[int, int]], np.ndarray | None]:
    """The phone owner's speech in the track [(a, b) samples] and their voiceprint (None = not enough clear speech).
    1. Every 0.5 s of speech: how much louder the phone is than the room (the owner speaks into it from close by; other people's
       voices reach both microphones more evenly). Stretches far below the owner's usual level are someone else: the units are cut there.
    2. The owner's voiceprint from the stretches loudest on the phone; a stretch is kept when its voice matches it."""
    import speakers

    step = int(step_s * SR)
    units = [(a, b) for a, b in speakers.vad_units(mic) if b - a >= min_s * SR]
    frames = [(a + k, min(b, a + k + step)) for a, b in units for k in range(0, b - a, step)]
    if not frames:
        return [], None
    rel = {f: _rel_db(mic, room, f[0], f[1], offset, drift) for f in frames}
    ref = float(np.percentile(list(rel.values()), 80))
    runs = []
    for a, b in units:  # cut each unit where someone else talks for a while
        grp = None
        for f in ((k, min(b, k + step)) for k in range(a, b, step)):
            if rel[f] < ref - OWN_DB:
                continue
            if grp and f[0] - grp[1] <= OTHER_RUN_S * SR:
                grp[1] = f[1]
            else:
                if grp:
                    runs.append(tuple(grp))
                grp = [f[0], f[1]]
        if grp:
            runs.append(tuple(grp))
    runs = [(a, b) for a, b in runs if b - a >= min_s * SR]
    if not runs:
        return [], None
    level = [float(np.mean([rel[f] for f in frames if f[0] >= a and f[1] <= b] or [-99])) for a, b in runs]
    embs = [embed(mic[a:b]) for a, b in runs]
    top, got, total = [], 0, sum(b - a for a, b in runs)
    for i in np.argsort(-np.array(level)):  # the loudest stretches on the phone, at least 4 s (or 40 % of all)
        if got >= max(4 * SR, 0.4 * total):
            break
        top.append(i)
        got += runs[i][1] - runs[i][0]
    if got < 3 * SR:
        return [], None
    cent = speakers.centroid([embs[i] for i in top])
    return [r for r, e in zip(runs, embs) if float(e @ cent) >= OWN_SIM], cent


def spans(s: dict, lines: list[dict], margin: float = 0.15, min_s: float = 0.6) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """A room line against the phone lines over it -> (stretches the phone does NOT cover, stretches it covers), in seconds."""
    cov = sorted((max(s["t0"], ln["t0"] - margin), min(s["t1"], ln["t1"] + margin)) for ln in lines if ln["t0"] < s["t1"] and ln["t1"] > s["t0"])
    free, pos = [], s["t0"]
    for a, b in cov:
        if a - pos >= min_s:
            free.append((pos, a))
        pos = max(pos, b)
    if s["t1"] - pos >= min_s:
        free.append((pos, s["t1"]))
    return free, cov


def merge(segs: list[dict], lines: list[dict], owner: str | None, rescue=None) -> tuple[list[dict], int]:
    """Room transcript + one person's phone lines -> merged transcript, room lines dropped. A room line goes when it is that person's voice
    and their phone covers it, or when it is a near-copy of a phone line at the same time (the room mixed the voices or mislabelled them).
    What else such a line held (other people talking before, after or over the person) is kept: rescue(line, free, covered) -> new lines
    made from the room audio of those stretches. Lines a person corrected or checked always stay."""
    import separation

    out, dropped, saved = [], 0, []
    for s in segs:
        if s.get("mic") or separation.protected(s):
            out.append(s)
            continue
        dur = max(0.1, s["t1"] - s["t0"])
        near = [ln for ln in lines if ln["t0"] < s["t1"] and ln["t1"] > s["t0"]]
        cover = sum(min(s["t1"], ln["t1"]) - max(s["t0"], ln["t0"]) for ln in near) / dur
        same = similar(s["text"], " ".join(ln["text"] for ln in near)) >= DUP_SIM if near else False
        if (owner and s.get("spk") == owner and cover >= (COVER_OVL if s.get("overlap") else COVER)) or same:
            dropped += 1
            if rescue:
                saved += rescue(s, *spans(s, near))
            continue
        out.append(s)
    for r in saved:  # not when a line that stays already holds that stretch (e.g. a voice the overlap pass separated before)
        held = sum(max(0.0, min(r["t1"], k["t1"]) - max(r["t0"], k["t0"])) for k in out if not k.get("mic"))
        if held < 0.5 * max(0.1, r["t1"] - r["t0"]):
            out.append(r)
    return sorted(out + lines, key=lambda x: x["t0"]), dropped
