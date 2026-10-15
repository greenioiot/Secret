"""File-based meeting store. Every meeting is a folder; state.json is the single source of truth, written atomically,
so after a crash or reboot each stage can resume from what is on disk."""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import time
from datetime import datetime
from pathlib import Path

import config

STAGES = ("recorded", "transcribed", "done")


def atomic_write(path: Path, data: str | bytes):
    tmp = path.with_name(path.name + ".tmp")
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(tmp, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj):
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1))


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def init():
    config.MEETINGS.mkdir(parents=True, exist_ok=True)


def settings() -> dict:
    return {**config.DEFAULT_SETTINGS, **(read_json(config.SETTINGS_FILE, {}) or {})}


def enroll_text() -> str:
    return settings().get("enroll_text") or config.ENROLL_TEXT


def save_settings(new: dict) -> dict:
    cur = settings()
    for k, v in new.items():
        d = config.DEFAULT_SETTINGS.get(k)
        if isinstance(d, float) and type(v) is int:
            v = float(v)
        if k in config.DEFAULT_SETTINGS and type(v) is type(d):
            cur[k] = v
    cur["guide_name"] = " ".join(str(cur["guide_name"]).split())[:30] or config.DEFAULT_SETTINGS["guide_name"]
    cur["guide_wifi"] = " ".join(str(cur["guide_wifi"]).split())[:200]
    if cur["guide_lang"] not in ("auto", "th", "en"):
        cur["guide_lang"] = "auto"
    if cur.get("language") not in config.LANGUAGES:
        cur["language"] = config.DEFAULT_SETTINGS["language"]
    cur["handsfree_wake"] = re.sub(r"[^\w\u0e00-\u0e7f -]", "", str(cur["handsfree_wake"])).strip()[:20] or config.DEFAULT_SETTINGS["handsfree_wake"]
    cur["enroll_text"] = " ".join(str(cur["enroll_text"]).split())[:config.ENROLL_TEXT_MAX]
    if cur["enroll_text"] == config.ENROLL_TEXT:
        cur["enroll_text"] = ""  # the default is not frozen into settings.json: a better default later reaches everyone who never edited it
    import schedule

    for k in ("batch_start", "batch_end"):
        if not schedule.valid_hm(cur[k]):
            cur[k] = config.DEFAULT_SETTINGS[k]
        h, m = str(cur[k]).split(":")
        cur[k] = f"{int(h):02d}:{m}"
    if cur["tts_engine"] not in ("piper", "pathumma-f5"):
        cur["tts_engine"] = "piper"
    if cur["handsfree_lang"] not in ("th", "en"):
        cur["handsfree_lang"] = "th"
    write_json(config.SETTINGS_FILE, cur)
    return cur


def mdir(mid: str) -> Path:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}(-[0-9]+)?", mid):
        raise ValueError("bad meeting id")
    return config.MEETINGS / mid


def new_meeting(title: str, mode: str, owner: str | None = None) -> dict:
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    mid, n = base, 1
    while (config.MEETINGS / mid).exists():
        n += 1
        mid = f"{base}-{n}"
    d = config.MEETINGS / mid
    d.mkdir(parents=True)
    st = {"id": mid, "title": title or f"Meeting {datetime.now():%Y-%m-%d %H:%M}",
          "created": datetime.now().isoformat(timespec="seconds"), "mode": mode, "status": "recording",
          "duration_s": 0.0, "progress": 0.0, "error": None, "history": [], "remark": "", "participants": [], "owner": owner}
    save_state(mid, st)
    return st


def load_state(mid: str) -> dict | None:
    return read_json(mdir(mid) / "state.json")


def save_state(mid: str, st: dict):
    write_json(mdir(mid) / "state.json", st)


def update_state(mid: str, **kw) -> dict:
    """Read-modify-write under a per-meeting lock: the server and the worker both update state.json."""
    import fcntl

    with open(mdir(mid) / "state.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = load_state(mid) or {"id": mid}
        st.update(kw)
        save_state(mid, st)
    return st


class transcript_lock:
    """Cross-process lock for read-modify-write of transcript.json (server edits vs the worker's speaker labels)."""

    def __init__(self, mid: str):
        self.path = mdir(mid) / "transcript.lock"

    def __enter__(self):
        import fcntl

        self.f = open(self.path, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        self.f.close()


def review_count(segs: list[dict]) -> int:
    """Lines a person reviewed or corrected: they are protected from automatic passes and from a full re-transcription."""
    return sum(1 for s in segs if s.get("reviewed") or s.get("history") or s.get("src") == "edited")


def text_rev(text: str | None) -> str:
    """Short fingerprint of a text: the editor sends the one it started from, so a save over a newer version is refused."""
    import hashlib

    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def add_summary_history(mid: str, kind: str, by: str | None, before: str | None, after: str | None):
    """summary-history.json: who changed the summary when, with the text before and after (edit / restore / ai)."""
    f = mdir(mid) / "summary-history.json"
    hist = read_json(f, []) or []
    hist.append({"at": datetime.now().strftime("%F %T"), "by": by, "kind": kind, "before": before, "after": after})
    write_json(f, hist[-100:])


def log_event(mid: str, msg: str):
    st = load_state(mid) or {}
    hist = (st.get("history") or [])[-59:] + [f"{datetime.now():%F %T} {msg}"]
    update_state(mid, history=hist)


def list_meetings() -> list[dict]:
    out = []
    for d in sorted(config.MEETINGS.glob("*-*"), reverse=True):
        st = read_json(d / "state.json")
        if st:
            out.append(st)
    return out


def pcm_path(mid: str) -> Path:
    return mdir(mid) / "audio.pcm"


def wav_path(mid: str) -> Path:
    return mdir(mid) / "audio.wav"


def pcm_seconds(mid: str) -> float:
    p = pcm_path(mid)
    return (p.stat().st_size // 2) / config.SR if p.exists() else 0.0


def wav_header(n_bytes: int) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, config.SR, config.SR * 2, 2, 16)
            + b"data" + struct.pack("<I", n_bytes))


def finalize_audio(mid: str) -> float:
    """audio.pcm (raw, appended while recording) -> audio.wav. Crash-safe: the wav appears atomically, pcm is removed last."""
    pcm, wav = pcm_path(mid), wav_path(mid)
    if pcm.exists():
        size = pcm.stat().st_size & ~1
        tmp = wav.with_name("audio.wav.tmp")
        with open(pcm, "rb") as src, open(tmp, "wb") as dst:
            dst.write(wav_header(size))
            left = size
            while left > 0:
                buf = src.read(min(8 << 20, left))
                if not buf:
                    break
                dst.write(buf)
                left -= len(buf)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, wav)
        pcm.unlink()
    return (wav.stat().st_size - 44) / 2 / config.SR if wav.exists() else 0.0


def load_audio(mid: str):
    """int16 memmap of the meeting audio (wav preferred)."""
    import numpy as np

    if wav_path(mid).exists():
        return np.memmap(wav_path(mid), dtype=np.int16, mode="r", offset=44)
    return np.memmap(pcm_path(mid), dtype=np.int16, mode="r")


def free_gb() -> float:
    return shutil.disk_usage(config.DATA).free / 1e9


def mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 99.0


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def speakers(mid: str) -> dict:
    return read_json(mdir(mid) / "speakers.json", {}) or {}


def speaker_names(mid: str, spk: dict | None = None) -> dict[str, str]:
    """cluster id -> display name (the user's name, else "ผู้เข้าร่วม N")."""
    spk = speakers(mid) if spk is None else spk
    return {sid: (v.get("name") or f"ผู้เข้าร่วม {sid[1:]}") for sid, v in spk.items()}


def clean_participants(raw) -> list[str]:
    """Names as typed by the user: trimmed, de-duplicated (case-insensitive), bounded."""
    out, seen = [], set()
    for x in raw if isinstance(raw, list) else []:
        n = " ".join(str(x).split())[:60]
        if n and n.casefold() not in seen:
            seen.add(n.casefold())
            out.append(n)
    return out[:50]


def context_text(st: dict) -> str:
    """Title / participants / remark as background for the summary model."""
    parts = []
    if st.get("title"):
        parts.append(f"ชื่อการประชุม: {st['title']}")
    if st.get("participants"):
        parts.append("ผู้เข้าร่วม: " + ", ".join(st["participants"]))
    if st.get("remark"):
        parts.append(f"หมายเหตุจากผู้จัด: {st['remark'][:1200]}")
    b = st.get("bot") or {}
    if b.get("enabled") and b.get("goal"):
        parts.append(f"เป้าหมายการประชุม: {b['goal']}")
    agenda = st.get("agenda") or (b.get("agenda") if b.get("enabled") else None)
    if agenda:
        parts.append("วาระการประชุม: " + "; ".join(f"{i + 1}) {a['title']}" + (f" ({a['note']})" if a.get("note") else "") for i, a in enumerate(agenda)))
    return "\n".join(parts)


def transcript_md(st: dict, segs: list[dict], names: dict[str, str] | None = None) -> str:
    lines = [f"# {st.get('title', '')}", "", f"- Date: {st.get('created', '')}", f"- Duration: {fmt_ts(st.get('duration_s', 0))}"]
    if st.get("participants"):
        lines.append("- Participants: " + ", ".join(st["participants"]))
    if st.get("remark"):
        lines.append("- Remark: " + " ".join(st["remark"].split()))
    lines.append("")
    for s in segs:
        who = f"{names.get(s.get('spk'), '')}: " if names and s.get("spk") else ""
        lines.append(f"[{fmt_ts(s['t0'])}] {who}{s['text']}")
    return "\n".join(lines) + "\n"


def recording_active() -> bool:
    """A meeting is being recorded: the server touches HEARTBEAT every second while audio comes in (processing pauses on it)."""
    try:
        return time.time() - config.HEARTBEAT.stat().st_mtime < 8
    except OSError:
        return False
