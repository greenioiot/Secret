"""Voice library: quality-checked enrolment with a preview step, retained sample speech for playback, notes.

Enrolment is two steps so nothing is saved by accident:
  1. analyse(pcm)  -> a DRAFT (data/voices_drafts/<id>.json + .wav): speech seconds, voice consistency, single-voice check, problems
  2. commit(draft) -> voiceprints added to a person (or a new person); "replace" must be asked for explicitly.
Up to 40 s of the person's own speech (pauses removed) is kept per person in data/voices/<vid>/ for playback.
Consistency is how similar the parts of the sample are to each other (cosine of CAM++ embeddings): it is a quality
indicator, NOT a recognition-accuracy percentage."""
from __future__ import annotations

import io
import os
import re
import secrets
import shutil
import time
import wave

import numpy as np

import config
import speakers
import store

SR = config.SR
DRAFTS = config.DATA / "voices_drafts"
DRAFT_TTL_S = 30 * 60
MIN_SPEECH_S = 10.0       # below this no voiceprint is made
TARGET_SPEECH_S = 20.0    # good enrolment: at least this much detected speech
TARGET_CONSISTENCY = 0.75
MAX_CAPTURE_S = 40.0      # capture / retained sample speech per person
MAX_UPLOAD_S = 180.0
WIN_S = 4.0               # consistency is measured on windows of this length


def _wav(pcm: np.ndarray) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(pcm, np.int16).tobytes())
    return b.getvalue()


def _private(d):
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)


def quality(pcm: np.ndarray, embed) -> dict:
    """Speech amount, consistency and single-voice check of a recording. Returns the report plus the speech-only audio and the voiceprints."""
    pcm = np.asarray(pcm, np.int16)
    units = speakers.vad_units(pcm) if len(pcm) else []
    speech = np.concatenate([pcm[a:b] for a, b in units]) if units else pcm[:0]
    secs = len(speech) / SR
    rep = {"recorded_s": round(len(pcm) / SR, 1), "speech_s": round(secs, 1), "consistency": None, "single_voice": None, "voices": None,
           "problems": [], "ok": False, "good": False, "targets": {"speech_s": TARGET_SPEECH_S, "consistency": TARGET_CONSISTENCY}}
    if secs < 3:
        rep["problems"].append("almost no speech was heard: speak closer to the microphone in a quiet room")
        return {**rep, "_speech": speech, "_embs": []}
    win = int(WIN_S * SR)
    n = max(2, min(12, len(speech) // win))
    step = len(speech) // n
    E = np.stack([speakers.unit(embed(speech[i * step:i * step + win])) for i in range(n)])
    c = speakers.unit(E.mean(0))
    sims = E @ c
    rep["consistency"] = round(float(sims.mean()), 3)
    labels = speakers.cluster(E, 0.5) if len(E) >= 3 else np.zeros(len(E), int)
    counts = sorted(np.bincount(labels).tolist(), reverse=True)
    big = [k for k in counts if k / len(E) >= 0.2]  # a second voice that holds at least a fifth of the sample
    rep["voices"] = len(big) or 1
    rep["single_voice"] = rep["voices"] == 1 and float(sims.min()) >= 0.35
    if secs < MIN_SPEECH_S:
        rep["problems"].append(f"only {secs:.0f} s of speech: at least {MIN_SPEECH_S:.0f} s are needed, {TARGET_SPEECH_S:.0f} s or more is better")
    elif secs < TARGET_SPEECH_S:
        rep["problems"].append(f"{secs:.0f} s of speech: usable, but {TARGET_SPEECH_S:.0f} s or more gives a better voiceprint")
    if not rep["single_voice"]:
        rep["problems"].append("it sounds like more than one voice (someone else talking, or strong echo): record only this person")
    elif rep["consistency"] < TARGET_CONSISTENCY:
        rep["problems"].append(f"the voice is not very consistent ({rep['consistency']:.2f} < {TARGET_CONSISTENCY}): noise, music or changing distance to the microphone")
    rep["ok"] = secs >= MIN_SPEECH_S and bool(rep["single_voice"])
    rep["good"] = rep["ok"] and secs >= TARGET_SPEECH_S and rep["consistency"] >= TARGET_CONSISTENCY
    k = max(2, min(5, len(speech) // (6 * SR)))
    st = len(speech) // k
    embs = [embed(speech[i * st:(i + 1) * st][: 10 * SR]) for i in range(k)] if rep["ok"] else []
    return {**rep, "_speech": speech, "_embs": embs}


# ---------------------------------------------------------------- drafts (analysed, not saved yet)
def _clean_drafts():
    if not DRAFTS.exists():
        return
    for f in DRAFTS.glob("*.json"):
        d = store.read_json(f) or {}
        if time.time() - d.get("created", 0) > DRAFT_TTL_S:
            f.unlink(missing_ok=True)
            f.with_suffix(".wav").unlink(missing_ok=True)


def make_draft(pcm: np.ndarray, embed, source: str, by: str) -> dict:
    _clean_drafts()
    q = quality(pcm, embed)
    did = "d" + secrets.token_hex(6)
    _private(DRAFTS)
    speech = q.pop("_speech")[: int(MAX_CAPTURE_S * SR)]
    embs = q.pop("_embs")
    store.atomic_write(DRAFTS / f"{did}.wav", _wav(speech))
    best = speakers.best_match(embs) if embs else None
    draft = {"id": did, "created": time.time(), "source": source, "by": by, "quality": q, "embs": [np.asarray(e, np.float32).round(5).tolist() for e in embs],
             "sample_s": round(len(speech) / SR, 1), "sounds_like": best}
    store.write_json(DRAFTS / f"{did}.json", draft)
    return public_draft(draft)


def public_draft(d: dict) -> dict:
    return {"id": d["id"], "quality": d["quality"], "sample_s": d["sample_s"], "source": d["source"], "sounds_like": d.get("sounds_like"),
            "audio": f"/api/voices/drafts/{d['id']}/audio"}


def get_draft(did: str) -> dict | None:
    if not re.fullmatch(r"d[0-9a-f]{12}", did or ""):
        return None
    d = store.read_json(DRAFTS / f"{did}.json")
    return d if d and time.time() - d["created"] <= DRAFT_TTL_S else None


def draft_wav(did: str):
    d = get_draft(did)
    return (DRAFTS / f"{did}.wav") if d else None


def drop_draft(did: str):
    if re.fullmatch(r"d[0-9a-f]{12}", did or ""):
        (DRAFTS / f"{did}.json").unlink(missing_ok=True)
        (DRAFTS / f"{did}.wav").unlink(missing_ok=True)


# ---------------------------------------------------------------- saving into the library
def _vdir(vid: str):
    return config.VOICES / vid


def commit(did: str, name: str, by: str, voice_id: str | None = None, replace: bool = False, notes: str | None = None, consent: bool = False) -> dict:
    d = get_draft(did)
    if not d:
        raise KeyError(did)
    if not consent:
        raise ValueError("the person's consent is needed: tick the consent box")
    if not d["quality"]["ok"] or not d["embs"]:
        raise ValueError("this sample did not pass the quality check: record again")
    name = " ".join(str(name).split())[:60]
    if not name and not voice_id:
        raise ValueError("a name is needed")
    existing = {v["id"]: v for v in speakers.load_voices()}
    if voice_id and voice_id not in existing:
        raise ValueError("that person is not in the library any more")
    if replace and voice_id:  # explicit: the old voiceprints and retained samples of this person are replaced by this sample
        v = existing[voice_id]
        shutil.rmtree(_vdir(voice_id), ignore_errors=True)
        v.update(embeddings=d["embs"], audio=[], updated=time.strftime("%F %T"), name=name or v["name"])
        v["sources"] = (v.get("sources") or [])[-9:] + [f"{d['source']} ({by}, replaced)"]
        store.write_json(speakers._vfile(voice_id), v)
    else:
        v = speakers.add_voice_samples(name or existing[voice_id]["name"], d["embs"], voice_id=voice_id, source=f"{d['source']} ({by})")
    # retained sample speech: newest first, at most MAX_CAPTURE_S seconds per person
    vd = _vdir(v["id"])
    _private(vd)
    fn = f"s{int(time.time())}{secrets.token_hex(2)}.wav"
    shutil.copyfile(DRAFTS / f"{did}.wav", vd / fn)
    os.chmod(vd / fn, 0o600)
    audio = [{"file": fn, "seconds": d["sample_s"], "at": time.strftime("%F %T"), "by": by, "source": d["source"],
              "speech_s": d["quality"]["speech_s"], "consistency": d["quality"]["consistency"]}] + list(v.get("audio") or [])
    keep, total = [], 0.0
    for a in audio:
        if total + a["seconds"] <= MAX_CAPTURE_S + 0.5 or not keep:
            keep.append(a)
            total += a["seconds"]
        else:
            (vd / a["file"]).unlink(missing_ok=True)
    if total > MAX_CAPTURE_S + 0.5:  # a single long sample: keep only its first 40 s of speech
        _trim(vd / keep[0]["file"], MAX_CAPTURE_S)
        keep[0]["seconds"] = MAX_CAPTURE_S
    v["audio"] = keep
    v.setdefault("consents", []).append({"at": time.strftime("%F %T"), "by": by, "how": f"confirmed by {by} at enrolment ({d['source']})"})
    if notes is not None:
        v["notes"] = str(notes).strip()[:500]
    store.write_json(speakers._vfile(v["id"]), v)
    drop_draft(did)
    return public(v)


def _trim(path, seconds: float):
    with wave.open(str(path), "rb") as r:
        frames = r.readframes(int(seconds * SR))
    store.atomic_write(path, _wav(np.frombuffer(frames, np.int16)))


def public(v: dict) -> dict:
    return {"id": v["id"], "name": v["name"], "samples": len(v.get("embeddings") or []), "created": v.get("created"), "updated": v.get("updated"),
            "notes": v.get("notes", ""), "sources": (v.get("sources") or [])[-5:], "consents": len(v.get("consents") or []),
            "audio": [{"n": i, "seconds": a["seconds"], "at": a["at"], "by": a.get("by"), "source": a.get("source"), "speech_s": a.get("speech_s"),
                       "consistency": a.get("consistency")} for i, a in enumerate(v.get("audio") or [])],
            "audio_s": round(sum(a["seconds"] for a in v.get("audio") or []), 1)}


def sample_path(vid: str, n: int):
    v = store.read_json(speakers._vfile(vid)) or {}
    a = (v.get("audio") or [])
    if not 0 <= n < len(a):
        return None
    p = _vdir(vid) / a[n]["file"]
    return p if p.exists() else None


def delete_sample(vid: str, n: int) -> dict:
    """Removes the retained AUDIO of one sample (the voiceprint stays: delete the person to remove everything)."""
    f = speakers._vfile(vid)
    v = store.read_json(f)
    if not v or not 0 <= n < len(v.get("audio") or []):
        raise KeyError(vid)
    a = v["audio"].pop(n)
    (_vdir(vid) / a["file"]).unlink(missing_ok=True)
    store.write_json(f, v)
    return public(v)


def delete_voice(vid: str):
    speakers._vfile(vid).unlink(missing_ok=True)
    shutil.rmtree(_vdir(vid), ignore_errors=True)


def set_notes(vid: str, name: str | None, notes: str | None) -> dict:
    f = speakers._vfile(vid)
    v = store.read_json(f)
    if not v:
        raise KeyError(vid)
    if name is not None:
        n = " ".join(str(name).split())[:60]
        if not n:
            raise ValueError("the name cannot be empty")
        v["name"] = n
    if notes is not None:
        v["notes"] = str(notes).strip()[:500]
    v["updated"] = time.strftime("%F %T")
    store.write_json(f, v)
    return public(v)


def pcm_from_b64(b64: str) -> np.ndarray:
    import base64

    raw = base64.b64decode(b64, validate=True)
    if len(raw) > MAX_UPLOAD_S * SR * 2:
        raise ValueError(f"the sample is too long: at most {MAX_UPLOAD_S / 60:.0f} minutes")
    return np.frombuffer(raw[: len(raw) & ~1], np.int16)
