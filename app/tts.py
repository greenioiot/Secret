"""Local text-to-speech for the hands-free enrolment guide. Two voices (setting tts_engine):
  * piper       - projectAI's Piper th_TH tsync2 (tts_child.py): about a second per sentence;
  * pathumma-f5 - NECTEC Pathumma F5-TTS (tts_f5_child.py, .venv-f5): natural, but about 50 s per sentence on this CPU
                  (8 flow steps; measured 2026-10-11). So it never makes anyone wait: a sentence it has not made yet is spoken by
                  Piper at once and queued for F5 in the background (only while nothing is being recorded); prewarm() queues the
                  fixed guidance sentences when the voice is chosen. Next time the sentence is in the Pathumma voice.
Helpers start on first use and stop after TTS_IDLE_S without requests (or with the server). WAVs are cached by text in
data/tts-cache (Piper) and data/tts-cache/f5 (Pathumma)."""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import config

log = logging.getLogger("secret.tts")
PY = config.PROJECT_AI / ".venv" / "bin" / "python"
VOICE_TH = config.PROJECT_AI / "models/piper/th_TH/th_TH-tsync2-medium.onnx"
CACHE = config.DATA / "tts-cache"
F5_CACHE = CACHE / "f5"
TTS_IDLE_S = 300
F5_PY = config.ROOT / ".venv-f5/bin/python"
F5_DIR = config.PATHUMMA / "f5-tts"


class Helper:
    def __init__(self, name: str, cmd: list[str], start_timeout_note: str):
        self.name, self.cmd, self.note = name, cmd, start_timeout_note
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.used = 0.0

    def _start(self):
        self.proc = config.spawn(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(config.DATA / "tts.log", "ab"), text=True, bufsize=1)
        line = self.proc.stdout.readline()
        if not line or not json.loads(line).get("ready"):
            self.stop()
            raise RuntimeError(f"the {self.name} voice did not start (see data/tts.log)")
        log.info("tts helper %s started (pid %s)", self.name, self.proc.pid)

    def make(self, text: str, out: Path) -> Path:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.used = time.time()
            self.proc.stdin.write(json.dumps({"text": text, "out": str(out) + ".tmp"}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            r = json.loads(self.proc.stdout.readline() or '{"ok": false, "error": "the speech voice stopped"}')
            self.used = time.time()
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "speech failed"))
        Path(str(out) + ".tmp").replace(out)
        for f in sorted(out.parent.glob("*.wav"), key=lambda f: f.stat().st_mtime)[:-300]:  # keep the cache small
            f.unlink(missing_ok=True)
        return out

    def stop(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()

    def release_idle(self):
        if self.proc is not None and time.time() - self.used > TTS_IDLE_S and self.lock.acquire(blocking=False):
            try:
                self.stop()
            finally:
                self.lock.release()


PIPER = Helper("Piper", [str(PY), str(Path(__file__).with_name("tts_child.py")), str(config.PROJECT_AI)], "")
F5 = Helper("Pathumma F5", ["nice", "-n", "15", str(F5_PY), str(Path(__file__).with_name("tts_f5_child.py")), str(config.ROOT), str(config.PROJECT_AI)], "")
_queue: collections.deque[str] = collections.deque(maxlen=200)
_queued: set[str] = set()
_qlock = threading.Lock()
_worker: threading.Thread | None = None


def engine() -> str:
    """'pathumma-f5' only when chosen in Settings and installed, else 'piper'."""
    import store

    e = store.settings().get("tts_engine", "piper")
    return e if e == "pathumma-f5" and f5_available() else "piper"


def f5_available() -> bool:
    return F5_PY.exists() and (F5_DIR / "model_ema.safetensors").exists() and (config.PATHUMMA / "f5-ref.wav").exists()


def piper_available() -> bool:
    return PY.exists() and VOICE_TH.exists() and (config.PROJECT_AI / "app" / "thai_speech.py").exists()


def available() -> bool:
    return piper_available() or f5_available()


def _clean(text: str) -> str:
    text = " ".join(str(text).split())[:600]
    if not text:
        raise ValueError("nothing to say")
    return text


def _path(text: str, f5: bool) -> Path:
    d = F5_CACHE if f5 else CACHE
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha256(text.encode()).hexdigest()[:20] + ".wav")


def _recording() -> bool:
    import store

    try:
        return any(m.get("status") == "recording" for m in store.list_meetings())
    except Exception:  # noqa: BLE001
        return True


def _run_queue():
    """Background: make queued sentences with F5, one at a time, never while a meeting is being recorded."""
    global _worker
    while True:
        with _qlock:
            if not _queue or engine() != "pathumma-f5":
                _queue.clear()
                _queued.clear()
                _worker = None
                return
            text = _queue[0]
        if _recording():
            F5.stop()  # give the CPU and memory back to the recording
            time.sleep(15)
            continue
        out = _path(text, True)
        if not out.exists():
            try:
                F5.make(text, out)
            except Exception as e:  # noqa: BLE001 - one bad sentence must not stop the rest
                log.warning("Pathumma F5 could not make %r: %s", text[:40], e)
        with _qlock:
            if _queue and _queue[0] == text:
                _queue.popleft()
            _queued.discard(text)


def _enqueue(texts: list[str]) -> int:
    global _worker
    n = 0
    with _qlock:
        for t in texts:
            if t not in _queued and not _path(t, True).exists():
                _queue.append(t)
                _queued.add(t)
                n += 1
        if _queue and _worker is None:
            _worker = threading.Thread(target=_run_queue, name="tts-f5", daemon=True)
            _worker.start()
    return n


def prewarm(texts: list[str]) -> int:
    """Queue sentences for the Pathumma voice (the fixed guidance lines); returns how many still have to be made."""
    if engine() != "pathumma-f5":
        return 0
    return _enqueue([_clean(t) for t in texts if str(t).strip()][:150])


def pending() -> int:
    with _qlock:
        return len(_queue)


def wav(text: str) -> Path:
    """WAV of the spoken text: the Pathumma voice when it has made this sentence before, else Piper now (and F5 later)."""
    text = _clean(text)
    if engine() == "pathumma-f5":
        out = _path(text, True)
        if out.exists():
            return out
        _enqueue([text])
        if not piper_available():  # no fast voice to stand in: wait for F5
            return F5.make(text, out)
    out = _path(text, False)
    if out.exists():
        return out
    return PIPER.make(text, out)


def stop():
    PIPER.stop()
    F5.stop()


def release_idle():
    """Called from the server's housekeeping loop: give the voices' memory back when nobody needed them for a while."""
    PIPER.release_idle()
    if pending() == 0:
        F5.release_idle()
