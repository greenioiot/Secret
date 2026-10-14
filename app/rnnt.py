"""Client for nemo_server.py: NECTEC Pathumma STT FastConformer-RNNT large (accuracy mode 'pathumma-rnnt'). Same interface as
asr.WhisperTH, so the worker can swap engines; the child process (a NeMo venv with torch) is started on first use and stopped with close()."""
from __future__ import annotations

import atexit
import subprocess
import threading

import config
import qwen_asr


def available() -> bool:
    return config.NEMO_PY.exists() and config.PATHUMMA_RNNT.exists()


class PathummaRNNT(qwen_asr.QwenASR):
    def __init__(self, threads: int = 8, batch: int = 8):
        if not available():
            raise RuntimeError(f"Pathumma FastConformer needs {config.NEMO_PY} and {config.PATHUMMA_RNNT}")
        self.threads, self.batch, self.bias = threads, batch, ""  # (not QwenASR.__init__: that checks for the Qwen model)
        self.lock = threading.Lock()
        self.proc = None
        atexit.register(self.close)

    def _start(self):
        self.proc = config.spawn([str(config.NEMO_PY), str(config.ROOT / "app/nemo_server.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=open(config.DATA / "nemo.log", "ab"), text=True, bufsize=1)
        r = self._call({"cmd": "load", "threads": self.threads, "model": str(config.PATHUMMA_RNNT)}, timeout=600)
        if not r.get("ok"):
            raise RuntimeError("Pathumma FastConformer could not load: " + r.get("error", "?"))

    def set_bias(self, terms):  # no contextual biasing in this model: names are applied after recognition, as for Whisper
        pass


class PathummaLive(PathummaRNNT):
    """The same model as the live caption engine (setting live_engine 'pathumma-rnnt'): the ThaiCTC interface (transcribe / transcribe_long ->
    (text, confidence)). Started at once (about 15 s), so the check before a meeting loads it and the first caption is not late. Measured on
    this CPU with 4 threads: an 8-s sentence in 0.94 s (median; Typhoon on the NPU about 0.1 s). The model takes long audio, no splitting."""
    kind = "pathumma-rnnt"
    backend = "cpu"
    npu_error = ""

    def __init__(self, threads: int = 4):
        super().__init__(threads, 1)
        with self.lock:
            self._start()

    def transcribe(self, pcm16, prompt: str = ""):
        text = self.transcribe_many([pcm16], 1)[0][0]
        return text, (0.9 if text else 0.0)  # no confidence from this model: a neutral value

    def transcribe_long(self, pcm16, max_s: float = 0):
        return self.transcribe(pcm16)
