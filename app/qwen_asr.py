"""Client for qwen_server.py: Typhoon ASR Qwen 1.7B with contextual biasing (the 'best' accuracy mode). Same interface as asr.WhisperTH
(transcribe / transcribe_many) so the worker can swap engines. The child process holds ~8 GB of RAM; it is stopped with close()
(also at exit) and the model is loaded on first use."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import tempfile
import threading
import time

import numpy as np

import config

MODEL = os.environ.get("SECRET_QWEN_ASR", "typhoon-ai/typhoon-asr-qwen-1.7b-ctx")
PY = config.ROOT / ".venv-qwen/bin/python"


def model_cached() -> bool:
    import glob

    return bool(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--" + MODEL.replace("/", "--") + "/snapshots/*/model.safetensors")))


def available() -> bool:
    """The environment and the model files are there (the model, ~4 GB, was removed to free disk: re-download with
    `.venv-qwen/bin/python -c "from huggingface_hub import snapshot_download as d; d('typhoon-ai/typhoon-asr-qwen-1.7b-ctx')"`)."""
    return PY.exists() and model_cached()


class QwenASR:
    def __init__(self, threads: int = 8, batch: int = 12):
        if not available():
            raise RuntimeError("the Qwen ASR environment (.venv-qwen) is not installed")
        self.threads, self.batch, self.bias = threads, batch, ""
        self.lock = threading.Lock()
        self.proc = None
        atexit.register(self.close)

    def _start(self):
        self.proc = subprocess.Popen([str(PY), str(config.ROOT / "app/qwen_server.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(config.DATA / "qwen.log", "ab"), text=True, bufsize=1)
        r = self._call({"cmd": "load", "threads": self.threads, "model": MODEL}, timeout=900)
        if not r.get("ok"):
            raise RuntimeError("Qwen ASR could not load: " + r.get("error", "?"))

    def _call(self, req: dict, timeout: float = 7200) -> dict:
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Qwen ASR process died ({self.proc.returncode}), see data/qwen.log")
            line = self.proc.stdout.readline()
            if line.startswith("@@"):
                return json.loads(line[2:])
            if time.time() - t0 > timeout:
                raise RuntimeError("Qwen ASR timed out")

    def set_bias(self, terms: list[str]):
        """Names and terms the model should spell exactly as written (the card recommends <= 50)."""
        seen, out = set(), []
        for t in terms:
            t = " ".join(str(t).split())
            if 1 < len(t) <= 40 and t.casefold() not in seen:
                seen.add(t.casefold())
                out.append(t)
        self.bias = " ".join(out[:50])

    def transcribe_many(self, pcms, batch_size: int | None = None) -> list[tuple[str, float, float]]:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            tmp = tempfile.mkdtemp(prefix="qwen-", dir=str(config.DATA))
            try:
                files = []
                for i, p in enumerate(pcms):
                    f = os.path.join(tmp, f"{i}.npy")
                    np.save(f, np.asarray(p, np.float32) / 32768.0)
                    files.append(f)
                r = self._call({"cmd": "asr", "files": files, "bias": self.bias, "batch": batch_size or self.batch})
            finally:
                for f in os.listdir(tmp):
                    os.remove(os.path.join(tmp, f))
                os.rmdir(tmp)
        if not r.get("ok"):
            raise RuntimeError("Qwen ASR failed: " + r.get("error", "?"))
        # the model gives no log-probability: a neutral value, so choose() relies on its repetition / Typhoon cross-checks
        return [(t, -0.3, 0.0) if t else ("", -9.0, 1.0) for t in r["texts"]]

    def transcribe(self, pcm, prompt: str = ""):
        return self.transcribe_many([pcm], 1)[0]

    def close(self):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.write('{"cmd":"quit"}\n')
                p.stdin.flush()
                p.wait(10)
            except Exception:  # noqa: BLE001
                p.kill()
