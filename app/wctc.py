"""Typhoon Whisper CTC on the Hexagon NPU, 16-bit: a frozen Thai Whisper-medium encoder + a 29M character-CTC head, no decoder, ONE graph
(models/whisper-ctc-npu/ctc_full.tflite: log-mel [1,80,3000] -> log-probs [1,1500,80]; exported with torch, compiled with AI Hub, run by the QNN TFLite
delegate). Measured on 80 public Thai clips: CER 10.2 % (identical to the CPU fp32 run), ~0.5 s per 30 s window. Same interface as asr.ThaiCTC, so it can
be the live-caption model or the engine of the 'npu' accuracy mode. Text comes out without spaces (Thai) and with digits / % , - . as the model writes them."""
from __future__ import annotations

import json
import threading
import time

import numpy as np

import config

DIR = config.ROOT / "models/whisper-ctc-npu"
T_FRAMES = 3000  # 30 s of 10 ms mel frames
SECS = 30.0


def available() -> bool:
    return (DIR / "ctc_full.tflite").exists() and (DIR / "symbols.json").exists()


class WhisperCTC:
    kind = "wctc"

    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter, load_delegate
        from faster_whisper.feature_extractor import FeatureExtractor

        self.backend, self.npu_error = "npu", ""
        self.symbols = json.loads((DIR / "symbols.json").read_text(encoding="utf-8"))
        self.fe = FeatureExtractor(feature_size=80)
        cache = DIR / "cache"
        cache.mkdir(exist_ok=True)  # the compiled graph is cached there: the first load takes ~2 minutes, later ones seconds
        delegate = load_delegate(config.QNN_DELEGATE, {"backend_type": "htp", "htp_performance_mode": "2", "cache_dir": str(cache), "model_token": "wctc_medium"})
        self._it = Interpreter(model_path=str(DIR / "ctc_full.tflite"), experimental_delegates=[delegate])
        self._it.allocate_tensors()
        self._delegate = delegate
        self._in, self._out = self._it.get_input_details()[0]["index"], self._it.get_output_details()[0]["index"]
        self.lock = threading.Lock()
        self.calls, self.secs_total, self.secs_max = 0, 0.0, 0.0  # statistics of the NPU calls (read by stats.py)

    def _logp(self, audio: np.ndarray) -> np.ndarray:
        """audio: float32 mono 16 kHz, <= 30 s -> log-probs [frames, 80] of the valid part."""
        from faster_whisper.audio import pad_or_trim

        n = len(audio)
        feats = pad_or_trim(self.fe(audio), T_FRAMES)[None].astype(np.float32)
        t = time.time()
        with self.lock:
            self._it.set_tensor(self._in, feats)
            self._it.invoke()
            lp = self._it.get_tensor(self._out)[0]
        dt = time.time() - t
        self.calls += 1
        self.secs_total += dt
        self.secs_max = max(self.secs_max, dt)
        return lp[: (n // 160 - 1) // 2 + 2]

    def _decode(self, lp: np.ndarray) -> tuple[str, float]:
        ids = lp.argmax(-1)
        emit = ids != 0
        conf = float(np.exp(lp.max(-1))[emit].mean()) if emit.any() else 0.0
        out, prev = [], 0
        for i in ids.tolist():
            if i != 0 and i != prev:
                out.append(self.symbols[i])
            prev = i
        return "".join(out).replace("<unk>", "").strip(), conf

    def transcribe(self, pcm16: np.ndarray) -> tuple[str, float]:
        a = np.asarray(pcm16, np.float32) / 32768.0
        if len(a) > SECS * config.SR:
            return self.transcribe_long(pcm16)
        text, conf = self._decode(self._logp(a))
        from asr import fix_thai

        return fix_thai(text), conf

    def transcribe_long(self, pcm16: np.ndarray, max_s: float = 28.0) -> tuple[str, float]:
        """One Whisper window holds 30 s: longer audio is cut at its quietest point near the limit."""
        n = len(pcm16)
        if n <= max_s * config.SR:
            return self.transcribe(pcm16)
        texts, confs, pos = [], [], 0
        while pos < n:
            end = min(n, pos + int(max_s * config.SR))
            if end < n:
                lo, hi = end - 3 * config.SR, end - config.SR // 2
                win = np.abs(np.asarray(pcm16[lo:hi], np.int32))[: ((hi - lo) // 1600) * 1600].reshape(-1, 1600).mean(1)
                end = lo + int(win.argmin()) * 1600 + 800
            t, c = self.transcribe(pcm16[pos:end])
            if t:
                texts.append(t)
                confs.append(c)
            pos = end
        return "".join(texts), float(np.mean(confs)) if confs else 0.0

    def transcribe_many(self, pcms, batch_size=None) -> list[tuple[str, float, float]]:
        """Same shape as WhisperTH.transcribe_many: (text, log-probability, no-speech probability) per piece."""
        out = []
        for p in pcms:
            text, conf = self.transcribe_long(p)
            out.append((text, float(np.log(max(conf, 1e-3))), 0.0) if text else ("", -9.0, 1.0))
        return out

    def stats(self) -> dict:
        return {"calls": self.calls, "npu_s_total": round(self.secs_total, 2), "npu_s_max": round(self.secs_max, 2),
                "npu_ms_mean": round(1000 * self.secs_total / self.calls, 1) if self.calls else None}
