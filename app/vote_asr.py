"""Accuracy mode "vote3": three Thai speech models transcribe every piece and the text the other two agree with most is kept.

Pathumma Whisper Large v3 + Pathumma Whisper medium + Pathumma FastConformer: they make different mistakes, so the "median" text cancels many
of them. Measured on the 80 FLEURS Thai clips (PROGRESS.md, Round 43): 5.25 % character errors vs 6.10 % for the best single model and
6.65 % for FastConformer alone; about 3.9x real time on this CPU (a 1-hour meeting takes about 4 hours), so it suits processing after the
meeting, not live captions. Where the models disagree the other versions are kept (last_alts) so a person can pick the right one."""
from __future__ import annotations

import numpy as np

import config

ORDER = ("pwl", "pwm", "rnnt")  # the order of the measured result (a tie goes to the first)
NAMES = {"pwl": "Pathumma Whisper Large", "pwm": "Pathumma Whisper medium", "rnnt": "Pathumma FastConformer"}
RAM_GB = 9.0


def available() -> bool:
    import rnnt

    return rnnt.available() and all((config.WHISPER_CT2[m][1] / "model.bin").exists() for m in ("pathumma-large", "pathumma-medium"))


def _norm(t: str) -> str:
    import textnorm

    return "".join(textnorm.polish(t or "", False, False).split())


def pick(texts: list[str]) -> tuple[int, float]:
    """-> (index of the text closest to the others, agreement 0..1 = 1 - its mean normalised distance to them)."""
    from rapidfuzz.distance import Levenshtein

    n = [_norm(t) for t in texts]
    cost = [sum(Levenshtein.distance(n[i], t) / max(1, len(t)) for j, t in enumerate(n) if j != i) for i in range(len(n))]
    i = min(range(len(n)), key=lambda k: cost[k])
    return i, max(0.0, 1.0 - cost[i] / max(1, len(n) - 1))


class VoteASR:
    """The three models; released by close() (idle) and loaded again on the next use, like the other engines (the worker's two lanes share it)."""

    def __init__(self, threads: int = 8):
        import threading

        self.threads, self.models = threads, {}
        self.lock, self._tl = threading.Lock(), threading.local()
        self._load()

    def _load(self):
        import asr
        import rnnt

        if not self.models:
            self.models = {"pwl": asr.WhisperTH(threads=self.threads, model_dir=config.WHISPER_CT2["pathumma-large"][1]),
                           "pwm": asr.WhisperTH(threads=self.threads, model_dir=config.WHISPER_CT2["pathumma-medium"][1]),
                           "rnnt": rnnt.PathummaRNNT(threads=self.threads)}

    @property
    def last_alts(self) -> list[dict]:
        """The other models' texts of this thread's last call (the worker's two lanes may use the engine at the same time)."""
        return getattr(self._tl, "alts", [])

    def set_bias(self, terms):  # no contextual biasing: names are applied after recognition
        pass

    def transcribe_many(self, pcms, batch_size: int = 8) -> list[tuple[str, float, float]]:
        """-> per piece (text, log-confidence, no-speech) like WhisperTH; self.last_alts = per piece {"agree", "alts": {engine: text}}."""
        with self.lock:
            self._load()
            outs = {k: self.models[k].transcribe_many(list(pcms), batch_size) for k in ORDER}
        res, alts = [], []
        for i in range(len(pcms)):
            texts = [outs[k][i][0] or "" for k in ORDER]
            if not any(t.strip() for t in texts):
                res.append(("", -9.0, 1.0))
                alts.append({"agree": 1.0, "alts": {}})
                continue
            j, agree = pick(texts)
            # the agreement is the confidence; floored so asr.choose() keeps the vote unless the text loops or is empty
            res.append((texts[j], float(np.log(max(agree, 0.35))), 0.0))
            alts.append({"agree": round(agree, 3), "pick": ORDER[j], "alts": {k: t for k, t in zip(ORDER, texts)}})
        self._tl.alts = alts
        return res

    def transcribe(self, pcm, prompt: str = ""):
        return self.transcribe_many([pcm], 1)[0]

    def close(self):
        with self.lock:
            m = self.models.get("rnnt")
            if m is not None and hasattr(m, "close"):
                m.close()
            self.models = {}
