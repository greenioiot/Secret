"""Thai speech recognition, tuned with the lessons from projectAI:
  * Typhoon ASR realtime (NeMo FastConformer CTC, ONNX): fast, used for live captions and as a cross-check.
    0.3 s of silence padding on both sides + "ํา" -> "ำ" normalisation (CER 8.2% -> 3.3% on clean, 10.8% -> 6.8% noisy).
  * Thonburian distil-Whisper (CTranslate2): the accurate pass after the meeting.
  * Whisper hallucination guard: repeated loops / very low confidence fall back to the Typhoon text.
"""
from __future__ import annotations

import json
import re
import threading
import zlib

import numpy as np

import config

SR = config.SR


def fix_thai(text: str) -> str:
    text = text.replace("ํา", "ำ")  # nikhahit + sara aa -> sara am (looks identical, never matches text)
    text = text.replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_rules(corrections: str) -> list[tuple[str, str]]:
    """The word collection: one "wrong=>right" per line, later lines override earlier ones for the same wrong word."""
    rules: dict[str, str] = {}
    for line in corrections.splitlines():
        if "=>" in line:
            a, b = (x.strip() for x in line.split("=>", 1))
            if a:
                rules[a] = b
    return list(rules.items())


def rules_text(rules) -> str:
    return "\n".join(f"{a}=>{b}" for a, b in rules)


def apply_corrections(text: str, corrections: str) -> str:
    for a, b in sorted(parse_rules(corrections), key=lambda r: -len(r[0])):  # longest first: a long rule is not broken by a shorter one
        text = text.replace(a, b)
    return text


def polish(text: str, cfg: dict) -> str:
    """Saved word corrections first (they win), then the writing style: English words in English, numbers as digits."""
    import textnorm

    text = apply_corrections(text, cfg.get("corrections", ""))
    text = textnorm.apply_rules(text, cfg.get("vocab_rules"))  # domain vocabulary packs (Vocab menu)
    return textnorm.polish(text, numbers=bool(cfg.get("numbers_arabic", True)), english=bool(cfg.get("english_words", True)))


def rule_problem(wrong: str, right: str) -> str:
    """Why a rule would be unsafe (Thai has no spaces: a one-letter rule would rewrite half of the language)."""
    if not wrong or wrong == right:
        return "The wrong and the right word must be different."
    if len(wrong) > 60 or len(right) > 60:
        return "Keep each word or phrase under 60 characters."
    if "=>" in wrong or "=>" in right or "\n" in wrong + right:
        return "Do not use => or line breaks inside a word."
    if len(wrong) < (3 if wrong.isascii() else 2):
        return "The wrong word is too short: it would also change correct words. Include a few letters around it."
    return ""


def suggest_rule(old: str, new: str) -> dict | None:
    """From a manual edit of one line, guess the 'wrong => right' pair (one changed place, padded with its neighbours so it is specific)."""
    import difflib

    ops = [o for o in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if o[0] != "equal"]
    if len(ops) != 1:
        return None
    _, i1, i2, j1, j2 = ops[0]
    left = True
    base = lambda t: sum(1 for c in t if not (c in "\u0e31\u0e47\u0e48\u0e49\u0e4a\u0e4b\u0e4c\u0e4d\u0e4e" or "\u0e34" <= c <= "\u0e3a"))  # letters, not tone marks / upper-lower vowels
    while (base(old[i1:i2]) < 4 or i1 == i2 or j1 == j2) and (i1 > 0 or i2 < len(old)):  # pad with context until specific enough
        if left and i1 > 0:
            i1 -= 1
            j1 -= 1
        elif i2 < len(old):
            i2 += 1
            j2 += 1
        elif i1 > 0:
            i1 -= 1
            j1 -= 1
        left = not left
    wrong, right = old[i1:i2], new[j1:j2]
    return None if rule_problem(wrong, right) else {"wrong": wrong, "right": right}


def repetition_ratio(text: str) -> float:
    """zlib compression ratio: Whisper loops ("ครับครับครับ...") compress extremely well.
    One byte per Thai character, not UTF-8: Thai is 3 bytes a character with shared lead bytes, so ordinary long Thai
    compressed past 2.4 and 13 % of good lines were thrown away as "loops" (real loops score 3-11, normal text <1.5)."""
    b = bytes((ord(c) - 0x0E00 + 0x80) if 0x0E00 <= ord(c) <= 0x0E7F else (ord(c) if ord(c) < 128 else 0x7F) for c in text)
    return len(b) / max(1, len(zlib.compress(b))) if len(b) > 40 else 1.0


class ThaiCTC:
    """Typhoon ASR. npu=True: mel on the CPU (ONNX), FastConformer encoder + CTC head on the Hexagon NPU (TFLite via the
    QNN delegate, fixed 16 s window, ~55 ms per call whatever the length; the CPU needs 100-470 ms). Falls back to the
    full CPU graph if the NPU model cannot load, or after any NPU error."""

    NPU_T = 1601  # feature frames of the 16 s window (+1)

    def __init__(self, threads: int = 2, model_dir=config.TYPHOON_ASR, npu: bool = False):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        meta = json.loads((model_dir / "typhoon_asr_ctc.json").read_text())
        self.vocab = (model_dir / "vocab.txt").read_text(encoding="utf-8").split("\n")[: meta["vocab_size"]]
        self.blank = meta["blank"]
        self.lock = threading.Lock()
        self.backend, self.npu_error, self.sess, self._it = "cpu", "", None, None
        if npu:
            try:
                self._load_npu(ort, so)
                self.backend = "npu"
            except Exception as e:  # missing model/runtime, NPU busy or out of memory: keep working on the CPU
                self.npu_error = f"{type(e).__name__}: {e}"[:200]
                self._it = None
        if self.backend == "cpu":
            self.sess = ort.InferenceSession(str(model_dir / "typhoon_asr_ctc.onnx"), so, providers=["CPUExecutionProvider"])

    def _load_npu(self, ort, so):
        from ai_edge_litert.interpreter import Interpreter, load_delegate

        d = config.TYPHOON_NPU
        self._pre = ort.InferenceSession(str(d / "pre.onnx"), so, providers=["CPUExecutionProvider"])
        # cache_dir: the compiled NPU graph is kept there; the first load of a model takes ~65 s, later loads 0.3 s
        cache = d / "cache"
        cache.mkdir(exist_ok=True)
        delegate = load_delegate(config.QNN_DELEGATE, {"backend_type": "htp", "htp_performance_mode": "2", "cache_dir": str(cache), "model_token": "typhoon_enc16"})
        it = Interpreter(model_path=str(d / "enc_16s.tflite"), experimental_delegates=[delegate])
        it.allocate_tensors()
        self._in, self._out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
        self._delegate, self._it = delegate, it  # keep the delegate alive as long as the interpreter

    def _logprobs(self, audio: np.ndarray) -> np.ndarray:
        if self._it is not None and len(audio) <= (self.NPU_T - 1) * 160:
            try:
                n = len(audio)
                feats = self._pre.run(None, {"audio": audio[None], "length": np.array([n], np.int64)})[0]
                nf = feats.shape[2]
                buf = np.zeros((1, 80, self.NPU_T), np.float32)  # shorter clips: zero features after the speech (as NeMo masks them)
                buf[:, :, :nf] = feats
                self._it.set_tensor(self._in, buf)
                self._it.invoke()
                return self._it.get_tensor(self._out)[0][: (nf + 7) // 8 + 1]
            except Exception as e:
                self.npu_error, self._it, self.backend = f"{type(e).__name__}: {e}"[:200], None, "cpu"
                import onnxruntime as ort

                self.sess = ort.InferenceSession(str(config.TYPHOON_ASR / "typhoon_asr_ctc.onnx"), providers=["CPUExecutionProvider"])
        if self.sess is None:  # NPU mode, clip longer than the window
            import onnxruntime as ort

            self.sess = ort.InferenceSession(str(config.TYPHOON_ASR / "typhoon_asr_ctc.onnx"), providers=["CPUExecutionProvider"])
        lp, n = self.sess.run(None, {"audio": audio[None], "length": np.array([len(audio)], np.int64)})
        return lp[0][: int(n[0])]

    def transcribe(self, pcm16: np.ndarray) -> tuple[str, float]:
        """(text, confidence). Confidence ~0.9+ for Thai speech, ~0.5 for other languages / noise."""
        pad = np.zeros(4800, np.float32)
        audio = np.concatenate([pad, np.asarray(pcm16, np.float32) / 32768.0, pad])
        with self.lock:
            lp = self._logprobs(audio)
        ids = lp.argmax(-1)
        emit = ids != self.blank
        conf = float(np.exp(lp.max(-1))[emit].mean()) if emit.any() else 0.0
        out, prev = [], -1
        for i in ids:
            if i != prev and i != self.blank:
                out.append(self.vocab[i])
            prev = i
        text = "".join(out).replace("▁", " ").replace("<s>", "").replace("</s>", "")
        return fix_thai(text), conf

    def transcribe_long(self, pcm16: np.ndarray, max_s: float = 15.0) -> tuple[str, float]:
        """Split anything longer than max_s at its quietest point (the CTC model is trained on short utterances)."""
        n = len(pcm16)
        if n <= max_s * SR:
            return self.transcribe(pcm16)
        texts, confs, pos = [], [], 0
        while pos < n:
            end = min(n, pos + int(max_s * SR))
            if end < n:
                lo, hi = end - 3 * SR, end - SR // 2  # look for the quietest 100 ms in the last 3 s
                win = np.abs(np.asarray(pcm16[lo:hi], np.int32)).reshape(-1)[: ((hi - lo) // 1600) * 1600].reshape(-1, 1600).mean(1)
                end = lo + int(win.argmin()) * 1600 + 800
            t, c = self.transcribe(pcm16[pos:end])
            if t:
                texts.append(t)
                confs.append(c)
            pos = end
        return " ".join(texts), float(np.mean(confs)) if confs else 0.0


class WhisperTH:
    """Thonburian distil-Whisper medium through faster-whisper (int8, CPU)."""

    def __init__(self, threads: int = 4, model_dir=config.WHISPER_TH, language: str = "th"):
        from faster_whisper import WhisperModel

        self.language = language  # "th" for the Thai-tuned models (they keep English words); "en" for the English model of English-only meetings
        self.model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)

    def transcribe(self, pcm16: np.ndarray, prompt: str = "") -> tuple[str, float, float]:
        """(text, avg_logprob, no_speech_prob) for one chunk of <= 30 s."""
        audio = np.concatenate([np.zeros(3200, np.float32), np.asarray(pcm16, np.float32) / 32768.0, np.zeros(3200, np.float32)])
        segs, _ = self.model.transcribe(
            # beam 2, one pass: measured on 51 s of Thai speech (4 cores) beam 5 + temperature fallback = 3.4 % CER in 145 s,
            # beam 2 = 3.4 % in 57 s, beam 1 = 4.2 % in 37 s. Loops / low confidence are caught afterwards by choose() (falls back to Typhoon).
            audio, language=self.language, task="transcribe", beam_size=2, temperature=0.0,
            compression_ratio_threshold=2.2, log_prob_threshold=-1.0, no_speech_threshold=0.6,
            # NO initial_prompt, on purpose: measured on Thai speech, any prompt (even two names) wrecks this distilled Thai Whisper
            # (error rate 3.4% without, 43% with names only, 49-94% with longer prompts). Names/terms are applied AFTER recognition instead.
            condition_on_previous_text=False, initial_prompt=None, vad_filter=False, without_timestamps=True)
        segs = list(segs)
        if not segs:
            return "", -9.0, 1.0
        text = "".join(s.text for s in segs)
        lp = float(np.mean([s.avg_logprob for s in segs]))
        ns = float(np.mean([s.no_speech_prob for s in segs]))
        return fix_thai(text), lp, ns


def _whisper_transcribe_many(self, pcms, batch_size: int = 8) -> list[tuple[str, float, float]]:
    """Several chunks decoded together. Decoding is limited by memory bandwidth (the same weights are read for every token), so a batch of
    chunks costs far less than the same chunks one by one: measured on 8 x 19.5 s of real meeting speech (4 cores): sequential 184 s,
    batch 4 = 95 s, batch 8 = 74 s, with IDENTICAL text. Result per chunk: (text, avg_logprob, no_speech_prob); the batched path does not
    compute no_speech_prob (0.0): silence is caught by choose() through empty text / low log-probability / the Typhoon cross-check."""
    from faster_whisper import BatchedInferencePipeline

    if getattr(self, "_batched", None) is None:
        self._batched = BatchedInferencePipeline(model=self.model)
    pad = np.zeros(3200, np.float32)
    parts, clips, pos, long_ones = [], [], 0, {}
    for k, p in enumerate(pcms):
        a = np.concatenate([pad, np.asarray(p, np.float32) / 32768.0, pad])
        if len(a) > 29.9 * SR:  # does not fit one Whisper window: the one-by-one path handles it
            long_ones[k] = None
            continue
        parts.append(a)
        clips.append({"start": pos / SR, "end": (pos + len(a)) / SR, "k": k})
        pos += len(a)
    out: list = [None] * len(pcms)
    if parts:
        segs, _ = self._batched.transcribe(
            np.concatenate(parts), clip_timestamps=[{"start": c["start"], "end": c["end"]} for c in clips], batch_size=batch_size,
            language=self.language, task="transcribe", beam_size=2, temperature=0.0, condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True)
        texts: dict[int, list] = {}
        for sg in segs:
            j = max((j for j, c in enumerate(clips) if c["start"] <= sg.start + 0.05), default=0)
            texts.setdefault(j, []).append(sg)
        for j, c in enumerate(clips):
            sg = texts.get(j)
            out[c["k"]] = (fix_thai("".join(x.text for x in sg)), float(np.mean([x.avg_logprob for x in sg])), 0.0) if sg else ("", -9.0, 1.0)
    for k in long_ones:
        out[k] = self.transcribe(pcms[k])
    return out


WhisperTH.transcribe_many = _whisper_transcribe_many


def speech_chunks(pcm16: np.ndarray, max_chunk_s: float = 28.0, min_silence_ms: int = 700):
    """Silero VAD (bundled with faster-whisper) -> chunks [(start_sample, end_sample)] of <= max_chunk_s, cut at pauses."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = np.asarray(pcm16, np.float32) / 32768.0
    opts = VadOptions(threshold=0.45, min_speech_duration_ms=250, max_speech_duration_s=max_chunk_s - 3,
                      min_silence_duration_ms=min_silence_ms, speech_pad_ms=300)
    stamps = get_speech_timestamps(audio, opts)
    chunks = []
    for s in stamps:
        a, b = int(s["start"]), int(s["end"])
        # merge neighbours separated by a short gap while the chunk stays below the limit
        if chunks and a - chunks[-1][1] < 1.2 * SR and b - chunks[-1][0] <= max_chunk_s * SR:
            chunks[-1] = (chunks[-1][0], b)
        else:
            chunks.append((a, b))
    return chunks


def fill_gaps(pcm16: np.ndarray, chunks, max_s: float = 24.0, margin_s: float = 0.3):
    """The voice-activity detector (Silero) misses speech under music / room noise (measured on a real meeting video: whole sentences).
    Returns extra [start, end] pieces (<= max_s, cut at quiet points) for every stretch that no chunk covers but that is clearly
    louder than the background. The worker keeps a piece only if the fast Typhoon pass finds words in it."""
    n = len(pcm16)
    fr = np.asarray(pcm16[: n // 320 * 320], np.float32).reshape(-1, 320)
    rms = np.sqrt((fr ** 2).mean(1)) + 1e-3
    thr = max(180.0, 2.5 * float(np.percentile(rms, 10)))
    covered = np.zeros(len(rms), bool)
    m = int(margin_s * 50)
    for c in chunks:
        covered[max(0, c[0] // 320 - m): c[1] // 320 + m + 1] = True
    active = (rms > thr) & ~covered
    # an "uncovered" piece: a run of frames (allowing gaps < 0.5 s) that are loud and not covered
    out, i, N = [], 0, len(rms)
    while i < N:
        if not active[i]:
            i += 1
            continue
        j, quiet = i, 0
        while j < N and not covered[j] and quiet < 25:
            quiet = 0 if rms[j] > thr else quiet + 1
            j += 1
        j -= quiet
        if (j - i) >= 40 and (rms[i:j] > thr).mean() >= 0.25:  # >= 0.8 s with real activity
            a, b = max(0, i - 8), min(N, j + 8)
            while b - a > max_s * 50:  # cut long pieces at the quietest 100 ms of the last 4 s
                lo, hi = a + int((max_s - 4) * 50), a + int(max_s * 50)
                cut = lo + int(np.argmin(np.convolve(rms[lo:hi], np.ones(5) / 5, "valid")))
                out.append([a * 320, cut * 320])
                a = cut
            out.append([a * 320, b * 320])
        i = max(j, i + 1)
    return out


class WhisperLive:
    """Whisper as the LIVE caption engine of an English-only meeting (small.en by default: a 30 s window costs ~1 s on 4 cores), with the
    interface of ThaiCTC (transcribe_long / transcribe -> (text, confidence)), so server.caption() can use either."""

    kind, backend = "whisper-en", "cpu"

    def __init__(self, threads: int = 4, model_dir=config.WHISPER_EN_LIVE, language: str = "en"):
        self.w = WhisperTH(threads=threads, model_dir=model_dir, language=language)

    def transcribe_long(self, pcm16: np.ndarray, max_s: float = 30.0) -> tuple[str, float]:
        text, lp, ns = self.w.transcribe(pcm16[: int(max_s * config.SR)])
        if not text or ns > 0.8 or repetition_ratio(text) > 2.4:
            return "", 0.0
        return text, float(np.exp(lp))

    transcribe = transcribe_long


def choose(whisper: tuple[str, float, float] | None, typhoon: tuple[str, float]) -> tuple[str, str, float]:
    """Pick the final text for a chunk -> (text, source, confidence).
    Whisper is the primary engine; fall back to Typhoon when Whisper loops, is empty, or is unsure while Typhoon is sure."""
    tt, tc = typhoon
    if whisper is None:
        return tt, "typhoon", tc
    wt, lp, ns = whisper
    if wt and tt and tc >= 0.8 and len(wt.replace(" ", "")) < 0.4 * len(tt.replace(" ", "")):  # the accurate engine dropped most of what Typhoon heard (seen: Qwen 0.6B "ที่ นี้ ที่ นั้น")
        return tt, "typhoon", tc
    if not wt or ns > 0.8:
        return (tt, "typhoon", tc) if tc >= 0.6 else ("", "silence", 0.0)
    if repetition_ratio(wt) > 2.4 or lp < -1.2:
        return (tt, "typhoon", tc) if tt and tc >= 0.8 else (wt, "whisper-low", float(np.exp(lp)))
    return wt, "whisper", float(np.exp(lp))
