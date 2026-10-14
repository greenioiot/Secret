"""Text-to-speech helper process (spoken guidance for hands-free enrolment). Runs with projectAI's Python, which has Piper + TLTK
(this app's venv cannot build TLTK). Reuses projectAI's installed voices and Thai text normaliser READ-ONLY.
Protocol: one JSON per line on stdin {"text", "out"} -> writes a 22.05 kHz WAV to "out", answers {"ok": true, "seconds"} or {"ok": false, "error"}.
Voices: Thai th_TH-tsync2-medium (CC BY-NC-SA 3.0, non-commercial), English en_US-lessac-medium."""
import io
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_AI = Path(sys.argv[1])
sys.path.insert(0, str(PROJECT_AI / "app"))
import thai_speech  # noqa: E402  (projectAI: numbers, times, abbreviations -> Thai words; mixed Thai/English split)
from piper import PiperVoice  # noqa: E402

MODELS = PROJECT_AI / "models" / "piper"
TH = PiperVoice.load(str(MODELS / "th_TH" / "th_TH-tsync2-medium.onnx"))
EN = PiperVoice.load(str(next(MODELS.rglob("en_US-lessac-medium.onnx"))))
TH.phonemize("สวัสดี")  # loads TLTK's dictionary now, not on the first sentence
RATE = TH.config.sample_rate
THAI = re.compile(r"[฀-๿]")


def synth(voice, text: str) -> np.ndarray:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        voice.synthesize_wav(text, w)
    b.seek(0)
    with wave.open(b, "rb") as r:
        a = np.frombuffer(r.readframes(r.getnframes()), np.int16)
        if r.getframerate() != RATE:
            a = np.interp(np.arange(0, len(a), r.getframerate() / RATE), np.arange(len(a)), a).astype(np.int16)
    return a


def speak(text: str) -> np.ndarray:
    if not THAI.search(text):
        return synth(EN, text)
    text = text.replace("ๆ", "")
    parts = thai_speech.split_mixed(thai_speech.normalize(text))
    return np.concatenate([synth(TH if lang == "th" else EN, chunk) for lang, chunk in parts]) if parts else np.zeros(0, np.int16)


print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        pcm = speak(str(req["text"])[:600])
        with wave.open(req["out"], "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm.tobytes())
        print(json.dumps({"ok": True, "seconds": round(len(pcm) / RATE, 2)}), flush=True)
    except Exception as e:  # noqa: BLE001 - answer every request
        print(json.dumps({"ok": False, "error": str(e)[:200]}), flush=True)
