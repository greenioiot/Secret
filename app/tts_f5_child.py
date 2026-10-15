"""Text-to-speech helper with NECTEC Pathumma F5-TTS (MIT; base SWivid/F5-TTS), run with .venv-f5 (torch CPU + f5-tts).
Same protocol as tts_child.py: one JSON per line {"text", "out"} -> 24 kHz WAV, {"ok": true, "seconds"} / {"ok": false, "error"}.
F5-TTS copies the voice of a short reference clip: models/pathumma/f5-ref.wav + f5-ref.txt (a FLEURS Thai test speaker,
CC-BY-4.0), never a meeting participant. Numbers / abbreviations are spelled out first with projectAI's Thai normaliser when present."""
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

OUT, sys.stdout = sys.stdout, sys.stderr  # the libraries print to stdout: only protocol lines go to the real one
ROOT = Path(sys.argv[1])
PROJECT_AI = Path(sys.argv[2]) if len(sys.argv) > 2 else None
M = ROOT / "models/pathumma"
NFE = int(os.environ.get("ZECRET_F5_NFE", "8"))  # flow-matching steps (library default 32). Measured here: 16 steps 90 s, 8 steps 48 s for 4.4 s of speech, both read back word-perfect by the ASR
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("ZECRET_F5_THREADS", "8")))
from f5_tts.api import F5TTS  # noqa: E402

normalize = None
if PROJECT_AI and (PROJECT_AI / "app/thai_speech.py").exists():
    sys.path.insert(0, str(PROJECT_AI / "app"))
    try:
        import thai_speech  # noqa: E402

        normalize = thai_speech.normalize
    except Exception:  # noqa: BLE001 - its dependencies may be missing in this venv: speak the text as written
        normalize = None
TTS = F5TTS(model=os.environ.get("ZECRET_F5_ARCH", "F5TTS_v1_Base"), ckpt_file=str(M / "f5-tts/model_ema.safetensors"), vocab_file=str(M / "f5-tts/vocab.txt"), device="cpu")
REF, REF_TEXT = str(M / "f5-ref.wav"), (M / "f5-ref.txt").read_text(encoding="utf-8").strip()


def speak(text: str):
    if normalize:
        try:
            text = normalize(text)
        except Exception:  # noqa: BLE001
            pass
    wav, sr, _ = TTS.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=text.replace("ๆ", ""), nfe_step=NFE, remove_silence=False, show_info=lambda *a, **k: None, progress=None)
    return (np.clip(np.asarray(wav, np.float32), -1, 1) * 32767).astype(np.int16), sr


if __name__ == "__main__":
    speak("สวัสดี")  # warm up (loads the vocoder, first-call compilation)
    print(json.dumps({"ready": True}), file=OUT, flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            pcm, sr = speak(str(req["text"])[:600])
            with wave.open(req["out"], "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm.tobytes())
            print(json.dumps({"ok": True, "seconds": round(len(pcm) / sr, 2)}), file=OUT, flush=True)
        except Exception as e:  # noqa: BLE001 - answer every request
            print(json.dumps({"ok": False, "error": str(e)[:200]}), file=OUT, flush=True)
