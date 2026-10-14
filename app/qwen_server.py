"""Typhoon ASR Qwen (Qwen3-ASR + contextual biasing) in its own process, run with .venv-qwen (torch + transformers).
Protocol: one JSON object per line on stdin, one reply line on stdout prefixed with '@@' (library noise goes to stderr).
  {"cmd":"load","threads":8,"model":"typhoon-ai/typhoon-asr-qwen-1.7b-ctx"}                 -> {"ok":true}
  {"cmd":"asr","files":["a.npy",...],"bias":"name1 name2","batch":12}   (float32 16 kHz mono arrays) -> {"ok":true,"texts":[...]}
Measured on this CPU (8 cores, fp32): 1.7B = about 1.7x real time with batches of 12; bf16 / int8 are slower or worse."""
import json
import sys

import numpy as np

proc = model = None


def reply(obj):
    sys.stdout.write("@@" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load(threads, mid):
    global proc, model
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.set_num_threads(threads)
    proc = AutoProcessor.from_pretrained(mid)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(mid, dtype=torch.float32).eval()


def asr(files, bias, batch):
    import torch

    auds = [np.load(f) for f in files]
    order = sorted(range(len(auds)), key=lambda i: len(auds[i]))  # similar lengths together: less padding
    texts = [""] * len(auds)
    system = (bias.strip() + "\n" if bias.strip() else "") + "Thai"
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        convs = [[{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": [{"type": "audio", "audio": auds[i]}]}] for i in idx]
        inp = proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=440)
        dec = proc.decode(out[:, inp["input_ids"].shape[1]:], return_format="transcription_only")
        for i, t in zip(idx, dec):
            texts[i] = t.strip()
    return texts


def main():
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req["cmd"] == "load":
                load(int(req.get("threads", 8)), req["model"])
                reply({"ok": True})
            elif req["cmd"] == "asr":
                reply({"ok": True, "texts": asr(req["files"], req.get("bias", ""), int(req.get("batch", 12)))})
            elif req["cmd"] == "quit":
                return
        except Exception as e:  # noqa: BLE001 - report, keep serving
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})


if __name__ == "__main__":
    main()
