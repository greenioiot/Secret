"""NECTEC Pathumma STT FastConformer-RNNT large (Thai-English, NeMo, CC-BY-4.0) in its own process, run with a NeMo venv
(config.NEMO_PY). Protocol as qwen_server.py: one JSON per line on stdin, one reply line on stdout prefixed with '@@'.
  {"cmd":"load","threads":8,"model":"/path/model.nemo"}             -> {"ok":true}
  {"cmd":"asr","files":["a.npy",...],"batch":8}  (float32 16 kHz)   -> {"ok":true,"texts":[...]}"""
import json
import logging
import os
import sys

import numpy as np

model = None


def reply(obj):
    sys.stdout.write("@@" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load(threads, path):
    global model
    import torch

    torch.set_num_threads(threads)
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(path, map_location="cpu").eval()


def asr(files, batch):
    auds = [np.load(f).astype(np.float32) for f in files]
    out = model.transcribe(auds, batch_size=batch, verbose=False)
    if isinstance(out, tuple):  # older NeMo: (best, all)
        out = out[0]
    return [(o.text if hasattr(o, "text") else str(o)).strip() for o in out]


def main():
    real_out = sys.stdout
    sys.stdout = sys.stderr  # NeMo prints progress / warnings to stdout: keep the protocol line clean
    global reply

    def reply(obj):  # noqa: F811
        real_out.write("@@" + json.dumps(obj, ensure_ascii=False) + "\n")
        real_out.flush()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req["cmd"] == "load":
                load(int(req.get("threads", 8)), req["model"])
                reply({"ok": True})
            elif req["cmd"] == "asr":
                reply({"ok": True, "texts": asr(req["files"], int(req.get("batch", 8)))})
            elif req["cmd"] == "quit":
                return
        except Exception as e:  # noqa: BLE001 - report, keep serving
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
