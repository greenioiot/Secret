"""Knowledge base for the meeting bot (RAG): documents -> chunks -> multilingual-e5 vectors -> hybrid search.

* Text comes from .txt/.md/.csv (UTF-8 or Thai Windows encodings), .pdf (pypdf) or .docx (read as a zip); everything stays on this device.
* Vectors: Xenova/multilingual-e5-small int8 (ONNX, CPU, ~0.5 GB while loaded, unloaded after 10 idle minutes). Thai top-1 5/5 in testing.
* Search: dense cosine + a character-trigram overlap score (the trigram part keeps names, numbers and rare words exact), then keep only
  results close to the best one. If the model files are missing, search still works with the trigram score alone.
* Summaries of the latest finished meetings are searchable too (so the bot can answer "what did we decide last time?").
"""
from __future__ import annotations

import io
import re
import threading
import time
import uuid
import zipfile

import numpy as np

import config
import store

KB = config.DATA / "kb"
MODEL = config.ROOT / "models/embed/model_quantized.onnx"
TOKENIZER = config.ROOT / "models/embed/tokenizer.json"
MAX_FILE = 2_000_000
MAX_DOCS = 60
MAX_CHUNKS = 5000
CHUNK, OVERLAP = 600, 100
IDLE_UNLOAD_S = 600


# ------------------------------------------------------------------ text extraction
def extract_text(name: str, data: bytes) -> str:
    low = name.lower()
    if low.endswith(".pdf"):
        from pypdf import PdfReader

        r = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in r.pages)
    if low.endswith(".docx"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
        xml = re.sub(r"</w:p>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    for enc in ("utf-8-sig", "cp874", "tis-620", "latin-1"):  # Thai text files from Windows are often cp874
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def clean(text: str) -> str:
    text = text.replace("\r", "\n").replace("​", "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, size: int = CHUNK, overlap: int = OVERLAP) -> list[str]:
    """Paragraph-aware windows. Thai has no spaces, so long paragraphs are cut at the last space/newline/sentence mark before `size`."""
    out, buf = [], ""
    for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        if len(buf) + len(para) + 2 <= size:
            buf = (buf + "\n\n" + para).strip()
            continue
        if buf:
            out.append(buf)
            buf = ""
        while len(para) > size:
            cut = max(para.rfind(c, size // 2, size) for c in ("\n", ". ", " ", "ฯ"))
            cut = cut + 1 if cut > 0 else size
            out.append(para[:cut].strip())
            para = para[max(0, cut - overlap):].strip() if overlap and cut > overlap else para[cut:].strip()
        buf = para
    if buf:
        out.append(buf)
    return [c for c in out if len(c) >= 12]


def trigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text.lower())
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))} or {t}


# ------------------------------------------------------------------ embedder (lazy, unloads when idle)
class Embedder:
    def __init__(self):
        self.lock = threading.RLock()
        self.sess = self.tok = None
        self.last = 0.0

    def available(self) -> bool:
        return MODEL.exists() and TOKENIZER.exists()

    def _load(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(TOKENIZER))
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding()

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        with self.lock:
            if self.sess is None:
                self._load()
            self.last = time.time()
            out = []
            for i in range(0, len(texts), 16):
                enc = self.tok.encode_batch([f"{kind}: {t}" for t in texts[i:i + 16]])
                ids = np.array([e.ids for e in enc], np.int64)
                mask = np.array([e.attention_mask for e in enc], np.int64)
                h = self.sess.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
                v = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
                out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
            return np.vstack(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)

    def maybe_unload(self):
        with self.lock:
            if self.sess is not None and time.time() - self.last > IDLE_UNLOAD_S:
                self.sess = self.tok = None
                import gc

                gc.collect()

    def loaded(self) -> bool:
        return self.sess is not None


EMB = Embedder()


# ------------------------------------------------------------------ collections: the global knowledge base and each meeting's own materials
class Collection:
    """Documents -> chunks -> vectors on disk under `base`/docs. The global knowledge base also searches the latest finished meetings."""

    def __init__(self, base, with_meetings: bool = False, max_docs: int = MAX_DOCS):
        self.base, self.with_meetings, self.max_docs = base, with_meetings, max_docs
        self._cache: dict = {"stamp": None, "items": []}

    @property
    def docs_dir(self):
        return self.base / "docs"

    def _paths(self, did: str):
        return self.docs_dir / f"{did}.json", self.docs_dir / f"{did}.npy"

    def list_docs(self) -> list[dict]:
        out = []
        for f in sorted(self.docs_dir.glob("d*.json")) if self.docs_dir.exists() else []:
            d = store.read_json(f)
            if d:
                out.append({k: d[k] for k in ("id", "name", "added", "chars", "enabled")} | {"chunks": len(d["chunks"]), "vectors": (self.docs_dir / f"{d['id']}.npy").exists()})
        return out

    def items(self) -> list[dict]:
        """Chunks of every enabled document (+ the latest finished meetings for the global base), with vectors; cached until something changes."""
        files = sorted(self.docs_dir.glob("d*.json")) if self.docs_dir.exists() else []
        stamp = (tuple((f.name, f.stat().st_mtime_ns) for f in files), tuple((m["id"], m.get("status")) for m in store.list_meetings()[:20]) if self.with_meetings else ())
        if self._cache["stamp"] == stamp:
            return self._cache["items"]
        items = []
        for f in files:
            d = store.read_json(f)
            if not d or not d.get("enabled", True):
                continue
            vecs = None
            try:
                vecs = np.load(self.docs_dir / f"{d['id']}.npy").astype(np.float32)
            except (OSError, ValueError):
                pass
            for i, c in enumerate(d["chunks"]):
                items.append({"doc": d["id"], "name": d["name"], "text": c, "vec": vecs[i] if vecs is not None and i < len(vecs) else None, "tri": trigrams(c)})
        if self.with_meetings:
            for m in store.list_meetings()[:20]:  # what was decided in earlier meetings
                p = store.mdir(m["id"]) / "summary.md"
                if m.get("status") == "done" and p.exists():
                    for c in chunk_text(f"{m.get('title', '')} ({m.get('created', '')[:10]})\n" + p.read_text(encoding="utf-8")):
                        items.append({"doc": "m:" + m["id"], "name": f"ประชุม: {m.get('title', '')} {m.get('created', '')[:10]}", "text": c, "vec": None, "tri": trigrams(c)})
        if EMB.available() and any(i["vec"] is None for i in items) and len(items) <= MAX_CHUNKS:
            todo = [i for i in items if i["vec"] is None]
            try:
                vs = EMB.embed([i["text"] for i in todo])
                for i, v in zip(todo, vs):
                    i["vec"] = v
            except Exception:  # noqa: BLE001 - lexical search still works
                pass
        self._cache.update(stamp=stamp, items=items)
        return items

    def add_document(self, name: str, data: bytes | str) -> dict:
        if len(self.list_docs()) >= self.max_docs:
            raise ValueError(f"At most {self.max_docs} documents.")
        raw = data.encode("utf-8") if isinstance(data, str) else data
        if len(raw) > MAX_FILE:
            raise ValueError("The file is larger than 2 MB.")
        text = clean(extract_text(name, raw))
        if len(text) < 20:
            raise ValueError("No readable text was found in this file.")
        chunks = chunk_text(text)
        if len(chunks) > 800:
            raise ValueError("The document is too long (over 800 passages): split it into parts.")
        did = "d" + uuid.uuid4().hex[:8]
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        jp, npy = self._paths(did)
        if EMB.available():
            np.save(npy, EMB.embed(chunks).astype(np.float16))
        store.write_json(jp, {"id": did, "name": name[:100], "added": time.strftime("%F %T"), "chars": len(text), "enabled": True, "chunks": chunks, "text": text[:400000]})
        return next(d for d in self.list_docs() if d["id"] == did)

    def texts(self) -> list[str]:
        """Full text of every enabled document (for key-term extraction)."""
        out = []
        for f in sorted(self.docs_dir.glob("d*.json")) if self.docs_dir.exists() else []:
            d = store.read_json(f)
            if d and d.get("enabled", True):
                out.append(d.get("text") or "\n\n".join(d["chunks"]))
        return out

    def delete_document(self, did: str) -> bool:
        if not re.fullmatch(r"d[0-9a-f]{8}", did):
            return False
        jp, npy = self._paths(did)
        existed = jp.exists()
        jp.unlink(missing_ok=True)
        npy.unlink(missing_ok=True)
        return existed

    def set_enabled(self, did: str, on: bool) -> bool:
        if not re.fullmatch(r"d[0-9a-f]{8}", did):
            return False
        jp, _ = self._paths(did)
        d = store.read_json(jp)
        if not d:
            return False
        d["enabled"] = bool(on)
        store.write_json(jp, d)
        return True

    def search(self, query: str, k: int = 4, doc_ids: list[str] | None = None) -> list[dict]:
        return search_items([i for i in self.items() if not doc_ids or i["doc"] in doc_ids], query, k)


def search_items(items: list[dict], query: str, k: int = 4) -> list[dict]:
    """Best passages for a question: [{score, name, doc, text}], best first, only those close to the best one."""
    if not items or not query.strip():
        return []
    qt = trigrams(query)
    lex = np.array([len(qt & i["tri"]) / (len(qt) ** 0.5 * len(i["tri"]) ** 0.5) for i in items], np.float32)
    dense = None
    if EMB.available() and any(i["vec"] is not None for i in items):
        try:
            q = EMB.embed([query], "query")[0]
            dense = np.array([float(q @ i["vec"]) if i["vec"] is not None else 0.0 for i in items], np.float32)
        except Exception:  # noqa: BLE001
            dense = None

    def norm(x):
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(x)

    score = 0.7 * norm(dense) + 0.3 * norm(lex) if dense is not None else norm(lex)
    order = np.argsort(-score)[:k]
    best = float(score[order[0]]) if len(order) else 0.0
    out = []
    for j in order:
        # a floor on the REAL similarity (a fixed top-k would return junk for unrelated questions); a mid-range dense hit also needs some shared words
        raw_ok = ((dense[j] >= 0.85) or (dense[j] >= 0.80 and lex[j] >= 0.03)) if dense is not None else (lex[j] >= 0.08)
        if raw_ok and score[j] >= best - 0.30:
            it = items[j]
            out.append({"score": round(float(score[j]), 3), "sim": round(float(dense[j]) if dense is not None else float(lex[j]), 3), "name": it["name"], "doc": it["doc"], "text": it["text"]})
    return out


GLOBAL = Collection(KB, with_meetings=True)
_materials: dict[str, Collection] = {}


def materials(mid: str) -> Collection:
    """The documents uploaded for ONE meeting (agenda, background papers...)."""
    base = store.mdir(mid) / "materials"
    if mid not in _materials:
        _materials[mid] = Collection(base, with_meetings=False, max_docs=20)
    return _materials[mid]


def search_meeting(mid: str, query: str, k: int = 4, include_meetings: bool = True) -> list[dict]:
    """Global knowledge base + this meeting's own materials, searched together. `include_meetings` adds the summaries of earlier meetings
    (useful for the bot's "what did we decide last time?", but NOT for summarising: another meeting's facts must never leak into this one)."""
    items = [i for i in GLOBAL.items() if include_meetings or not str(i["doc"]).startswith("m:")] + (materials(mid).items() if (store.mdir(mid) / "materials").exists() else [])
    return search_items(items, query, k)


# the original module-level API (global knowledge base)
list_docs = GLOBAL.list_docs
add_document = GLOBAL.add_document
delete_document = GLOBAL.delete_document
set_enabled = GLOBAL.set_enabled
search = GLOBAL.search


def status() -> dict:
    docs = list_docs()
    return {"docs": docs, "model": EMB.available(), "loaded": EMB.loaded(), "chunks": sum(d["chunks"] for d in docs),
            "limits": {"max_docs": MAX_DOCS, "max_file_mb": MAX_FILE / 1e6}}
