"""Domain vocabulary packs (Banking, Insurance, Regulator, Law, your own): a pack is a list of terms written the way you want to read them
plus the Thai spellings a recogniser tends to produce for them. Active packs (global default, or chosen per meeting) are used for
  * live captions and transcripts: "Thai spelling -> term" rules (applied after the word collection),
  * the spelling repair after the transcript, the summary's spelling list, and the 'Best' ASR context list: the terms themselves.
Stored in data/vocab/<id>.json; the starter packs are copied there once (your edits are never overwritten)."""
from __future__ import annotations

import csv
import io
import re
import time

import config
import store

DIR = config.DATA / "vocab"
CATS = ("term", "acronym", "org", "law", "name")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
MAX_PACKS, MAX_ENTRIES = 40, 5000


def _file(pid: str):
    if not ID_RE.match(pid):
        raise ValueError("bad pack id")
    return DIR / f"{pid}.json"


def seed():
    """Copy the starter packs that are not on disk yet (a pack you deleted stays deleted: it is remembered in .removed)."""
    import vocab_seed

    DIR.mkdir(parents=True, exist_ok=True)
    removed = set((store.read_json(DIR / ".removed") or []))
    for pid, name, desc, rows in vocab_seed.PACKS:
        if _file(pid).exists() or pid in removed:
            continue
        entries = [{"term": t, "variants": list(v), "cat": c, "note": n} for t, v, c, n in rows]
        store.write_json(_file(pid), {"id": pid, "name": name, "desc": desc, "builtin": True, "enabled": False, "entries": entries,
                                      "updated": time.strftime("%F %T"), "additions": sorted(vocab_seed.ADDITIONS), "rev": 1})
    for tag, packs in vocab_seed.ADDITIONS.items():
        for pid, rows in packs.items():
            p = load(pid) if _file(pid).exists() else None
            if not p or tag in (p.get("additions") or []):
                continue
            have = {e["term"].casefold() for e in p["entries"]}
            p["entries"] += [{"term": t, "variants": list(v), "cat": c, "note": n} for t, v, c, n in rows if t.casefold() not in have]
            p["additions"] = sorted(set(p.get("additions") or []) | {tag})
            save(p)


def load(pid: str) -> dict | None:
    return store.read_json(_file(pid))


def save(pack: dict):
    """Every save bumps the pack's revision: an editor that started from an older revision is refused (server checks 'rev')."""
    pack["updated"] = time.strftime("%F %T")
    pack["rev"] = int(pack.get("rev") or 0) + 1
    store.write_json(_file(pack["id"]), pack)


class StaleEdit(Exception):
    pass


def check_rev(pack: dict, rev) -> None:
    if rev is not None and int(rev) != int(pack.get("rev") or 0):
        raise StaleEdit("this pack was changed in another tab or by another administrator: it has been reloaded, check it and try again")


def list_packs() -> list[dict]:
    seed()
    out = []
    for f in sorted(DIR.glob("*.json")):
        p = store.read_json(f)
        if p and p.get("id"):
            out.append({"id": p["id"], "name": p.get("name", p["id"]), "desc": p.get("desc", ""), "builtin": bool(p.get("builtin")), "context": p.get("context", ""),
                        "enabled": bool(p.get("enabled")), "count": len(p.get("entries", [])), "updated": p.get("updated", ""), "rev": int(p.get("rev") or 0)})
    out.sort(key=lambda p: (not p["builtin"], p["name"].casefold()))
    return out


def clean_entry(e: dict) -> dict:
    import asr

    term = " ".join(str(e.get("term", "")).split())[:80]
    if not term:
        raise ValueError("the term is empty")
    variants, seen = [], set()
    raw = e.get("variants", [])
    if isinstance(raw, str):
        raw = re.split(r"[,\n|;]", raw)
    for v in raw:
        v = " ".join(str(v).split())[:60]
        if not v or v == term or v in seen:
            continue
        why = asr.rule_problem(v, term)
        if why:
            raise ValueError(f"“{v}”: {why}")
        seen.add(v)
        variants.append(v)
    cat = e.get("cat") if e.get("cat") in CATS else "term"
    raw = e.get("aliases", [])
    if isinstance(raw, str):
        raw = re.split(r"[,\n|;]", raw)
    aliases = []
    for a in raw:  # other correct ways to write the term (Thai name, English name, short form): shown and searched, never used as replace rules
        a = " ".join(str(a).split())[:80]
        if a and a != term and a not in aliases:
            aliases.append(a)
    return {"term": term, "variants": variants, "aliases": aliases[:10], "cat": cat, "note": str(e.get("note", "")).strip()[:200]}


def upsert(pid: str, entry: dict, old_term: str | None = None) -> dict:
    p = load(pid)
    if not p:
        raise KeyError(pid)
    e = clean_entry(entry)
    ents = p["entries"]
    key = (old_term or e["term"]).casefold()
    for i, x in enumerate(ents):
        if x["term"].casefold() == key:
            ents[i] = e
            break
    else:
        if len(ents) >= MAX_ENTRIES:
            raise ValueError(f"a pack holds at most {MAX_ENTRIES} entries")
        ents.append(e)
    save(p)
    return e


def delete_entry(pid: str, term: str) -> bool:
    p = load(pid)
    if not p:
        raise KeyError(pid)
    n = len(p["entries"])
    p["entries"] = [x for x in p["entries"] if x["term"].casefold() != term.casefold()]
    save(p)
    return len(p["entries"]) < n


def create(name: str, desc: str = "") -> dict:
    seed()
    if len(list_packs()) >= MAX_PACKS:
        raise ValueError(f"at most {MAX_PACKS} packs")
    name = " ".join(name.split())[:60]
    if not name:
        raise ValueError("the pack needs a name")
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "pack"
    pid, n = base, 2
    while _file(pid).exists():
        pid, n = f"{base}-{n}", n + 1
    p = {"id": pid, "name": name, "desc": desc.strip()[:300], "builtin": False, "enabled": False, "entries": []}
    save(p)
    return p


def delete_pack(pid: str):
    f = _file(pid)
    p = load(pid)
    if not p:
        raise KeyError(pid)
    f.unlink()
    if p.get("builtin"):  # remember it, so it is not re-seeded
        removed = set(store.read_json(DIR / ".removed") or [])
        removed.add(pid)
        store.write_json(DIR / ".removed", sorted(removed))


def reset_builtin(pid: str):
    """Restore a starter pack to its original list (your additions are lost)."""
    import vocab_seed

    for sid, name, desc, rows in vocab_seed.PACKS:
        if sid == pid:
            old = load(pid) or {}
            rows = list(rows) + [r for packs in vocab_seed.ADDITIONS.values() for r in packs.get(pid, [])]
            ents, seen = [], set()
            for t, v, c, n in rows:
                if t.casefold() not in seen:
                    seen.add(t.casefold())
                    ents.append({"term": t, "variants": list(v), "cat": c, "note": n})
            save({"id": pid, "name": name, "desc": desc, "builtin": True, "enabled": bool(old.get("enabled")), "context": old.get("context", ""),
                  "entries": ents, "additions": sorted(vocab_seed.ADDITIONS), "rev": old.get("rev", 0)})
            removed = set(store.read_json(DIR / ".removed") or [])
            if pid in removed:
                removed.discard(pid)
                store.write_json(DIR / ".removed", sorted(removed))
            return
    raise KeyError(pid)


MAX_IMPORT_BYTES = 1_000_000
MAX_IMPORT_LINES = MAX_ENTRIES


def import_bytes(pid: str, data: bytes, replace: bool = False) -> dict:
    """An uploaded file: at most 1 MB, UTF-8 (with or without BOM), CSV or TSV, at most 5000 lines; each entry is validated like a typed one."""
    if len(data) > MAX_IMPORT_BYTES:
        raise ValueError(f"the file is too large ({len(data) / 1e6:.1f} MB): at most {MAX_IMPORT_BYTES // 1_000_000} MB")
    if b"\x00" in data:
        raise ValueError("this is not a text file (CSV or TSV saved as UTF-8)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        line = data[: e.start].count(b"\n") + 1
        raise ValueError(f"the file is not UTF-8 (line {line}): in Excel use “Save as → CSV UTF-8”") from None
    return import_text(pid, text, replace)


def import_text(pid: str, text: str, replace: bool = False) -> dict:
    """CSV / TSV with a header or without: term, thai spellings (separated by | or ,), category, note. Returns counts and the rejected lines."""
    p = load(pid)
    if not p:
        raise KeyError(pid)
    text = text.lstrip("﻿")
    if len(text.encode("utf-8")) > MAX_IMPORT_BYTES:
        raise ValueError(f"the text is too large: at most {MAX_IMPORT_BYTES // 1_000_000} MB")
    if text.count("\n") >= MAX_IMPORT_LINES + 1:
        raise ValueError(f"too many lines: at most {MAX_IMPORT_LINES}")
    dialect = csv.excel_tab if text.count("\t") > text.count(",") else csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    if rows and rows[0] and rows[0][0].strip().lower() in ("term", "english", "canonical", "word"):
        rows = rows[1:]
    ok, bad = 0, []
    keep = {} if replace else {x["term"].casefold(): x for x in p["entries"]}
    for i, r in enumerate(rows, 1):
        if not r or not "".join(r).strip():
            continue
        try:
            if len(r) > 5:
                raise ValueError(f"{len(r)} columns: expected term, Thai spellings, category, note, aliases")
            e = clean_entry({"term": r[0], "variants": r[1] if len(r) > 1 else "", "cat": (r[2].strip().lower() if len(r) > 2 else "term"),
                             "note": r[3] if len(r) > 3 else "", "aliases": r[4] if len(r) > 4 else ""})
            keep[e["term"].casefold()] = e
            ok += 1
        except ValueError as ex:
            bad.append({"line": i, "text": ",".join(r)[:80], "why": str(ex)})
    if len(keep) > MAX_ENTRIES:
        raise ValueError(f"a pack holds at most {MAX_ENTRIES} entries")
    p["entries"] = list(keep.values())
    save(p)
    return {"imported": ok, "rejected": bad[:50], "total": len(p["entries"])}


def export_csv(pid: str) -> str:
    p = load(pid)
    if not p:
        raise KeyError(pid)
    out = io.StringIO()
    out.write("﻿")
    w = csv.writer(out)
    w.writerow(["term", "thai_spellings", "category", "note", "aliases"])
    for e in p["entries"]:
        w.writerow([e["term"], "|".join(e["variants"]), e["cat"], e["note"], "|".join(e.get("aliases") or [])])
    return out.getvalue()


# ---------------------------------------------------------------- using the packs
def active_ids(state: dict | None = None) -> list[str]:
    """Packs of a meeting: its own choice (state['vocab_packs']) or, when it has none, the packs switched on by default."""
    packs = list_packs()
    exist = {p["id"] for p in packs}
    chosen = (state or {}).get("vocab_packs")
    if isinstance(chosen, list):
        return [i for i in chosen if i in exist]
    return [p["id"] for p in packs if p["enabled"]]


def rules(ids: list[str]) -> list[tuple[str, str]]:
    """(Thai spelling, term) pairs of the given packs, unsafe or ambiguous ones dropped; first pack wins on a clash."""
    seen: dict[str, str] = {}
    for pid in ids:
        p = load(pid)
        for e in (p or {}).get("entries", []):
            for v in e.get("variants", []):
                seen.setdefault(v, e["term"])
    import vocab_meeting

    for g in vocab_meeting.GUARD:  # a longer word that holds a spelling stays as it is (textnorm.apply_rules: longest match wins)
        if g not in seen and any(v in g for v in seen):
            seen[g] = g
    return list(seen.items())


def contexts(ids: list[str], limit: int = 600) -> str:
    """Domain descriptions of the packs (e.g. 'Life insurance company; OIC = regulator'): background for the summary, within a small budget."""
    out = []
    for pid in ids:
        c = ((load(pid) or {}).get("context") or "").strip()
        if c:
            out.append(f"- {(load(pid) or {}).get('name', pid)}: {c}")
    return "\n".join(out)[:limit]


GENERAL_PACKS = {"meeting", "business-services"}  # packs of everyday business English: spelling rules and summary terms yes, fuzzy repair / ASR bias only for acronyms


def terms(ids: list[str], limit: int = 400, skip_thai_words: bool = False) -> list[str]:
    """The terms of the packs. skip_thai_words: leave out plain Thai words (สินเชื่อ, มาตรา ...): matching them fuzzily could 'repair' ordinary words."""
    out, seen = [], set()
    for pid in ids:
        for e in (load(pid) or {}).get("entries", []):
            t = e["term"]
            if skip_thai_words and e.get("cat") == "term" and not re.search(r"[A-Za-z]", t):
                continue
            if skip_thai_words and pid in GENERAL_PACKS and e.get("cat") != "acronym":
                continue  # everyday words (Plan, Cost, Lead ...): a one-letter fuzzy match would turn Plant / Most / Read into them
            if t.casefold() not in seen and 2 <= len(t) <= 40:
                seen.add(t.casefold())
                out.append(t)
    return out[:limit]


# ---------------------------------------------------------------- term suggestions from transcript review (inactive until an admin approves them)
SUGGEST_FILE = config.DATA / "vocab_suggestions.json"
MAX_SUGGESTIONS = 500


def suggestions(status: str = "pending") -> list[dict]:
    items = (store.read_json(SUGGEST_FILE) or {}).get("items", [])
    return [s for s in items if not status or s["status"] == status]


def add_suggestion(term: str, spelling: str, mid: str, by: str, pack: str | None = None, source: str = "review") -> dict | None:
    """A reviewer's correction becomes a SUGGESTED pack entry (term + the Thai spelling the recogniser wrote). It changes nothing until an
    administrator approves it into a pack; corrections themselves stay meeting-specific."""
    if len(re.sub(r"\W", "", term)) < 2:
        raise ValueError("the term needs at least 2 letters")
    e = clean_entry({"term": term, "variants": [spelling] if spelling else [], "cat": "acronym" if re.fullmatch(r"[A-Z0-9&.-]{2,12}", term.strip()) else "term"})
    if pack and not (ID_RE.match(pack) and _file(pack).exists()):
        pack = None
    data = store.read_json(SUGGEST_FILE) or {"items": []}
    for s in data["items"]:
        if s["status"] == "pending" and s["term"].casefold() == e["term"].casefold() and s["variants"] == e["variants"]:
            s["count"] += 1
            s["meetings"] = sorted(set(s["meetings"]) | {mid})[-20:]
            s["pack"] = s.get("pack") or pack
            store.write_json(SUGGEST_FILE, data)
            return s
    if sum(1 for s in data["items"] if s["status"] == "pending") >= MAX_SUGGESTIONS:
        return None
    s = {"id": f"s{int(time.time() * 1000):x}", "term": e["term"], "variants": e["variants"], "cat": e["cat"], "pack": pack, "meetings": [mid], "by": by,
         "at": time.strftime("%F %T"), "count": 1, "source": source, "status": "pending"}
    data["items"] = [x for x in data["items"] if x["status"] == "pending" or x.get("decided_at", "") >= time.strftime("%F", time.localtime(time.time() - 90 * 86400))] + [s]
    store.write_json(SUGGEST_FILE, data)
    return s


def decide_suggestion(sid: str, approve: bool, by: str, pack: str | None = None, term: str | None = None, variants=None) -> dict:
    data = store.read_json(SUGGEST_FILE) or {"items": []}
    s = next((x for x in data["items"] if x["id"] == sid and x["status"] == "pending"), None)
    if not s:
        raise KeyError(sid)
    if approve:
        pid = pack or s.get("pack")
        p = load(pid) if pid and ID_RE.match(pid) else None
        if not p:
            raise ValueError("choose the pack the term goes into")
        new = clean_entry({"term": term or s["term"], "variants": variants if variants is not None else s["variants"], "cat": s["cat"],
                           "note": f"suggested in review ({', '.join(s['meetings'][:3])})"})
        old = next((x for x in p["entries"] if x["term"].casefold() == new["term"].casefold()), None)
        if old:  # merge the spelling into the existing entry
            old["variants"] = old["variants"] + [v for v in new["variants"] if v not in old["variants"]]
            upsert(pid, old)
        else:
            upsert(pid, new)
        s["pack"] = pid
    s.update(status="approved" if approve else "rejected", decided_by=by, decided_at=time.strftime("%F %T"))
    store.write_json(SUGGEST_FILE, data)
    return s
