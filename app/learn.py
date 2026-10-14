"""Learning from a meeting's agenda and documents (no model training: three safe, inspectable uses).

1. KEY TERMS: names, acronyms and repeated domain words found in the documents, the agenda and the participant list.
2. SPEECH RECOGNITION: the best terms are given to Whisper as its initial prompt (it then spells them the way the documents do), and a
   conservative fuzzy pass repairs near-misses in the transcript ("แผล" -> "แผน"): every change is logged and the original text is kept.
3. SUMMARY: the agenda, the terms and the relevant passages of the documents go into the summary prompt, and an agenda-coverage check
   tells the summary which agenda items were really discussed (so it does not invent decisions for items nobody talked about).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

import kb
import store

THAI = re.compile(r"[ก-๛]+")
NOT_START = set("ะาำิีึืฺุูัํ็่้๊๋์ๆฯ")          # dependent vowels, tone marks, repetition mark: a word cannot start with these
NOT_END = set("เแโใไ")                             # leading vowels: a word cannot end with these
HONORIFIC = re.compile(r"(?:คุณ|นาย|นางสาว|นาง|ดร\.|ผศ\.|รศ\.|ศ\.|พี่|น้อง|อาจารย์|อ\.)\s*([ก-๛]{2,15})")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*")
EN_STOP = set("the and for with this that from have will are was were not but you your our their they them his her its has had can may all any one two new use used using per via also into out about over after before during between more most other such than then there these those what when where which who why how".split())
TH_STOP = ["การประชุม", "ที่ประชุม", "ดังนั้น", "เนื่องจาก", "ซึ่งเป็น", "อย่างไรก็ตาม", "ประเด็นสำคัญ", "ผู้เข้าร่วม", "ดำเนินการ"]  # too generic to be worth teaching


def _thai_candidates(text: str, max_chars: int = 150_000) -> Counter:
    c: Counter = Counter()
    seen = 0
    for run in THAI.findall(text):
        if len(run) < 4:
            continue
        seen += len(run)
        if seen > max_chars:
            break
        for ln in range(4, 15):
            for i in range(len(run) - ln + 1):
                s = run[i:i + ln]
                if s[0] not in NOT_START and s[-1] not in NOT_END:
                    c[s] += 1
    return c


def _maximal(cands: dict[str, int]) -> list[tuple[str, int]]:
    """Drop a repeated string that only occurs inside a longer repeated one (keep "ซัพพลายเออร์", drop "ซัพพลาย"/"พลายเออร์")."""
    kept: list[tuple[str, int]] = []
    for s, n in sorted(cands.items(), key=lambda kv: (-len(kv[0]), -kv[1])):
        if any(s in k and n <= kn * 1.25 for k, kn in kept):
            continue
        kept.append((s, n))
    return kept


def extract_terms(texts: list[str], names: list[str] | None = None, agenda_titles: list[str] | None = None, limit: int = 60) -> list[dict]:
    """-> [{"term", "score", "kind"}] best first. kinds: person, agenda, name, acronym, word."""
    out: dict[str, dict] = {}

    def add(term: str, score: float, kind: str):
        term = term.strip(" .,-")
        if len(term) < 2 or term.lower() in EN_STOP:
            return
        k = term.lower()
        if k not in out or out[k]["score"] < score:
            out[k] = {"term": term, "score": round(score, 1), "kind": kind}

    for n in names or []:
        add(n, 100, "person")
    for t in agenda_titles or []:
        if 2 < len(t) <= 40:
            add(t, 60, "agenda")
        for w in LATIN.findall(t):
            if len(w) >= 3:
                add(w, 55, "agenda")
        for w in THAI.findall(t):
            if len(w) >= 4 and w[0] not in NOT_START:
                add(w, 45, "agenda")
    text = "\n".join(texts)
    for m in HONORIFIC.finditer(text):
        add(m.group(1), 80, "person")
    lat = Counter(LATIN.findall(text))
    for w, n in lat.items():
        if len(w) < 3 or w.lower() in EN_STOP:
            continue
        acronym = w.isupper() and len(w) >= 2
        mixed = any(ch.isupper() for ch in w[1:]) or any(ch.isdigit() for ch in w)
        capital = w[0].isupper() and len(w) >= 4  # a capitalised Latin word is likely a name/product, even if it is mentioned once
        if acronym or mixed or capital:
            add(w, 30 + 8 * min(n, 6) + (10 if acronym else 0), "acronym" if acronym or mixed else "name")
    th = _thai_candidates(text)
    cands = {s: n for s, n in th.items() if (n >= 3 or (n >= 2 and len(s) >= 7)) and s not in TH_STOP}
    for s, n in _maximal(cands)[:400]:
        add(s, min(70, 6 * min(n, 8) + len(s) * 1.5), "word")
    return sorted(out.values(), key=lambda t: -t["score"])[:limit]


def asr_prompt(terms: list[str], limit: int = 200) -> str:
    """Terms for Whisper's initial prompt: names first, short ones first, within the ~200 characters it can use."""
    out, n = [], 0
    for t in terms:
        if len(t) > 30 or t in out:
            continue
        if n + len(t) + 2 > limit:
            continue
        out.append(t)
        n += len(t) + 2
    return ", ".join(out)


# ------------------------------------------------------------------ the terms of one meeting
def _terms_path(mid: str):
    return store.mdir(mid) / "materials" / "terms.json"


def refresh_terms(mid: str) -> dict:
    """Re-read the meeting's documents + agenda + participants; the user's own additions/removals are kept."""
    st = store.load_state(mid) or {}
    cur = store.read_json(_terms_path(mid), {}) or {}
    auto = extract_terms(kb.materials(mid).texts(), st.get("participants") or [], [a["title"] for a in st.get("agenda") or []])
    data = {"auto": auto, "extra": cur.get("extra", []), "removed": cur.get("removed", [])}
    (store.mdir(mid) / "materials").mkdir(parents=True, exist_ok=True)
    store.write_json(_terms_path(mid), data)
    return data


def effective_terms(mid: str, refresh: bool = False) -> list[str]:
    """Final list (best first): the user's extras, then the automatic ones minus the removed."""
    d = store.read_json(_terms_path(mid)) if not refresh else None
    if d is None:
        d = refresh_terms(mid)
    removed = {r.lower() for r in d.get("removed", [])}
    out = list(d.get("extra", [])) + [t["term"] for t in d.get("auto", []) if t["term"].lower() not in removed]
    seen, res = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            res.append(t)
    return res


def trusted_terms(mid: str, refresh: bool = False) -> list[str]:
    """The terms safe to act on (speech prompt, transcript correction): names, acronyms, agenda words, the user's own terms and the
    well-supported repeated words. Junk fragments of the automatic Thai word finder (short, weakly supported) are left out."""
    d = store.read_json(_terms_path(mid)) if not refresh else None
    if d is None:
        d = refresh_terms(mid)
    removed = {r.lower() for r in d.get("removed", [])}
    out = list(d.get("extra", []))
    for t in d.get("auto", []):
        if t["term"].lower() in removed:
            continue
        if t["kind"] in ("person", "acronym", "name") or (t["kind"] == "agenda" and len(t["term"]) >= 4) or (t["kind"] == "word" and len(t["term"]) >= 5 and t["score"] >= 24):
            out.append(t["term"])
    seen, res = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            res.append(t)
    return res


def correction_terms(mid: str) -> list[str]:
    """Terms the transcript may be CORRECTED towards. Stricter than trusted_terms(): the automatic Thai word finder can produce truncated
    fragments, which are fine as spelling hints but must never rewrite the transcript."""
    d = store.read_json(_terms_path(mid)) or refresh_terms(mid)
    removed = {r.lower() for r in d.get("removed", [])}
    out = list(d.get("extra", []))
    for t in d.get("auto", []):
        if t["term"].lower() not in removed and (t["kind"] in ("person", "acronym", "name") or (t["kind"] == "agenda" and len(t["term"]) >= 4)):
            out.append(t["term"])
    seen, res = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            res.append(t)
    return res


def save_user_terms(mid: str, extra: list[str], removed: list[str]) -> dict:
    d = store.read_json(_terms_path(mid)) or refresh_terms(mid)
    clean = lambda xs: [" ".join(str(x).split())[:40] for x in xs if str(x).strip()][:100]  # noqa: E731
    d["extra"], d["removed"] = clean(extra), clean(removed)
    store.write_json(_terms_path(mid), d)
    return d


# ------------------------------------------------------------------ conservative fuzzy correction of near-miss terms
def correct_text(text: str, terms: list[str], protect: set[str] | None = None) -> tuple[str, list[dict]]:
    """Replace a stretch of `text` that is ONE edit (two for long terms) away from a known term, only for terms >= 4 characters.
    Never touches a stretch that is itself a known term; Latin terms need word boundaries. Returns (new text, [{from, to}])."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    known = {t.lower() for t in terms} | (protect or set())
    changes: list[dict] = []
    for term in sorted({t for t in terms if len(t) >= 4}, key=len, reverse=True):
        latin = bool(re.fullmatch(r"[A-Za-z0-9\-. ]+", term))
        allow = 1 if len(term) <= 9 else 2
        pos = 0
        for _ in range(8):
            al = fuzz.partial_ratio_alignment(term.lower(), text[pos:].lower(), score_cutoff=70)
            if not al:
                break
            s, e = pos + al.dest_start, pos + al.dest_end
            win = text[s:e]
            # the aligned window can be a little shorter/longer than the term: try the neighbouring lengths and keep the closest
            best = (99, 99, s, e)
            for ds in (-2, -1, 0, 1, 2):
                for de in (-2, -1, 0, 1, 2):
                    s2, e2 = s + ds, e + de
                    if not (0 <= s2 < e2 <= len(text)):
                        continue
                    if latin and not (text[s2].isalnum() and text[e2 - 1].isalnum()):
                        continue
                    if not latin and ((e2 < len(text) and text[e2] in NOT_START and text[e2] not in "ๆฯ") or text[s2] in NOT_START):
                        continue  # the window would cut a Thai syllable in half (leaving a stray vowel / tone mark behind)
                    cand = (Levenshtein.distance(term.lower(), text[s2:e2].lower()), abs(e2 - s2 - len(term)), s2, e2)
                    if cand[:2] < best[:2]:  # fewest edits first, then the window whose length is closest to the term's
                        best = cand
            d, _, s, e = best
            win = text[s:e]
            pos = e
            if d == 0 or d > allow or win.lower() in known:
                continue
            if latin and (not win[:1].isalnum() or not win[-1:].isalnum() or (s > 0 and text[s - 1].isascii() and text[s - 1].isalnum()) or (e < len(text) and text[e].isascii() and text[e].isalnum())):
                continue  # a Latin term must be a whole ASCII word (Thai letters next to it are fine)
            if not latin and abs(len(win) - len(term)) > allow:
                continue
            changes.append({"from": win, "to": term})
            text = text[:s] + term + text[e:]
            pos = s + len(term)
    return text, changes


def correct_segments(segs: list[dict], terms: list[str]) -> list[dict]:
    """In place: fixes segment text, keeps the original in seg["orig"]; returns the list of changes (with the segment's start time)."""
    log = []
    for s in segs:
        new, ch = correct_text(s["text"], terms)
        if ch:
            s["orig"] = s["text"]
            s["text"] = new
            log += [{**c, "t": s["t0"]} for c in ch]
    return log


# ------------------------------------------------------------------ agenda coverage
def agenda_coverage(agenda: list[dict], segs: list[dict]) -> list[dict]:
    """Which agenda items were actually talked about. Every sentence is assigned to the ONE agenda item it fits best (so chit-chat and
    sentences about other items cannot make an undiscussed item look covered); an item counts when it gets a clearly similar sentence."""
    keep = [(n, s["t0"], s["text"]) for n, s in enumerate(segs) if len(s["text"].strip()) >= 15]
    empty = [{"index": i, "title": a["title"], "covered": False, "sim": 0.0, "first_t": None, "mentions": 0, "seg_idx": []} for i, a in enumerate(agenda)]
    if not agenda or not keep:
        return empty
    seg_no, times, texts = [k[0] for k in keep], [k[1] for k in keep], [k[2] for k in keep]
    keys = [kb.trigrams(a["title"]) for a in agenda]
    segtri = [kb.trigrams(t) for t in texts]
    lex = np.array([[len(k & t) / (len(k) ** 0.5 * max(1, len(t)) ** 0.5) for t in segtri] for k in keys], np.float32)
    dense = None
    if kb.EMB.available():
        try:
            dense = kb.EMB.embed([f"{a['title']} {a.get('note', '')}".strip() for a in agenda], "query") @ kb.EMB.embed([t[:300] for t in texts], "passage").T
        except Exception:  # noqa: BLE001
            dense = None
    score = (0.7 * dense + 0.3 * lex) if dense is not None else lex
    owner = score.argmax(axis=0)  # the agenda item each sentence belongs to
    out = []
    for i, a in enumerate(agenda):
        idx = [j for j in range(len(texts)) if owner[j] == i and ((dense is not None and dense[i, j] >= 0.83 and (lex[i, j] >= 0.02 or dense[i, j] >= 0.86)) or (dense is None and lex[i, j] >= 0.25))]
        out.append({"index": i, "title": a["title"], "covered": bool(idx), "sim": round(float((dense if dense is not None else lex)[i].max()), 3), "first_t": float(times[idx[0]]) if idx else None,
                    "mentions": len(idx), "seg_idx": [seg_no[j] for j in idx]})
    return out
