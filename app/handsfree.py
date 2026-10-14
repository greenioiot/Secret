"""EXPERIMENTAL hands-free voice enrolment: what a recognised utterance means at each step of the spoken dialogue.

The browser streams the microphone only after the person pressed Start (it is never an always-on service). Utterances are
recognised on this device (Typhoon ASR) and matched here with fuzzy string matching, so English words arrive as Thai
spellings ("เอ็นโรล มาย วอยซ์"). The sensitivity presets are similarity thresholds of that matching, NOT a measured
recognition accuracy."""
from __future__ import annotations

import re

from rapidfuzz import fuzz

SENSITIVITY = {"strict": 90, "normal": 80, "loose": 70}  # % similarity needed for the wake name / commands
POLITE = ("นะครับ", "นะคะ", "ครับผม", "ครับ", "คับ", "ค่ะ", "คะ", "ค่า", "นะ", "จ้ะ", "จ้า", "ฮะ", "หน่อย", "ด้วย", "please")
TONES = re.compile(r"[็-์\s,.!?'\"-]")  # tone marks, maitaikhu, thanthakhat, spaces, punctuation
HEY = ("เฮ้", "เฮ", "เฮย์", "เห้", "เฮ้ย", "hey", "hi", "ไฮ", "หวัดดี", "สวัสดี")
ENROLL = ("ลงทะเบียนเสียง", "ลงทะเบียน", "บันทึกเสียงฉัน", "บันทึกเสียงผม", "จำเสียงฉัน", "จำเสียงผม", "จำเสียง", "เอ็นโรลมายวอยซ์", "เอนโรลมายวอยซ์",
          "อินโรลมายวอยซ์", "เอ็นโรลวอยซ์", "เอ็นโรล", "enroll my voice", "enrol my voice", "enroll", "register my voice")
CANCEL = ("ยกเลิก", "หยุด", "พอแล้ว", "ไม่เอาแล้ว", "แคนเซิล", "แคนเซิ่ล", "cancel", "stop")
YES = ("ยินยอม", "ยอมรับ", "ตกลง", "ใช่", "ถูกต้อง", "ถูก", "ได้", "โอเค", "เยส", "ไอแอกรี", "yes", "ok", "okay", "agree", "i agree", "correct", "ลองใหม่", "again")
NO = ("ไม่ยินยอม", "ไม่ยอม", "ไม่ตกลง", "ไม่ใช่", "ไม่ถูก", "ผิด", "ไม่ได้", "ไม่", "โน", "no", "wrong", "not")
NAME_PREFIX = re.compile(r"^(?:ผม|ฉัน|ดิฉัน|หนู|เรา|กระผม|ข้าพเจ้า)?\s*ชื่อ(?:ว่า)?\s*|^(?:my name is|i am|i'm|name is|มายเนมอีส|มายเนมอิส)\s*", re.I)


def wake_forms(name: str) -> list[str]:
    """Spellings the Thai recogniser may write for the assistant's name (default "Secrets")."""
    base = {"secrets": ["ซีเคร็ต", "ซีเคร็ท", "ซีเครท", "ซีเครต", "ซีเคร็ทส์", "ซีเคร็ตส์", "ซีเครทส์", "ซีกเร็ต", "ซิเคร็ต", "สีเคร็ด", "ซีเคร็ด", "secrets", "secret"],
            "zecret": ["ซีเคร็ต", "ซีเคร็ท", "ซีเครท", "ซีกรีต", "zecret", "secret"]}
    k = name.strip().lower()
    return base.get(k, []) + ([k] if k not in base else [])


def _norm(t: str) -> str:
    """Lower case, no tone marks / spaces, ร = ล (spoken Thai often merges them: the recogniser writes ซีเคร็ท as ซีเคล็ด)."""
    return TONES.sub("", t.lower()).replace("ร", "ล")


def strip_polite(text: str) -> str:
    t = " ".join(text.split())
    changed = True
    while changed:
        changed = False
        for p in POLITE:
            if t.lower().endswith(p) and len(t) > len(p):
                t, changed = t[: -len(p)].rstrip(" ,."), True
    return t.strip()


def _hit(text: str, forms, threshold: int) -> tuple[bool, float]:
    """Best fuzzy (partial) match of any form inside the text, tone marks and spaces ignored."""
    n = _norm(text)
    best = 0.0
    for f in forms:
        fn = _norm(f)
        if not fn:
            continue
        s = 100.0 if fn in n else (fuzz.partial_ratio(fn, n) if len(fn) >= 4 else 0.0)
        best = max(best, s)
    return best >= threshold, best


def _word(text: str, words) -> bool:
    """Exact word / phrase match (yes, no, cancel): short words must not be found fuzzily."""
    n = _norm(strip_polite(text))
    return any(_norm(w) == n or (len(_norm(w)) >= 4 and _norm(w) in n) for w in words)


def extract_name(text: str) -> str | None:
    t = strip_polite(text)
    t = NAME_PREFIX.sub("", t).strip()
    t = strip_polite(t)
    t = re.sub(r"[^\w฀-๿ .-]", "", t).strip()
    if not t or len(t) > 40 or _word(t, CANCEL):
        return None
    return t


def parse(text: str, stage: str, wake_name: str = "Secrets", sensitivity: str = "normal") -> dict:
    """-> {"intent": ..., "score"?, "name"?}. Intents: cancel, wake_enroll, wake, enroll, yes, no, name, unknown."""
    text = (text or "").strip()
    thr = SENSITIVITY.get(sensitivity, 80)
    if not text:
        return {"intent": "unknown"}
    if stage != "name" and _word(text, CANCEL):
        return {"intent": "cancel"}
    if stage in ("wake", "command"):
        wk, ws = _hit(text, wake_forms(wake_name), thr)
        en, es = _hit(text, ENROLL, thr)
        if wk and en:
            return {"intent": "wake_enroll", "score": round(min(ws, es))}
        if en and stage == "command":
            return {"intent": "enroll", "score": round(es)}
        if wk and ws >= min(100, thr + 10):  # the name alone (no command) must match more closely: everyday words (เครดิต) come near it
            return {"intent": "wake", "score": round(ws)}
        return {"intent": "unknown", "score": round(max(ws, es))}
    if stage in ("consent", "confirm", "retry"):
        if _word(text, NO):  # "ไม่ใช่" contains "ใช่": the negative is checked first
            return {"intent": "no"}
        if _word(text, YES) or (stage == "confirm" and _norm(text) in {_norm(w) for w in ("ครับ", "ค่ะ", "คะ", "คับ")}):
            return {"intent": "yes"}
        return {"intent": "unknown"}
    if stage == "name":
        if _word(text, CANCEL) and len(_norm(text)) <= 10:
            return {"intent": "cancel"}
        n = extract_name(text)
        return {"intent": "name", "name": n} if n else {"intent": "unknown"}
    return {"intent": "unknown"}
