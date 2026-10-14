"""Writing style of the transcript: numbers as Arabic digits (not Thai number words), English words in English letters
(not Thai transliteration). Thai has no spaces, so every rule is conservative: a number word is only turned into a digit
when the context says it is a number (a unit follows, several number words in a row, a number-context word before it)."""
from __future__ import annotations

import re
from pathlib import Path

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
DIGIT = {"ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "ยี่": 2, "สาม": 3, "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9}
MULT = {"สิบ": 10, "ร้อย": 100, "พัน": 1000, "หมื่น": 10000, "แสน": 100000, "ล้าน": 1000000}
TOK = "|".join(sorted([*DIGIT, *MULT], key=len, reverse=True)).replace("ร้อย", "ร้อย(?!ละ)")
_NUM = f"(?:{TOK})+(?:จุด(?:ศูนย์|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)+)?"
NUM_RE = re.compile(_NUM)
TOK_RE = re.compile(f"({TOK})")
MONTHS = ("มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม")
UNITS = MONTHS + ("บาท", "ดอลลาร์", "คน", "ท่าน", "วัน", "เดือน", "ปี", "ชั่วโมง", "นาที", "วินาที", "สัปดาห์", "อาทิตย์", "ครั้ง", "ชิ้น", "อัน", "ราย", "แห่ง", "ข้อ",
                  "หน้า", "ห้อง", "ชั้น", "เครื่อง", "ตัว", "ชุด", "รอบ", "เปอร์เซ็นต์", "เปอร์เซนต์", "%", "กิโล", "เมตร", "กรัม", "โมง", "ทุ่ม", "เท่า", "คะแนน", "เฟส",
                  "สาขา", "ประเทศ", "คัน", "หลัง", "ฉบับ", "รุ่น", "ทีม", "กลุ่ม", "ไตรมาส", "เซนต์", "หมื่น", "พัน", "ร้อย", "แสน", "ล้าน", "บรรทัด", "ตัวอักษร", "เรื่อง",
                  "ประการ", "ด้าน", "ขั้นตอน", "ระดับ", "ปีงบประมาณ", "ปีก่อน", "ภาค", "เทอม", "สมาชิก", "พนักงาน", "ลูกค้า", "ผู้", "คอร์ส", "เซสชัน", "จุด", "แผน", "ทาง")
CONTEXT_BEFORE = ("วันที่", "ครั้งที่", "ลำดับที่", "ข้อที่", "ข้อ", "ชั้น", "ปี", "พ.ศ.", "ค.ศ.", "เวอร์ชัน", "ร้อยละ", "ราคา", "งบ", "จำนวน", "ประมาณ", "เกือบ", "กว่า", "ไม่เกิน",
                  "อย่างน้อย", "เบอร์", "โทร", "ห้อง", "ระดับ", "ที่", "เวลา", "เพิ่ม", "ลด", "version", "ไตรมาส", "เฟส", "รอบ", "หน้า", "ทีม", "กลุ่ม", "แผน", "ตั้งแต่", "ถึง", "เป็น", "ได้")
NOT_AFTER = ("ง", "อี้", "ล้ม", "ธ", "ริ้ว", "เอก", "แถว", "ล้อ", "แยก", "เหลี่ยม", "มารถ", "ัญ")  # ห้าง เก้าอี้ หกล้ม พันธมิตร สิบเอก สองแถว สามล้อ สี่แยก ...
_MARKS = "ะาำัิีึืฺุู็่้๊๋์ํ๎"


def _parse(tokens: list[str]) -> int | str | None:
    if all(t in DIGIT for t in tokens):
        if len(tokens) == 1:
            return DIGIT[tokens[0]]
        return "".join(str(DIGIT[t]) for t in tokens) if len(tokens) >= 3 else None  # "สองสาม" = a few, not 23; 3+ digits are read out (phone, year)
    total = sec = 0
    pend: int | None = None
    last = 10**9
    for i, t in enumerate(tokens):
        if t in DIGIT:
            if pend is not None or (t == "ยี่" and (i + 1 >= len(tokens) or tokens[i + 1] != "สิบ")):
                return None
            pend = DIGIT[t]
        elif t == "ล้าน":
            total += ((sec + (pend or 0)) or 1) * 1_000_000
            sec, pend, last = 0, None, 10**9
        else:
            v = MULT[t]
            if v >= last:
                return None
            sec += (1 if pend is None else pend) * v
            pend, last = None, v
    return total + sec + (pend or 0)


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    return str(v) if v < 1000 or 1900 <= v <= 2100 or 2400 <= v <= 2600 else f"{v:,}"  # years are written without a comma


def numbers_to_digits(text: str) -> str:
    text = text.translate(THAI_DIGITS)
    out, pos, dec_end = [], 0, -1
    for m in NUM_RE.finditer(text):
        s, e = m.span()
        if s < pos:
            continue
        run = m.group(0)
        whole, _, dec = run.partition("จุด")
        toks = TOK_RE.findall(whole)
        after, before = text[e:], text[:s]
        if after[:1] and (after[0] in _MARKS) or any(after.startswith(x) for x in NOT_AFTER):
            continue  # the number word is the beginning of a longer word
        val = _parse(toks)
        if val is None:
            continue
        has_digit = any(t in DIGIT for t in toks)
        unit = any(after.startswith(u) for u in UNITS)
        ctx = any(before.endswith(c) for c in CONTEXT_BEFORE)
        multi = len(toks) >= 2 and not (isinstance(val, str) and len(toks) < 3)
        if not (unit or ctx or (multi and has_digit)):
            continue
        if not has_digit and (not unit or s == dec_end):  # "2.5" + "ล้าน": the multiplier stays a word
            continue
        if not has_digit and before.rstrip()[-1:].isdigit():  # "5,800 ล้านบาท": the multiplier belongs to the digits before it (it became "5,800 1,000,000 บาท")
            continue
        txt = _fmt(val)
        if dec:
            ds = "".join(str(DIGIT[t]) for t in TOK_RE.findall(dec))
            txt = f"{val}.{ds}"
            dec_end = e
        pre = " " if before and re.match(r"[ก-๙]", before[-1]) else ""
        post = " " if after and re.match(r"[ก-๙]", after[0]) else ""
        out.append(text[pos:s] + pre + txt + post)
        pos = e
    out.append(text[pos:])
    return re.sub(r" {2,}", " ", "".join(out))


_LOAN: list[tuple[str, str]] | None = None
_LOAN_RE: re.Pattern | None = None


def _load_loanwords():
    global _LOAN, _LOAN_RE
    pairs = []
    for line in (Path(__file__).with_name("loanwords.tsv")).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "\t" in line:
            a, b = line.split("\t", 1)
            for v in a.split("/"):
                pairs.append((v.strip(), b.strip()))
    _LOAN = dict(sorted(pairs, key=lambda p: -len(p[0])))
    _LOAN_RE = re.compile("|".join(re.escape(k) for k in _LOAN))


def english_words(text: str) -> str:
    """Thai-spelled English words back to English (brands, acronyms, office / tech / business words)."""
    if _LOAN is None:
        _load_loanwords()

    def sub(m):
        nxt = text[m.end(): m.end() + 1]
        if nxt and nxt in _MARKS:  # the match stops inside a syllable
            return m.group(0)
        eng = _LOAN[m.group(0)]
        pre = " " if m.start() > 0 and re.match(r"[ก-๙]", text[m.start() - 1]) else ""
        post = " " if nxt and re.match(r"[ก-๙]", nxt) else ""
        return pre + eng + post

    return re.sub(r" {2,}", " ", _LOAN_RE.sub(sub, text)).strip()


def polish(text: str, numbers: bool = True, english: bool = True) -> str:
    if english:
        text = english_words(text)
    if numbers:
        text = numbers_to_digits(text)
    return text


def apply_rules(text: str, rules) -> str:
    """Replace Thai spellings by terms (domain vocabulary packs): longest first, never inside a syllable, with a space between Thai and Latin text."""
    if not rules:
        return text
    pairs = sorted(rules, key=lambda r: -len(r[0]))
    rx = re.compile("|".join(re.escape(a) for a, _ in pairs))
    table = dict(pairs)

    def sub(m):
        nxt = text[m.end(): m.end() + 1]
        if (nxt and nxt in _MARKS) or table[m.group(0)] == m.group(0):  # inside a syllable, or a guarded longer word
            return m.group(0)
        pre = " " if m.start() > 0 and re.match(r"[ก-๙]", text[m.start() - 1]) else ""
        post = " " if nxt and re.match(r"[ก-๙]", nxt) else ""
        return pre + table[m.group(0)] + post

    return re.sub(r" {2,}", " ", rx.sub(sub, text)).strip()
