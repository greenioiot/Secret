"""Participation report: deterministic statistics per person and for the whole meeting, plus (optional) LLM narratives
written by summarize.py. Statistics measure speaking participation (who talked, how much, how often, who asked), not
the quality or the attention of the people: the report says so."""
from __future__ import annotations

import math
import re
import time

import store

# Thai + English question markers (heuristic: the ASR gives no punctuation)
QUESTION = re.compile(r"ไหม|มั้ย|หรือเปล่า|หรือยัง|หรือไม่|ใช่ไหม|ใช่มั้ย|หรอ(?:ครับ|คะ|ค่ะ)|อะไร|ทำไม|อย่างไร|ยังไง|เมื่อไหร่|เมื่อไร|"
                      r"ที่ไหน|ใคร|เท่าไหร่|เท่าไร|กี่|แค่ไหน|อันไหน|แบบไหน|\b(?:what|why|how|when|where|who)\b|\?", re.I)
TURN_GAP_S = 2.0       # same speaker within this gap = same turn
BUCKETS = 20           # timeline resolution


def _chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def level(ratio: float) -> str:
    """ratio = share of speaking time x number of people (1.0 = an equal share)."""
    if ratio >= 1.5:
        return "leading"
    if ratio >= 0.7:
        return "balanced"
    if ratio >= 0.3:
        return "moderate"
    return "quiet"


def compute(mid: str) -> dict | None:
    d = store.mdir(mid)
    tr = store.read_json(d / "transcript.json")
    st = store.load_state(mid)
    if not tr or not st:
        return None
    spk = store.speakers(mid)
    names = store.speaker_names(mid, spk)
    segs = [s for s in tr["segments"] if s["text"].strip()]
    dur = float(st.get("duration_s") or (segs[-1]["t1"] if segs else 0)) or 1.0
    identified = bool(spk) and any(s.get("spk") for s in segs)
    people: dict[str, dict] = {}
    prev = None  # (person, t1)
    turns_total, changes = 0, 0
    longest = (0.0, None, 0.0)
    for s in segs:
        who = names.get(s.get("spk"), "ผู้เข้าร่วม") if identified else "ทุกคน"
        p = people.setdefault(who, {"name": who, "clusters": set(), "talk_s": 0.0, "turns": 0, "chars": 0, "questions": 0, "first": s["t0"], "last": s["t1"],
                                    "longest_turn_s": 0.0, "timeline": [0.0] * BUCKETS, "turn_start": None, "cur_turn_s": 0.0})
        if s.get("spk"):
            p["clusters"].add(s["spk"])
        seg_s = max(0.0, s["t1"] - s["t0"])
        new_turn = prev is None or prev[0] != who or s["t0"] - prev[1] > TURN_GAP_S
        if new_turn:
            turns_total += 1
            p["turns"] += 1
            p["cur_turn_s"] = 0.0
            if prev is not None and prev[0] != who:
                changes += 1
        p["cur_turn_s"] += seg_s
        p["longest_turn_s"] = max(p["longest_turn_s"], p["cur_turn_s"])
        if p["cur_turn_s"] > longest[0]:
            longest = (p["cur_turn_s"], who, s["t1"])
        p["talk_s"] += seg_s
        p["chars"] += _chars(s["text"])
        p["questions"] += len(QUESTION.findall(s["text"])) and 1 or 0
        p["last"] = s["t1"]
        # spread the segment over the timeline buckets it covers
        b0, b1 = int(s["t0"] / dur * BUCKETS), int(min(s["t1"], dur - 1e-6) / dur * BUCKETS)
        for b in range(max(0, b0), min(BUCKETS - 1, b1) + 1):
            lo, hi = max(s["t0"], b * dur / BUCKETS), min(s["t1"], (b + 1) * dur / BUCKETS)
            p["timeline"][b] += max(0.0, hi - lo)
        prev = (who, s["t1"])
    listed = st.get("participants") or []
    total_talk = sum(p["talk_s"] for p in people.values()) or 1.0
    n = len(people)
    out_people = []
    for p in sorted(people.values(), key=lambda p: -p["talk_s"]):
        share = p["talk_s"] / total_talk
        out_people.append({
            "name": p["name"], "clusters": sorted(p["clusters"]), "talk_s": round(p["talk_s"], 1), "share": round(share, 4), "turns": p["turns"],
            "avg_turn_s": round(p["talk_s"] / max(1, p["turns"]), 1), "longest_turn_s": round(p["longest_turn_s"], 1), "chars": p["chars"],
            "chars_per_min": round(p["chars"] / max(0.1, p["talk_s"] / 60)), "questions": p["questions"], "first": round(p["first"], 1),
            "last": round(p["last"], 1), "level": level(share * n) if identified else None,
            "timeline": [round(x, 1) for x in p["timeline"]]})
    spoken = {p["name"].casefold() for p in out_people}
    silent = [n for n in listed if n.casefold() not in spoken]
    for n in silent:   # listed as present but nothing was attributed to them
        out_people.append({"name": n, "clusters": [], "talk_s": 0.0, "share": 0.0, "turns": 0, "avg_turn_s": 0.0, "longest_turn_s": 0.0, "chars": 0,
                           "chars_per_min": 0, "questions": 0, "first": None, "last": None, "level": "silent", "silent": True, "timeline": [0.0] * BUCKETS})
    shares = [p["share"] for p in out_people if p["share"] > 0]
    balance = (-sum(x * math.log(x) for x in shares) / math.log(len(shares))) if len(shares) > 1 else None
    overview = {"duration_s": round(dur, 1), "talk_s": round(total_talk, 1), "silence_pct": round(max(0.0, 1 - total_talk / dur) * 100),
                "participants": n if identified else None, "attendees": len(listed) or None, "silent": silent, "turns": turns_total, "speaker_changes": changes,
                "changes_per_min": round(changes / (dur / 60), 1), "questions": sum(p["questions"] for p in out_people),
                "balance": round(balance, 2) if balance is not None else None,
                "longest_monologue": {"name": longest[1], "seconds": round(longest[0], 1)} if longest[1] else None, "identified": identified}
    rep = {"generated": time.strftime("%F %T"), "overview": overview, "people": out_people, "names_key": names_key(mid)}
    rep["overview_text"] = overview_text(rep)
    return rep


def overview_text(rep: dict) -> str:
    """Participation overview written from the numbers (an LLM paraphrase contradicted the statistics in testing)."""
    o, ps = rep["overview"], rep["people"]
    if not o["identified"] or len(ps) < 2:
        return ""
    ps = [p for p in ps if not p.get("silent")]
    if len(ps) < 2:
        return ""
    top, low = ps[0], ps[-1]
    bal = o["balance"] or 0
    shape = "การพูดกระจายค่อนข้างสมดุล" if bal >= 0.9 else "การพูดค่อนข้างสมดุล" if bal >= 0.75 else "การพูดกระจุกตัวอยู่กับบางคน"
    out = [f"{top['name']} พูดมากที่สุด ({top['share'] * 100:.0f}% ของเวลาพูด) ส่วน {low['name']} พูดน้อยที่สุด ({low['share'] * 100:.0f}%) — {shape}."]
    asker = max(ps, key=lambda p: p["questions"])
    if o["questions"]:
        out.append(f"มีคำถามรวม {o['questions']} ข้อ" + (f" ส่วนใหญ่จาก {asker['name']}" if asker["questions"] * 2 > o["questions"] and len(ps) > 1 else "") + ".")
    else:
        out.append("ไม่พบคำถามในการประชุมนี้.")
    out.append(f"มีการสลับผู้พูด {o['speaker_changes']} ครั้ง ({o['changes_per_min']} ครั้ง/นาที) และช่วงเงียบ {o['silence_pct']}% ของเวลาประชุม.")
    if o["longest_monologue"]:
        m = o["longest_monologue"]
        out.append(f"{m['name']} พูดต่อเนื่องนานที่สุด {mmss(m['seconds'])} นาที.")
    if o.get("silent"):
        out.append("ผู้เข้าร่วมที่ไม่พบว่าได้พูด: " + ", ".join(o["silent"]) + " (ถ้ามีผู้เข้าร่วมที่ยังไม่ได้ตั้งชื่อ อาจเป็นคนเหล่านี้).")
    return " ".join(out)


def names_key(mid: str) -> str:
    """Changes whenever someone is renamed / merged: tells the UI that the summary and narratives were written with other names."""
    spk = store.speakers(mid)
    return "|".join(f"{k}={v.get('name') or ''}" for k, v in sorted(spk.items()))


def refresh(mid: str) -> dict | None:
    """Recompute the statistics (instant) and keep the narratives written earlier."""
    stats = compute(mid)
    if stats is None:
        return None
    old = store.read_json(store.mdir(mid) / "report.json", {}) or {}
    stats["narrative"] = old.get("narrative")
    stats["narrative_names_key"] = old.get("narrative_names_key")
    if old.get("narrative_ai"):
        stats["narrative_ai"] = old["narrative_ai"]  # hand-edited notes: what the model wrote
    store.write_json(store.mdir(mid) / "report.json", stats)
    store.atomic_write(store.mdir(mid) / "report.md", to_md(mid, stats))
    return stats


def mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


LEVEL_TH = {"leading": "นำการสนทนา", "balanced": "สมดุล", "moderate": "ปานกลาง", "quiet": "พูดน้อย", "silent": "ไม่ได้พูด"}


def stats_text(rep: dict) -> str:
    """Compact factual digest fed to the LLM for the overview narrative."""
    o = rep["overview"]
    lines = [f"ระยะเวลาประชุม {mmss(o['duration_s'])} นาที, เวลาพูดรวม {mmss(o['talk_s'])}, ช่วงเงียบ {o['silence_pct']}%, "
             f"ผู้เข้าร่วม {o['participants'] or 'ไม่ทราบ'} คน, สลับผู้พูด {o['speaker_changes']} ครั้ง, คำถามรวม {o['questions']}"]
    for p in rep["people"]:
        lines.append(f"- {p['name']}: พูด {mmss(p['talk_s'])} นาที ({p['share'] * 100:.0f}%), {p['turns']} ครั้ง, ถาม {p['questions']} คำถาม, "
                     f"ช่วงยาวสุด {mmss(p['longest_turn_s'])}" + (f", ระดับ: {LEVEL_TH.get(p['level'], '')}" if p.get("level") else ""))
    return "\n".join(lines)


def to_md(mid: str, rep: dict) -> str:
    st = store.load_state(mid) or {}
    o = rep["overview"]
    nar = rep.get("narrative") or {}
    md = [f"# รายงานการมีส่วนร่วม: {st.get('title', '')}", "",
          "> วัดการมีส่วนร่วมด้านการพูด (เวลา จำนวนครั้ง คำถาม) ไม่ใช่คุณภาพหรือความตั้งใจของผู้เข้าร่วม", "", "## ภาพรวม", "",
          f"- ระยะเวลา: {mmss(o['duration_s'])} นาที (เวลาพูดรวม {mmss(o['talk_s'])}, เงียบ {o['silence_pct']}%)"]
    if st.get("participants"):
        md.append("- รายชื่อผู้เข้าร่วม: " + ", ".join(st["participants"]))
    if st.get("remark"):
        md.append("- หมายเหตุ: " + " ".join(st["remark"].split()))
    if o["identified"]:
        md.append(f"- ผู้เข้าร่วม: {o['participants']} คน; สลับผู้พูด {o['speaker_changes']} ครั้ง ({o['changes_per_min']} ครั้ง/นาที); คำถามรวม {o['questions']}")
        if o["balance"] is not None:
            md.append(f"- ความสมดุลของการพูด: {o['balance']:.2f} (1.00 = ทุกคนพูดเท่ากัน)")
        if o["longest_monologue"]:
            md.append(f"- พูดต่อเนื่องนานที่สุด: {o['longest_monologue']['name']} ({mmss(o['longest_monologue']['seconds'])} นาที)")
    if rep.get("overview_text"):
        md += ["", rep["overview_text"]]
    md += ["", "## รายบุคคล", "", "| ชื่อ | เวลาพูด | สัดส่วน | ครั้ง | เฉลี่ย/ครั้ง | คำถาม | ระดับ |", "|---|---|---|---|---|---|---|"]
    for p in rep["people"]:
        md.append(f"| {p['name']} | {mmss(p['talk_s'])} | {p['share'] * 100:.0f}% | {p['turns']} | {mmss(p['avg_turn_s'])} | {p['questions']} | {LEVEL_TH.get(p.get('level'), '-')} |")
    for p in rep["people"]:
        note = (nar.get("people") or {}).get(p["name"])
        if note:
            md += ["", f"### {p['name']}", "", "_สรุปโดย AI อาจมีข้อผิดพลาด ตรวจสอบกับบันทึกถอดเสียง_", "", note]
    return "\n".join(md) + "\n"
