"""Searchable index of all meetings for the Meetings table: sorting, paging, keyword search (title, remark, people, summary, transcript),
date/time and attribute filters, CSV export, ids for "select all matching".

SQLite (data/index.db) with an FTS5 *trigram* index: substring search that works for Thai (no spaces) and English, case-insensitive.
Words shorter than 3 characters cannot use trigrams and fall back to a LIKE on title / remark / people. The index is rebuilt per meeting
only when a meeting's files changed (mtime stamp), so listing is fast and nothing is stored twice except the searchable text.
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import sqlite3
import threading
import time

import config
import store

DB = config.DATA / "index.db"
LOCK = threading.RLock()
_last_sync = 0.0
SORTS = {"created": "m.sort_at", "title": "m.title COLLATE NOCASE", "duration": "m.duration", "status": "m.status", "people": "m.npeople", "size": "m.size", "speakers": "m.nspk"}
FLAGS = {"summary": "f_summary", "report": "f_report", "speakers": "f_speakers", "bot": "f_bot", "materials": "f_materials", "email": "f_email", "agenda": "f_agenda"}
ACTIVE = ("recording", "transcribing", "summarizing", "analysing")  # never deleted while in use


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript("""
        CREATE TABLE IF NOT EXISTS meetings(id TEXT PRIMARY KEY, title TEXT, remark TEXT, participants TEXT, npeople INT, status TEXT, created TEXT, planned_at TEXT, sort_at TEXT,
            duration REAL, nspk INT, size INT, f_summary INT, f_report INT, f_speakers INT, f_bot INT, f_materials INT, f_email INT, f_agenda INT, stamp TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(id UNINDEXED, title, remark, participants, summary, transcript, tokenize='trigram');
        CREATE INDEX IF NOT EXISTS ix_sort ON meetings(sort_at);
    """)
    if "owner" not in {r["name"] for r in c.execute("PRAGMA table_info(meetings)")}:  # added with accounts (round 31): re-index every row once
        c.execute("ALTER TABLE meetings ADD COLUMN owner TEXT DEFAULT ''")
        c.execute("UPDATE meetings SET stamp=''")
        c.commit()
    return c


def _stamp(d) -> str:
    parts = []
    for n in ("state.json", "summary.md", "transcript.json", "report.json", "speakers.json", "bot.jsonl", "audio.wav", "audio.pcm"):
        try:
            st = os.stat(d / n)
            parts.append(f"{n}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            pass
    for sub in ("materials", "materials/docs"):
        try:
            parts.append(f"{sub}:{os.stat(d / sub).st_mtime_ns}")
        except OSError:
            pass
    return "|".join(parts)


def _dir_size(d) -> int:
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _read(p) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def sync(force: bool = False):
    """Bring the index in line with the meeting folders (changed ones only)."""
    global _last_sync
    with LOCK:
        if not force and time.time() - _last_sync < 3:
            return
        c = conn()
        have = {r["id"]: r["stamp"] for r in c.execute("SELECT id, stamp FROM meetings")}
        seen = set()
        for d in config.MEETINGS.glob("*-*"):
            st = store.read_json(d / "state.json")
            if not st:
                continue
            mid = st["id"]
            seen.add(mid)
            stamp = _stamp(d)
            if have.get(mid) == stamp:
                continue
            tr = store.read_json(d / "transcript.json") or {}
            names = store.speaker_names(mid)
            transcript = "\n".join((f"{names.get(s.get('spk'), '')}: " if s.get("spk") else "") + s["text"] for s in tr.get("segments", []))
            summary = _read(d / "summary.md")
            people = st.get("participants") or []
            sort_at = (st.get("planned_at") if st.get("status") == "planned" and st.get("planned_at") else st.get("created", "")) or ""
            mats = list((d / "materials" / "docs").glob("d*.json")) if (d / "materials" / "docs").exists() else []
            row = (mid, st.get("title", ""), st.get("remark", ""), " | ".join(people), len(people), st.get("status", ""), st.get("created", ""), st.get("planned_at", ""), sort_at,
                   float(st.get("duration_s") or 0), len((store.speakers(mid) or {})), _dir_size(d), int(bool(summary.strip())), int((d / "report.json").exists()),
                   int(bool(store.speakers(mid))), int((d / "bot.jsonl").exists()), int(bool(mats)), int(bool(st.get("email_log"))), int(bool(st.get("agenda") or (st.get("bot") or {}).get("agenda"))), stamp, st.get("owner") or "")
            c.execute("INSERT OR REPLACE INTO meetings(id,title,remark,participants,npeople,status,created,planned_at,sort_at,duration,nspk,size,f_summary,f_report,f_speakers,f_bot,"
                      "f_materials,f_email,f_agenda,stamp,owner) VALUES (" + ",".join("?" * 21) + ")", row)
            c.execute("DELETE FROM fts WHERE id=?", (mid,))
            c.execute("INSERT INTO fts(id,title,remark,participants,summary,transcript) VALUES (?,?,?,?,?,?)", (mid, st.get("title", ""), st.get("remark", ""), " | ".join(people), summary, transcript))
        for gone in set(have) - seen:
            c.execute("DELETE FROM meetings WHERE id=?", (gone,))
            c.execute("DELETE FROM fts WHERE id=?", (gone,))
        c.commit()
        c.close()
        _last_sync = time.time()


def _tokens(q: str) -> list[str]:
    """Words of the query; "quoted phrases" stay together."""
    return [a or b for a, b in re.findall(r'"([^"]+)"|(\S+)', q or "")][:8]


def _where(p: dict) -> tuple[str, list]:
    cond, args = [], []
    fields = p.get("fields") or "all"
    cols = {"all": "", "title": "{title}: ", "people": "{participants}: ", "content": "{summary transcript}: ", "notes": "{remark}: "}.get(fields, "")
    for tok in _tokens(p.get("q", "")):
        if len(tok) >= 3:
            cond.append(f"m.id IN (SELECT id FROM fts WHERE fts MATCH ?)")
            args.append(cols + '"' + tok.replace('"', '""') + '"')
        else:
            like = f"%{tok.replace('%', '').replace('_', '')}%"
            cond.append("(m.title LIKE ? OR m.remark LIKE ? OR m.participants LIKE ?)")
            args += [like, like, like]
    sts = [s for s in (p.get("status") or "").split(",") if s]
    if sts:
        cond.append("m.status IN (" + ",".join("?" * len(sts)) + ")")
        args += sts
    if p.get("date_from"):
        cond.append("m.sort_at >= ?")
        args.append(p["date_from"])
    if p.get("date_to"):
        t = p["date_to"]
        cond.append("m.sort_at <= ?")
        args.append(t + "T23:59:59" if len(t) == 10 else t + ":59")
    if p.get("person"):
        cond.append("m.participants LIKE ?")
        args.append(f"%{p['person']}%")
    if p.get("min_min") not in (None, ""):
        cond.append("m.duration >= ?")
        args.append(float(p["min_min"]) * 60)
    if p.get("max_min") not in (None, ""):
        cond.append("m.duration <= ?")
        args.append(float(p["max_min"]) * 60)
    for fl in (p.get("has") or "").split(","):
        if fl in FLAGS:
            cond.append(f"m.{FLAGS[fl]} = 1")
    return (" WHERE " + " AND ".join(cond)) if cond else "", args


def _row(r: sqlite3.Row, q: str, c: sqlite3.Connection) -> dict:
    out = {"id": r["id"], "title": r["title"], "remark": r["remark"], "participants": [x for x in r["participants"].split(" | ") if x], "status": r["status"], "created": r["created"],
           "planned_at": r["planned_at"], "duration_s": r["duration"], "speakers": r["nspk"], "size_mb": round(r["size"] / 1e6, 1), "owner": r["owner"] or None,
           "has": {k: bool(r[v]) for k, v in FLAGS.items()}}
    toks = [t for t in _tokens(q) if t]
    if toks:
        t = c.execute("SELECT summary, transcript FROM fts WHERE id=?", (r["id"],)).fetchone()
        hay = {"title": r["title"], "remark": r["remark"], "people": r["participants"], "summary": t["summary"] if t else "", "transcript": t["transcript"] if t else ""}
        matched, snippet = [], ""
        for col, text in hay.items():
            low = text.lower()
            if any(tok.lower() in low for tok in toks):
                matched.append(col)
                if not snippet and col in ("summary", "transcript", "remark"):
                    i = min((low.find(tok.lower()) for tok in toks if tok.lower() in low))
                    snippet = ("…" if i > 40 else "") + text[max(0, i - 40): i + 90].replace("\n", " ") + "…"
        out["matched"], out["snippet"], out["terms"] = matched, snippet, toks
    return out


def query(p: dict) -> dict:
    sync()
    size = max(5, min(200, int(p.get("size") or 25)))
    page = max(1, int(p.get("page") or 1))
    sort = SORTS.get(p.get("sort") or "created", SORTS["created"])
    direction = "ASC" if (p.get("dir") or "desc").lower() == "asc" else "DESC"
    where, args = _where(p)
    with LOCK:
        c = conn()
        total = c.execute("SELECT COUNT(*) FROM meetings m" + where, args).fetchone()[0]
        pages = max(1, -(-total // size))
        page = min(page, pages)
        rows = c.execute(f"SELECT m.* FROM meetings m{where} ORDER BY {sort} {direction}, m.id DESC LIMIT ? OFFSET ?", args + [size, (page - 1) * size]).fetchall()
        out = [_row(r, p.get("q", ""), c) for r in rows]
        w2, a2 = _where({**p, "status": ""})
        facets = {r["status"]: r["n"] for r in c.execute(f"SELECT m.status, COUNT(*) n FROM meetings m{w2} GROUP BY m.status", a2)}
        people: dict[str, int] = {}
        for r in c.execute("SELECT participants FROM meetings WHERE participants != ''"):
            for n in r["participants"].split(" | "):
                people[n] = people.get(n, 0) + 1
        c.close()
    return {"total": total, "page": page, "pages": pages, "size": size, "rows": out, "facets": facets, "people": sorted(people, key=lambda n: (-people[n], n.casefold()))[:50]}


def ids(p: dict, cap: int = 5000) -> list[str]:
    sync()
    where, args = _where(p)
    with LOCK:
        c = conn()
        out = [r["id"] for r in c.execute(f"SELECT m.id FROM meetings m{where} ORDER BY m.sort_at DESC LIMIT ?", args + [cap])]
        c.close()
    return out


def export_csv(p: dict) -> str:
    sync()
    where, args = _where(p)
    sort = SORTS.get(p.get("sort") or "created", SORTS["created"])
    direction = "ASC" if (p.get("dir") or "desc").lower() == "asc" else "DESC"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "title", "date/time", "planned for", "status", "duration (min)", "participants", "speakers detected", "remark", "has summary", "has report", "has materials", "size (MB)"])
    with LOCK:
        c = conn()
        for r in c.execute(f"SELECT m.* FROM meetings m{where} ORDER BY {sort} {direction}, m.id DESC LIMIT 20000", args):
            w.writerow([r["id"], r["title"], r["created"].replace("T", " "), (r["planned_at"] or "").replace("T", " "), r["status"], round(r["duration"] / 60, 1), r["participants"].replace(" | ", "; "),
                        r["nspk"], " ".join(r["remark"].split()), "yes" if r["f_summary"] else "", "yes" if r["f_report"] else "", "yes" if r["f_materials"] else "", round(r["size"] / 1e6, 1)])
        c.close()
    return "﻿" + buf.getvalue()  # BOM so Excel reads Thai correctly


def delete_meetings(mids: list[str], active: set[str]) -> dict:
    """Delete meeting folders. Meetings that are recording or being processed are refused (their files are in use)."""
    done, skipped = [], []
    for mid in mids[:5000]:
        try:
            d = store.mdir(mid)
        except ValueError:
            skipped.append({"id": mid, "reason": "not a meeting"})
            continue
        st = store.read_json(d / "state.json")
        if not st:
            skipped.append({"id": mid, "reason": "not found"})
        elif mid in active or st.get("status") in ACTIVE or st.get("queue"):
            skipped.append({"id": mid, "title": st.get("title", ""), "reason": "it is being recorded or processed right now"})
        else:
            shutil.rmtree(d, ignore_errors=True)
            done.append({"id": mid, "title": st.get("title", "")})
    sync(force=True)
    return {"deleted": done, "skipped": skipped}
