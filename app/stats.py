"""Statistics per meeting, kept for comparing configurations (engines, models, modes) over many meetings.

meetings/<id>/stats.json holds sections written by the stage that produced them:
  recording   started / ended, duration, live caption model, captions, interim updates, latency of each caption (p50 / p95 / max)
  transcribe  engine, mode, seconds, real-time factor, pieces (reused from the live pass / silent / computed), source mix, confidence,
              disagreement with the fast Typhoon pass, CPU / RAM during the stage, NPU call statistics
  speakers    seconds, number of speakers
  summary     model, backend (NPU both cores / one core / CPU), seconds, model calls, prompt / generation tokens and tokens per second, sizes
  done        finished_at, time to result (stop -> done), queue wait
Meetings made before this existed are back-filled from their history lines and files (marked "backfilled").
Quality has no ground truth here, so there are proxies: confidence, disagreement between engines, the share of lines YOU corrected, and your 1-5 rating."""
from __future__ import annotations

import csv
import fcntl
import io
import json
import re
import statistics
import threading
import time
from contextlib import contextmanager

import config
import store


def _file(mid: str):
    return store.mdir(mid) / "stats.json"


def load(mid: str) -> dict:
    return store.read_json(_file(mid)) or {}


def record(mid: str, section: str, **fields):
    """Merge fields into a section of the meeting's stats.json (safe from the server and the worker at once)."""
    try:
        d = store.mdir(mid)
        with open(d / "stats.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            cur = load(mid)
            cur.setdefault(section, {}).update(fields)
            cur.pop("backfilled", None) if section in ("recording", "transcribe", "summary") and not fields.get("backfilled") else None
            store.write_json(_file(mid), cur)
    except OSError:
        pass  # statistics must never break a meeting


class Sampler(threading.Thread):
    """CPU %, free RAM and temperature while a stage runs (every 3 s, read straight from /proc and /sys)."""

    def __init__(self, period: float = 3.0):
        super().__init__(daemon=True)
        self.period, self.stop_ev, self.cpu, self.mem, self.temp = period, threading.Event(), [], [], []

    @staticmethod
    def _cpu():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]

    def run(self):
        try:
            t0, i0 = self._cpu()
            while not self.stop_ev.wait(self.period):
                t1, i1 = self._cpu()
                if t1 > t0:
                    self.cpu.append(100.0 * (1 - (i1 - i0) / (t1 - t0)))
                t0, i0 = t1, i1
                for ln in open("/proc/meminfo"):
                    if ln.startswith("MemAvailable:"):
                        self.mem.append(int(ln.split()[1]) / 1048576)
                        break
                from pathlib import Path

                ts = []
                for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
                    try:
                        ts.append(int(p.read_text()) / 1000)
                    except (OSError, ValueError):
                        pass
                if ts:
                    self.temp.append(max(ts))
        except Exception:  # noqa: BLE001
            pass

    def summary(self) -> dict:
        return {"cpu_avg_pct": round(statistics.mean(self.cpu), 1) if self.cpu else None, "cpu_peak_pct": round(max(self.cpu), 1) if self.cpu else None,
                "ram_free_min_gb": round(min(self.mem), 1) if self.mem else None, "temp_max_c": round(max(self.temp), 1) if self.temp else None}


@contextmanager
def stage(mid: str, name: str, **fields):
    """with stats.stage(mid, 'transcribe') as st: ...  -> st['x'] = ... adds fields; seconds and resource use are recorded on exit."""
    st: dict = dict(fields)
    smp = Sampler()
    smp.start()
    t0 = time.time()
    try:
        yield st
    finally:
        smp.stop_ev.set()
        st.update(smp.summary())
        st["secs"] = round(time.time() - t0 - float(st.get("paused_rec_s") or 0), 1)  # waits for a recording are not processing time
        st["ended_at"] = time.strftime("%F %T")
        record(mid, name, **st)


def _pct(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    return round(v[min(len(v) - 1, int(p / 100 * len(v)))], 2)


def latency_summary(lat: list[float]) -> dict:
    return {"n": len(lat), "p50_s": _pct(lat, 50), "p95_s": _pct(lat, 95), "max_s": round(max(lat), 2) if lat else None}


# ---------------------------------------------------------------- back-fill from history and files (meetings made before the statistics existed)
_T = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"


def _ts(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def backfill(mid: str) -> dict:
    cur = load(mid)
    if cur.get("done") and cur.get("transcribe") and cur.get("summary") is not None:
        return cur
    st = store.load_state(mid) or {}
    hist = st.get("history", [])
    d = store.mdir(mid)
    tr = store.read_json(d / "transcript.json") or {}
    new: dict = {}
    for h in hist:
        m = re.match(_T + r" recording finished, (\d+)s", h)
        if m and "recording" not in cur:
            new["recording"] = {"ended": m.group(1), "duration_s": float(m.group(2)), "backfilled": True}
        m = re.match(_T + r" transcribed (\d+) segments in (\d+)s \((\w+)\)", h)
        if m and "transcribe" not in cur:
            new["transcribe"] = {"ended_at": m.group(1), "segments": int(m.group(2)), "secs": float(m.group(3)), "mode": m.group(4), "backfilled": True}
        m = re.match(_T + r" identified (\d+) speaker\(s\) in (\d+)s", h)
        if m and "speakers" not in cur:
            new["speakers"] = {"n": int(m.group(2)), "secs": float(m.group(3)), "backfilled": True}
        m = re.match(_T + r" summary and report written in (\d+)s", h)
        if m and "summary" not in cur:
            new["summary"] = {**new.get("summary", {}), "secs": float(m.group(2)), "ended_at": m.group(1), "backfilled": True}
        m = re.match(_T + r" summary model running on (.+)", h)
        if m:
            new.setdefault("summary", {}).update({"backend": m.group(2).strip(), "backfilled": True})
    dur = (new.get("recording") or cur.get("recording") or {}).get("duration_s") or st.get("duration_s") or 0
    if "recording" not in cur and "recording" not in new and dur:
        new["recording"] = {"duration_s": float(dur), "backfilled": True}
    if "transcribe" in new or "transcribe" in cur:
        tt = {**cur.get("transcribe", {}), **new.get("transcribe", {})}
        segs = tr.get("segments", [])
        srcs: dict = {}
        for s in segs:
            srcs[s.get("src", "?")] = srcs.get(s.get("src", "?"), 0) + 1
        tt.setdefault("engine", tr.get("engine"))
        tt.setdefault("src_mix", srcs)
        tt.setdefault("segments", len(segs))
        if dur and tt.get("secs"):
            tt.setdefault("rtf", round(tt["secs"] / dur, 3))
        new["transcribe"] = tt
    if "summary" in new and dur and new["summary"].get("secs"):
        new["summary"].setdefault("rtf", round(new["summary"]["secs"] / dur, 3))
    if new.get("summary") is not None:
        new["summary"].setdefault("model", "typhoon-4b (assumed)" if "NPU" in str(new["summary"].get("backend")) else None)
    fin = [h for h in hist if "summary and report written" in h]
    rec = [h for h in hist if "recording finished" in h]
    fin = [h for h in fin if rec and h[:19] >= rec[0][:19]]  # the first result after the recording; later lines are reprocessing runs
    if fin and rec and "done" not in cur:
        try:
            new["done"] = {"finished_at": fin[0][:19], "time_to_result_s": round(_ts(fin[0][:19]) - _ts(rec[0][:19])), "backfilled": True}
        except ValueError:
            pass
    if new:
        for k, v in new.items():
            record(mid, k, **v)
        cur = load(mid)
    return cur


# ---------------------------------------------------------------- the flat rows used by the comparison screens
def _levenshtein_ratio(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b) / max(1, len(a))


def row(mid: str) -> dict | None:
    st = store.load_state(mid)
    if not st or st.get("status") in ("recording", "planned"):
        return None
    s = backfill(mid)
    d = store.mdir(mid)
    tr = store.read_json(d / "transcript.json") or {}
    segs = tr.get("segments", [])
    rec, tc, sp, sm, dn = (s.get(k) or {} for k in ("recording", "transcribe", "speakers", "summary", "done"))
    dur = rec.get("duration_s") or st.get("duration_s") or 0
    n_edit = sum(1 for x in segs if x.get("src") == "edited")
    confs = [x["conf"] for x in segs if isinstance(x.get("conf"), (int, float))]
    chars = sum(len(x["text"]) for x in segs)
    mode = tc.get("mode") or tc.get("accuracy")
    r = {
        "id": mid, "title": st.get("title", ""), "created": st.get("created", ""), "week": time.strftime("%G-W%V", time.strptime(st["created"][:10], "%Y-%m-%d")) if st.get("created") else "",
        "duration_min": round(dur / 60, 1) if dur else None, "speakers": sp.get("n") or len((store.read_json(d / "speakers.json") or {})),
        # dimensions
        "accuracy_mode": mode, "asr_engine": (tc.get("engine") or tr.get("engine") or "").replace("+typhoon", ""), "live_engine": rec.get("live_engine"),
        "summary_model": sm.get("model"), "summary_backend": re.sub(r"\s*\(.*", "", str(sm.get("backend") or "")) or None,
        "packs": ",".join(tc.get("vocab_packs") or []) or None,
        # speed
        "transcribe_s": tc.get("secs"), "rtf_transcribe": round(tc["secs"] / dur, 3) if dur and tc.get("secs") else None, "speakers_s": sp.get("secs"), "summary_s": sm.get("secs"),
        "rtf_summary": round(sm["secs"] / dur, 3) if dur and sm.get("secs") else None,
        "time_to_result_min": round(dn["time_to_result_s"] / 60, 1) if dn.get("time_to_result_s") else None,
        "summary_prompt_tps": sm.get("prompt_tps"), "summary_gen_tps": sm.get("gen_tps"), "summary_calls": sm.get("calls"),
        # resources
        "cpu_avg_transcribe": tc.get("cpu_avg_pct"), "cpu_peak_transcribe": tc.get("cpu_peak_pct"), "ram_free_min_gb": min([v for v in (tc.get("ram_free_min_gb"), sm.get("ram_free_min_gb")) if v] or [None]) if any((tc.get("ram_free_min_gb"), sm.get("ram_free_min_gb"))) else None,
        "temp_max_c": max([v for v in (tc.get("temp_max_c"), sm.get("temp_max_c")) if v] or [None]) if any((tc.get("temp_max_c"), sm.get("temp_max_c"))) else None,
        # live captions
        "captions": (rec.get("captions") or None), "caption_latency_p50_s": (rec.get("latency") or {}).get("p50_s"), "caption_latency_p95_s": (rec.get("latency") or {}).get("p95_s"), "interim_updates": rec.get("partials"),
        # quality proxies
        "segments": len(segs) or None, "chars_per_min": round(chars / (dur / 60)) if dur and chars else None, "conf_mean": round(statistics.mean(confs), 3) if confs else None,
        "low_conf_pct": round(100 * sum(1 for c in confs if c < 0.8) / len(confs), 1) if confs else None,
        "disagreement_pct": tc.get("disagreement_pct"), "edited_lines_pct": round(100 * n_edit / len(segs), 1) if segs else None,
        "summary_edited_pct": (100.0 if st.get("summary_edited") else 0.0) if (d / "summary.md").exists() else None,
        "term_corrections": tc.get("term_corrections"), "decisions_found": sm.get("decisions_found"), "tasks_found": sm.get("tasks_found"), "summary_chars": sm.get("summary_chars"),
        "rating_transcript": (st.get("rating") or {}).get("transcript"), "rating_summary": (st.get("rating") or {}).get("summary"),
        "backfilled": bool(tc.get("backfilled") or sm.get("backfilled")),
    }
    return r


def rows(date_from: str = "", date_to: str = "") -> list[dict]:
    out = []
    for st in store.list_meetings():
        c = st.get("created", "")[:10]
        if (date_from and c < date_from) or (date_to and c > date_to):
            continue
        try:
            r = row(st["id"])
        except Exception:  # noqa: BLE001 - one broken meeting must not hide the others
            r = None
        if r:
            out.append(r)
    return out


METRICS = {  # key -> (label, unit, "lower is better" | "higher is better" | "")
    "rtf_transcribe": ("Transcription speed (real-time factor)", "×", "lower"), "transcribe_s": ("Transcription time", "s", "lower"),
    "summary_s": ("Summary time", "s", "lower"), "rtf_summary": ("Summary time / meeting length", "×", "lower"), "time_to_result_min": ("Stop → finished", "min", "lower"),
    "summary_prompt_tps": ("Summary: reading speed", "tok/s", "higher"), "summary_gen_tps": ("Summary: writing speed", "tok/s", "higher"),
    "cpu_avg_transcribe": ("CPU while transcribing (average)", "%", "lower"), "ram_free_min_gb": ("Lowest free RAM", "GB", "higher"), "temp_max_c": ("Highest temperature", "°C", "lower"),
    "caption_latency_p50_s": ("Live caption delay (median)", "s", "lower"), "caption_latency_p95_s": ("Live caption delay (95 %)", "s", "lower"),
    "conf_mean": ("Mean confidence", "", "higher"), "low_conf_pct": ("Low-confidence lines", "%", "lower"), "disagreement_pct": ("Disagreement with the fast Typhoon pass", "%", "lower"),
    "edited_lines_pct": ("Lines you corrected", "%", "lower"), "summary_edited_pct": ("Summaries you corrected by hand", "% of meetings", "lower"), "term_corrections": ("Term corrections applied", "", ""), "chars_per_min": ("Text per minute", "chars", ""),
    "decisions_found": ("Decisions found", "", ""), "tasks_found": ("Tasks found", "", ""), "rating_transcript": ("Your rating: transcript", "/5", "higher"), "rating_summary": ("Your rating: summary", "/5", "higher"),
    "duration_min": ("Meeting length", "min", ""), "speakers": ("Participants", "", ""),
}
DIMENSIONS = {"accuracy_mode": "Accuracy mode", "asr_engine": "Speech engine", "live_engine": "Live caption model", "summary_model": "Summary model", "summary_backend": "Summary runs on",
              "week": "Week", "packs": "Vocabulary packs", "speakers": "Number of participants"}


def compare(metric: str, by: str, date_from: str = "", date_to: str = "") -> dict:
    if metric not in METRICS or by not in DIMENSIONS:
        raise ValueError("unknown metric or grouping")
    groups: dict[str, list] = {}
    for r in rows(date_from, date_to):
        v = r.get(metric)
        if v is None:
            continue
        groups.setdefault(str(r.get(by) if r.get(by) not in (None, "") else "(not recorded)"), []).append((float(v), r["id"]))
    out = []
    for k, vs in groups.items():
        nums = [v for v, _ in vs]
        out.append({"group": k, "n": len(nums), "mean": round(statistics.mean(nums), 3), "median": round(statistics.median(nums), 3), "min": round(min(nums), 3), "max": round(max(nums), 3), "meetings": [m for _, m in vs]})
    out.sort(key=lambda g: g["group"])
    lab, unit, better = METRICS[metric]
    return {"metric": metric, "label": lab, "unit": unit, "better": better, "by": by, "by_label": DIMENSIONS[by], "groups": out}


def export_csv(date_from: str = "", date_to: str = "") -> str:
    rs = rows(date_from, date_to)
    out = io.StringIO()
    out.write("﻿")
    if rs:
        w = csv.DictWriter(out, fieldnames=list(rs[0].keys()))
        w.writeheader()
        w.writerows(rs)
    return out.getvalue()


# ---------------------------------------------------------------- participants across meetings
UNNAMED = re.compile(r"^(?:ผู้เข้าร่วม|ผู้พูด)(?:\s*\d+)?$")  # default labels: they only mean something inside one meeting
LEVELS = ("leading", "balanced", "moderate", "quiet", "silent")


def participants(date_from: str = "", date_to: str = "") -> dict:
    """Speaking participation per NAMED person over the meetings in the date range (names matched case-insensitively).
    Computed fresh from each transcript (report.compute), so names given after a report was written count. Measures how
    much people spoke, not the quality of what they said or whether they listened."""
    import report  # report imports store only; local import keeps stats importable on its own
    people: dict[str, dict] = {}
    tot = {"meetings": 0, "analysed": 0, "not_analysed": 0, "meeting_s": 0.0, "talk_s": 0.0, "named_talk_s": 0.0, "unnamed_talk_s": 0.0, "unnamed_labels": 0}
    for st in store.list_meetings():
        c = st.get("created", "")[:10]
        if (date_from and c < date_from) or (date_to and c > date_to) or st.get("status") in ("recording", "planned"):
            continue
        try:
            rep = report.compute(st["id"])
        except Exception:  # noqa: BLE001 - one broken meeting must not hide the others
            rep = None
        if not rep:
            continue
        o = rep["overview"]
        tot["meetings"] += 1
        tot["meeting_s"] += o["duration_s"]
        tot["talk_s"] += o["talk_s"]
        if not o["identified"]:
            tot["not_analysed"] += 1
            continue
        tot["analysed"] += 1
        spoke = [p for p in rep["people"] if not p.get("silent")]
        for p in rep["people"]:
            if UNNAMED.match(p["name"]):
                tot["unnamed_talk_s"] += p["talk_s"]
                tot["unnamed_labels"] += 1
                continue
            tot["named_talk_s"] += p["talk_s"]
            k = p["name"].casefold()
            a = people.setdefault(k, {"name": p["name"], "meetings": 0, "spoke_in": 0, "talk_s": 0.0, "meeting_s": 0.0, "turns": 0, "questions": 0, "chars": 0,
                                      "longest_turn_s": 0.0, "shares": [], "levels": dict.fromkeys(LEVELS, 0), "first": c, "last": c, "list": []})
            a["meetings"] += 1
            a["spoke_in"] += 0 if p.get("silent") else 1
            a["talk_s"] += p["talk_s"]
            a["meeting_s"] += o["duration_s"]
            a["turns"] += p["turns"]
            a["questions"] += p["questions"]
            a["chars"] += p["chars"]
            a["longest_turn_s"] = max(a["longest_turn_s"], p["longest_turn_s"])
            if len(spoke) > 1 and not p.get("silent"):
                a["shares"].append(p["share"])  # share of the talk only means something when others spoke too
            if p.get("level") and (p.get("silent") or len(spoke) > 1):  # alone, the share is always 100 %: no level
                a["levels"][p["level"]] = a["levels"].get(p["level"], 0) + 1
            a["first"], a["last"] = min(a["first"], c), max(a["last"], c)
            if c >= a["last"]:
                a["name"] = p["name"]  # the most recent spelling
            a["list"].append({"id": st["id"], "title": st.get("title", ""), "created": st.get("created", ""), "duration_min": round(o["duration_s"] / 60, 1),
                              "talk_min": round(p["talk_s"] / 60, 1), "share": p["share"], "time_pct": round(100 * p["talk_s"] / max(1.0, o["duration_s"]), 1),
                              "turns": p["turns"], "questions": p["questions"], "level": p.get("level") if (p.get("silent") or len(spoke) > 1) else None, "others": len(spoke) - (0 if p.get("silent") else 1)})
    out = []
    for a in people.values():
        lv = {k: v for k, v in a["levels"].items() if v}
        out.append({
            "name": a["name"], "meetings": a["meetings"], "spoke_in": a["spoke_in"], "talk_min": round(a["talk_s"] / 60, 1), "meeting_min": round(a["meeting_s"] / 60, 1),
            "time_pct": round(100 * a["talk_s"] / a["meeting_s"], 1) if a["meeting_s"] else None,  # of the time of the meetings they were in
            "avg_share_pct": round(100 * statistics.mean(a["shares"]), 1) if a["shares"] else None,   # of the talk, meetings with 2+ speaking people
            "turns": a["turns"], "avg_turn_s": round(a["talk_s"] / a["turns"], 1) if a["turns"] else None, "longest_turn_s": round(a["longest_turn_s"], 1),
            "questions": a["questions"], "questions_per_hour": round(a["questions"] / (a["talk_s"] / 3600), 1) if a["talk_s"] >= 60 else None,
            "chars_per_min": round(a["chars"] / (a["talk_s"] / 60)) if a["talk_s"] >= 30 else None,
            "typical": max(lv, key=lv.get) if lv else None, "levels": lv, "first": a["first"], "last": a["last"],
            "meetings_list": sorted(a["list"], key=lambda m: m["created"], reverse=True)})
    out.sort(key=lambda p: -p["talk_min"])
    totals = {k: (round(v / 60, 1) if k.endswith("_s") else v) for k, v in tot.items()}
    totals = {k.replace("_s", "_min") if k.endswith("_s") else k: v for k, v in totals.items()}
    totals["named"] = len(out)
    totals["named_pct"] = round(100 * tot["named_talk_s"] / tot["talk_s"], 1) if tot["talk_s"] else None
    return {"people": out, "totals": totals}


def participants_csv(date_from: str = "", date_to: str = "") -> str:
    ps = participants(date_from, date_to)["people"]
    out = io.StringIO()
    out.write("﻿")
    cols = ["name", "meetings", "spoke_in", "talk_min", "meeting_min", "time_pct", "avg_share_pct", "turns", "avg_turn_s", "longest_turn_s", "questions",
            "questions_per_hour", "chars_per_min", "typical", "first", "last"]
    w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(ps)
    return out.getvalue()
