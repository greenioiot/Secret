"""When meetings are processed and when each one will be ready.

Processing window: the work after a meeting (transcript, speakers, summary) runs only outside working hours, e.g. 18:00-06:00 and all
weekend (settings batch_night / batch_start / batch_end / batch_weekends, set by an administrator). A meeting marked "rush" (Process now,
or reprocessing asked by an administrator) runs at once. The worker stops at the next checkpoint when the window closes and goes on
from there when it opens again.

estimate() simulates the worker's two lanes (transcript lane; speakers + summary lane) over the queue, inside the window, with stage
times learned from this box's past meetings, and gives every waiting or running meeting its expected start and finish time."""
from __future__ import annotations

import os
import statistics
import time

import store

DAY = 86400
# stage time = fixed + per_s x meeting seconds (measured on this box; corrected by its own past meetings, see _ratio)
TRANSCRIBE = {"vote3": (150, 4.6), "best": (120, 3.0), "pathumma-large": (90, 2.3), "large": (90, 2.9), "pathumma-medium": (60, 1.6),
              "accurate": (30, 0.55), "pathumma-rnnt": (15, 0.15), "npu": (20, 0.1), "fast": (10, 0.05)}
TRANSCRIBE_OTHER = (60, 1.0)
SPEAKERS = (5, 0.05)
SUMMARY = {"typhoon-30b": (600, 0.35), "typhoon-4b": (200, 0.25)}
GRACE_S = int(os.environ.get("SECRET_GRACE_S", 15 * 60))  # a stage that will finish within this when the window closes is allowed to finish (tests: 0)


def _hm(s: str) -> int:
    try:
        h, m = str(s).split(":")
        return max(0, min(23, int(h))) * 3600 + max(0, min(59, int(m))) * 60
    except ValueError:
        return 0


def valid_hm(s) -> bool:
    try:
        h, m = str(s).split(":")
        return len(m) == 2 and 0 <= int(h) <= 23 and 0 <= int(m) <= 59
    except ValueError:
        return False


def _day_start(t: float) -> float:
    lt = time.localtime(t)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


def _open_periods(cfg: dict, t: float, days: int = 9):
    """Yield the open (start, end) periods from the one containing or following t, in order, merged."""
    a, b, wk = _hm(cfg.get("batch_start", "18:00")), _hm(cfg.get("batch_end", "06:00")), bool(cfg.get("batch_weekends", True))
    raw = []
    d0 = _day_start(t) - DAY
    for i in range(days + 2):
        ds = _day_start(d0 + i * DAY + 3600)  # +1 h: safe across a daylight-saving change
        if wk and time.localtime(ds).tm_wday >= 5:
            raw.append((ds, _day_start(ds + DAY + 3600)))
        if a == b:
            raw.append((ds, _day_start(ds + DAY + 3600)))  # the same start and end: open all day
        elif a < b:
            raw.append((ds + a, ds + b))
        else:
            raw.append((ds, ds + b))
            raw.append((ds + a, _day_start(ds + DAY + 3600)))
    raw.sort()
    merged: list[list[float]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for s, e in merged:
        if e > t:
            yield max(s, t), e


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("batch_night"))


def is_open(cfg: dict, t: float | None = None) -> bool:
    if not enabled(cfg):
        return True
    t = time.time() if t is None else t
    for s, e in _open_periods(cfg, t, 2):
        return s <= t < e
    return False


def next_open(cfg: dict, t: float | None = None) -> float:
    t = time.time() if t is None else t
    if not enabled(cfg):
        return t
    for s, _e in _open_periods(cfg, t):
        return s
    return t


def closes_at(cfg: dict, t: float | None = None) -> float | None:
    """When the open period containing t ends (None: always open)."""
    t = time.time() if t is None else t
    if not enabled(cfg):
        return None
    for s, e in _open_periods(cfg, t):
        return e if s <= t else None
    return None


def may_run(cfg: dict, st: dict, t: float | None = None) -> bool:
    return bool(st.get("rush")) or is_open(cfg, t)


def work_until(cfg: dict, start: float, secs: float, rush: bool = False) -> float:
    """The time a piece of work of `secs` seconds started (or waiting) at `start` is finished, working only in open periods."""
    if rush or not enabled(cfg):
        return start + secs
    left = secs
    for s, e in _open_periods(cfg, start, 400):
        if s + left <= e:
            return s + left
        left -= e - s
    return start + secs


# ---------------------------------------------------------------- stage times
_cache: dict = {"t": 0.0, "ratios": {}}


def _ratio(key: str) -> float:
    """measured / predicted for this kind of stage over the box's recent meetings (1 = the table above is right)."""
    if time.time() - _cache["t"] > 300:
        import stats

        acc: dict = {}
        for st in store.list_meetings()[:40]:
            try:
                s = stats.load(st["id"])
            except Exception:  # noqa: BLE001
                continue
            dur = (s.get("recording") or {}).get("duration_s") or st.get("duration_s") or 0
            if dur < 20:
                continue
            tc, sp, sm = s.get("transcribe") or {}, s.get("speakers") or {}, s.get("summary") or {}
            mode = tc.get("mode") or tc.get("accuracy")
            if tc.get("secs") and mode and not tc.get("backfilled") and not tc.get("split"):
                acc.setdefault("t:" + mode, []).append(tc["secs"] / _secs(TRANSCRIBE.get(mode, TRANSCRIBE_OTHER), dur))
            if sp.get("secs") and not sp.get("backfilled") and not sp.get("split"):
                acc.setdefault("speakers", []).append(sp["secs"] / _secs(SPEAKERS, dur))
            mk = _summary_key(str(sm.get("model") or ""))
            if sm.get("secs") and mk and not sm.get("split"):
                acc.setdefault("s:" + mk, []).append(sm["secs"] / _secs(SUMMARY[mk], dur))
        _cache.update(t=time.time(), ratios={k: min(3.0, max(0.33, statistics.median(v[:10]))) for k, v in acc.items()})
    return _cache["ratios"].get(key, 1.0)


def _summary_key(model: str) -> str | None:
    m = model.lower()
    return "typhoon-30b" if "30b" in m else "typhoon-4b" if "4b" in m else None


def _secs(rule, dur: float) -> float:
    return rule[0] + rule[1] * dur


def stage_secs(stage: str, dur: float, cfg: dict) -> float:
    if stage == "transcribe":
        mode = cfg.get("accuracy", "accurate")
        return _secs(TRANSCRIBE.get(mode, TRANSCRIBE_OTHER), dur) * _ratio("t:" + mode)
    if stage == "speakers":
        return _secs(SPEAKERS, dur) * _ratio("speakers")
    if stage in ("summarize", "report"):
        key = cfg.get("summary_model", "typhoon-4b")
        key = key if key in SUMMARY else "typhoon-4b"  # a removed model falls back to the 4B, like engines.summary_plan
        t = _secs(SUMMARY[key], dur) * _ratio("s:" + key)
        return t * (0.4 if stage == "report" else 1.0)
    return 0.0


def stages_left(st: dict, cfg: dict) -> list[str]:
    """The stages still to run for a meeting, its current one first."""
    q, status = st.get("queue"), st.get("status")
    cur = {"transcribing": "transcribe", "analysing": "speakers", "summarizing": "summarize"}.get(status) or q
    if not cur:
        return []
    if cur == "report" or (status == "summarizing" and st.get("phase") == "report"):
        return ["report"]
    order = ["transcribe"]
    if cfg.get("identify_speakers") and cfg.get("speakers_after"):
        order.append("speakers")
    if cfg.get("auto_summary"):
        order.append("summarize")
    if cur not in order:
        return [cur]
    return order[order.index(cur):]


# ---------------------------------------------------------------- the whole queue
def estimate(cfg: dict | None = None, now: float | None = None, metas: list[dict] | None = None) -> dict:
    """-> {"window": {...}, "meetings": {mid: {"start", "finish", "stages": [[stage, start, end]], "running", "rush"}}} (epoch seconds)."""
    cfg = cfg or store.settings()
    now = time.time() if now is None else now
    metas = store.list_meetings() if metas is None else metas
    jobs = []
    for st in reversed(metas):  # oldest first, like the worker
        if st.get("status") in ("recording", "planned", "interrupted", "error"):
            continue
        left = stages_left(st, cfg)
        if not left:
            continue
        dur = float(st.get("duration_s") or 0)
        secs = [stage_secs(s, dur, cfg) for s in left]
        running = st.get("status") in ("transcribing", "analysing", "summarizing")
        if running:  # the stage under way: what the worker measures beats the table
            eta = st.get("eta_s")
            if isinstance(eta, (int, float)):
                secs[0] = max(0.0, eta - max(0.0, now - float(st.get("eta_at") or now)))
            else:
                secs[0] *= max(0.05, 1 - float(st.get("progress") or 0))
        jobs.append({"id": st["id"], "title": st.get("title") or "", "left": left, "secs": secs, "running": running, "rush": bool(st.get("rush")), "plan": []})

    # lane 1: transcripts, in order (a rush meeting goes first, as the worker skips the others while the window is closed)
    t_free = now
    ready: dict = {}
    tq = [j for j in jobs if j["left"][0] == "transcribe"]
    tq.sort(key=lambda j: (not j["running"], not j["rush"]))
    for j in tq:
        s = t_free if j["rush"] or j["running"] else max(t_free, next_open(cfg, t_free))
        e = work_until(cfg, t_free, j["secs"][0], j["rush"] or (j["running"] and _grace(cfg, now, j["secs"][0])))
        j["plan"].append(["transcribe", s, e])
        t_free = e
        ready[j["id"]] = e
    # lane 2: speakers / summary, oldest ready first
    post = []
    for j in jobs:
        k = 1 if j["left"][0] == "transcribe" else 0
        if k < len(j["left"]):
            post.append((ready.get(j["id"], now), j, k))
    t_free = now
    waiting = sorted(post, key=lambda x: (not (x[1]["running"] and x[2] == 0), x[0]))  # the stage under way first
    while waiting:
        avail = [w for w in waiting if w[0] <= t_free] or [min(waiting, key=lambda w: w[0])]
        r, j, k = avail[0]
        waiting.remove(avail[0])
        t = max(t_free, r)
        for i in range(k, len(j["left"])):
            free = j["rush"] or (j["running"] and i == 0 and _grace(cfg, now, j["secs"][0]))
            s = t if free else next_open(cfg, t)
            e = work_until(cfg, t, j["secs"][i], free)
            j["plan"].append([j["left"][i], s, e])
            t = e
        t_free = t
    out = {}
    for j in jobs:
        if j["plan"]:
            out[j["id"]] = {"title": j["title"], "start": round(j["plan"][0][1]), "finish": round(j["plan"][-1][2]), "running": j["running"], "rush": j["rush"],
                            "stages": [[s, round(a), round(b)] for s, a, b in j["plan"]]}
    return {"window": window_info(cfg, now), "meetings": out}


def _grace(cfg: dict, now: float, secs: float) -> bool:
    c = closes_at(cfg, now)
    return c is None or now + secs <= c + GRACE_S


def window_info(cfg: dict, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    on = enabled(cfg)
    o = is_open(cfg, now)
    return {"on": on, "start": cfg.get("batch_start", "18:00"), "end": cfg.get("batch_end", "06:00"), "weekends": bool(cfg.get("batch_weekends", True)),
            "open": o, "opens_at": None if o else round(next_open(cfg, now)), "closes_at": round(closes_at(cfg, now)) if on and o and closes_at(cfg, now) else None}
