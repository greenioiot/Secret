"""'Who is this?': unnamed participants of all meetings, longest talk first, with name suggestions for an administrator.

Suggestions come from
  * the meeting's own participant list (names typed in the plan / at the start), always;
  * voices, ONLY when the admin switched on 'cross_meeting_match' (off by default): the voiceprint centroid of the unnamed
    participant compared with enrolled voices (the voice library, collected with consent), with participants NAMED in other
    meetings, and with UNNAMED participants of other meetings ("same voice as ผู้เข้าร่วม 2 on 8 Oct").
Nothing is named or stored automatically: a person confirms with Name (the meeting owner's normal rename) or says
"Not them" (kept in data/whois_rejects.json). Comparisons use the centroids already in speakers.json: no audio is processed
and no voiceprint of an unnamed person is saved anywhere new.
Scores are cosine similarities of speaker embeddings. On the user's data (2026-10-11) the same named person scored a median
0.66 across meetings (min 0.29) and different people a median 0.24 (max 0.67): a suggestion, never proof."""
from __future__ import annotations

import re
import threading

import numpy as np

import config
import speakers
import store

SUGGEST = 0.55        # show a voice suggestion from this similarity
STRONG = 0.70         # "likely"
MIN_TALK_S = 5.0      # shorter participants are not listed (too little speech to judge)
REF_MIN_TALK_S = 10.0 # other meetings' participants used as references need this much speech
SHORT_S = 30.0        # below this the voiceprint is unreliable: said so in the list
REJECTS = config.DATA / "whois_rejects.json"
UNNAMED = re.compile(r"^(?:ผู้เข้าร่วม|ผู้พูด)(?:\s*\d+)?$")
_lock = threading.Lock()


def _unit(c) -> np.ndarray | None:
    v = np.asarray(c or [], dtype=float)
    n = float(np.linalg.norm(v)) if v.size else 0.0
    return v / n if n > 0 else None


def _when(mid: str) -> str:
    """20261008-183948 -> 2026-10-08 18:39"""
    return f"{mid[:4]}-{mid[4:6]}-{mid[6:8]} {mid[9:11]}:{mid[11:13]}" if len(mid) >= 13 else mid


def _label(sid: str) -> str:
    return f"ผู้เข้าร่วม {sid[1:]}"


def rejects() -> dict:
    return store.read_json(REJECTS, {}) or {}


def reject(mid: str, sid: str, cand: str) -> None:
    """'Not them' for one unnamed participant: cand = a name (case-insensitive) or 'mid/sid' of another unnamed participant."""
    cand = cand.strip()[:80]
    if not cand:
        return
    with _lock:
        r = rejects()
        lst = r.setdefault(f"{mid}/{sid}", [])
        if cand.casefold() not in lst:
            lst.append(cand.casefold())
        if "/" in cand:  # symmetric: B is not A either
            r.setdefault(cand, [])
            if f"{mid}/{sid}".casefold() not in r[cand]:
                r[cand].append(f"{mid}/{sid}".casefold())
        store.write_json(REJECTS, r)


def _clip(segs: list, sid: str) -> dict | None:
    """The longest line of this participant (to listen to) and two of their longest lines as text."""
    mine = sorted((s for s in segs if s.get("spk") == sid and s.get("text", "").strip()), key=lambda s: s["t1"] - s["t0"], reverse=True)
    if not mine:
        return None
    a = mine[0]
    return {"t0": round(a["t0"], 1), "t1": round(min(a["t1"], a["t0"] + 12), 1), "lines": [s["text"].strip()[:160] for s in mine[:2]]}


def listing(user: dict, voice_match: bool, mid: str | None = None) -> dict:
    """All meetings, or only meeting `mid` (then every unnamed voice, however short; other meetings still serve as references)."""
    meetings, unnamed, named_refs = [], [], {}
    for st in store.list_meetings():
        if st.get("status") in ("recording", "planned"):
            continue
        spk = store.speakers(st["id"])
        if not spk:
            continue
        meetings.append(st)
        tr = store.read_json(store.mdir(st["id"]) / "transcript.json") or {}
        segs = tr.get("segments", [])
        for sid, v in spk.items():
            c, talk = _unit(v.get("centroid")), float(v.get("talk_s") or 0)
            name = (v.get("name") or "").strip()
            if name and not UNNAMED.match(name):
                if c is not None and talk >= REF_MIN_TALK_S:
                    named_refs.setdefault(name.casefold(), {"name": name, "refs": []})["refs"].append((c, st["id"]))
                continue
            if talk < MIN_TALK_S and not (st["id"] == mid and talk > 0):
                continue
            unnamed.append({"mid": st["id"], "sid": sid, "label": _label(sid), "title": st.get("title", ""), "created": st.get("created", ""),
                            "talk_s": round(talk, 1), "short": talk < SHORT_S, "clip": _clip(segs, sid), "owner": st.get("owner"),
                            "invited": store.clean_participants(st.get("participants") or []),
                            "named_here": sorted({(x.get("name") or "").strip() for x in spk.values() if x.get("name")} - {""}),
                            "_c": c})
    lib = []
    if voice_match:
        for v in speakers.load_voices():
            try:
                lib.append((v["name"], speakers.unit(speakers.voice_centroid(v))))
            except Exception:  # noqa: BLE001 - a broken profile must not hide the list
                continue
    rj = rejects()
    for u in unnamed:
        key, no = f"{u['mid']}/{u['sid']}", set(rejects_of(rj, u))
        here = {n.casefold() for n in u["named_here"]}
        sugg: dict[str, dict] = {}
        if voice_match and u["_c"] is not None:
            for name, c in lib:
                s = float(u["_c"] @ c)
                k = name.casefold()
                if s >= SUGGEST and k not in no and k not in here and s > sugg.get(k, {}).get("score", -1):
                    sugg[k] = {"kind": "voice", "name": name, "score": round(s, 2), "why": "enrolled voice"}
            for k, ref in named_refs.items():
                best = max(((float(u["_c"] @ c), m) for c, m in ref["refs"] if m != u["mid"]), default=(-1.0, None))
                if best[0] >= SUGGEST and k not in no and k not in here and best[0] > sugg.get(k, {}).get("score", -1):
                    sugg[k] = {"kind": "voice", "name": ref["name"], "score": round(best[0], 2), "why": f"named in the meeting of {_when(best[1])}", "ref_mid": best[1]}
        invited = [n for n in u["invited"] if n.casefold() not in here and n.casefold() not in no]
        for s in sugg.values():
            s["invited"] = s["name"].casefold() in {n.casefold() for n in invited}
            s["likely"] = s["score"] >= STRONG
        u["suggestions"] = sorted(sugg.values(), key=lambda s: (not s["likely"], not s["invited"], -s["score"]))[:4]  # likely, then on the list, then score
        u["invite_picks"] = [n for n in invited if n.casefold() not in sugg]  # on the list, no voice evidence: plain quick picks
        same = []
        if voice_match and u["_c"] is not None:
            for o in unnamed:
                if o["mid"] == u["mid"] or o["_c"] is None or o["talk_s"] < REF_MIN_TALK_S:
                    continue
                okey = f"{o['mid']}/{o['sid']}"
                if okey.casefold() in no:
                    continue
                s = float(u["_c"] @ o["_c"])
                if s >= SUGGEST:
                    same.append({"mid": o["mid"], "sid": o["sid"], "label": o["label"], "title": o["title"], "created": o["created"], "score": round(s, 2), "likely": s >= STRONG})
        u["same_voice"] = sorted(same, key=lambda x: -x["score"])[:3]
        problem = None
        try:
            import auth
            problem = auth.edit_problem(user, store.load_state(u["mid"]))
        except Exception:  # noqa: BLE001
            problem = "cannot check the owner"
        u["can_name"], u["why_not"] = problem is None, problem
        u["key"] = key
    for u in unnamed:
        u.pop("_c", None)
    if mid:
        unnamed = [u for u in unnamed if u["mid"] == mid]
    unnamed.sort(key=lambda u: -u["talk_s"])
    return {"voice_match": voice_match, "people": unnamed, "meetings": len(meetings), "unnamed_min": round(sum(u["talk_s"] for u in unnamed) / 60, 1),
            "thresholds": {"suggest": SUGGEST, "likely": STRONG, "short_s": SHORT_S}}


def rejects_of(rj: dict, u: dict) -> list[str]:
    return rj.get(f"{u['mid']}/{u['sid']}", [])
