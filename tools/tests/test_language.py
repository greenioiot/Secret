"""Round 54: the meeting language (Thai | Thai + English | English) through the live captions, the transcript and the summary.
Needs a scratch instance on 8445 (see README) and a folder with english.wav (10 s of English speech, 16 kHz mono), mixed.wav (Thai then English),
thai.wav (Thai speech). Run: .venv/bin/python -W ignore tools/tests/test_language.py <folder>"""
import asyncio, json, ssl, sys, time, wave
import httpx, websockets
S = sys.argv[1]; B = "https://127.0.0.1:8445"
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
ok = fail = 0
def check(name, cond, info=""):
    global ok, fail
    ok += cond; fail += (not cond); print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{str(info)[:200]}]"))
def pcm(path):
    w = wave.open(path); assert w.getframerate() == 16000; return w.readframes(w.getnframes())
c = httpx.Client(base_url=B, verify=False, timeout=120, headers={"Origin": B})
assert c.post("/api/auth/login", json={"username": "boss", "password": "adminpassword123"}).status_code == 200
cookie = "; ".join(f"{k}={v}" for k, v in c.cookies.items()); H = {"X-CSRF-Token": c.cookies.get("secret_csrf")}
put = lambda d: c.put("/api/settings", headers=H, json=d).json()
s = put({"batch_night": False, "auto_process": True, "auto_summary": True, "allow_join": False, "accuracy": "pathumma-rnnt", "language": "th", "identify_speakers": False, "live_accurate": False})
check("settings: language default th", s.get("language") == "th", s.get("language"))
check("settings: an unknown language falls back to th", put({"language": "xx"}).get("language") == "th")
e = c.get("/api/engines").json()
check("/api/engines: three languages + default", set(e.get("languages", {})) == {"th", "th-en", "en"} and e.get("language_default") == "th", e.get("languages"))
lp = e.get("language_plans", {})
check("plan en: English Whisper after, small.en live", "Turbo" in lp.get("en", {}).get("after", {}).get("uses", "") and "small.en" in lp.get("en", {}).get("live", {}).get("name", ""), lp.get("en"))
check("plan th-en: a Thai-only choice is replaced by Large v3", "has no English" in lp.get("th-en", {}).get("after", {}).get("note", "") and "Large v3" in lp.get("th-en", {}).get("after", {}).get("name", ""), lp.get("th-en"))
check("plan th: unchanged", lp.get("th", {}).get("after", {}).get("mode") == "pathumma-rnnt", lp.get("th"))
# planned meeting with a language
p = c.post("/api/meetings/plan", headers=H, json={"title": "plan-en", "language": "en"}).json()
check("plan: language stored", p.get("language") == "en", p)
check("patch: bad language -> 400", c.patch(f"/api/meetings/{p['id']}", headers=H, json={"language": "xx"}).status_code == 400)
check("patch: th-en accepted", c.patch(f"/api/meetings/{p['id']}", headers=H, json={"language": "th-en"}).json().get("language") == "th-en")
c.delete(f"/api/meetings/{p['id']}", headers=H)

async def meeting(title, mode, lang, wav, tail_s=1.5):
    msgs = []
    ws = await websockets.connect(f"wss://127.0.0.1:8445/ws/record?mode={mode}&title={title}&lang={lang}", ssl=ctx, additional_headers={"Cookie": cookie, "Origin": B})
    async def reader():
        try:
            async for m in ws: msgs.append(json.loads(m))
        except Exception: pass
    rd = asyncio.create_task(reader())
    data = pcm(wav) + b"\0" * int(tail_s * 32000)
    for i in range(0, len(data), 640):
        await ws.send(data[i:i + 640]); await asyncio.sleep(0.02)
    await asyncio.sleep(4)  # the last sentence's caption
    await ws.send(json.dumps({"type": "stop"}))
    try: await asyncio.wait_for(rd, 30)
    except Exception: pass
    return msgs
def wait_done(mid, limit=600, until=("done",)):
    t = time.time()
    while time.time() - t < limit:
        st = c.get(f"/api/meetings/{mid}").json()["state"]
        if st["status"] in until or st["status"] == "error": return st
        time.sleep(3)
    return st
thai = lambda t: any("฀" <= ch <= "๿" for ch in t)

# 1. English, live captions
m = asyncio.run(meeting("english", "live", "en", f"{S}/english.wav"))
st_msg = next((x for x in m if x["type"] == "started"), {})
mid = st_msg.get("id")
check("en live: started with language en + English live model", st_msg.get("engines", {}).get("language") == "en" and "small.en" in st_msg.get("engines", {}).get("live", {}).get("name", ""), st_msg.get("engines"))
caps = " ".join(x["text"] for x in m if x["type"] == "caption")
check("en live: English captions arrived", "budget" in caps.lower() or "forecast" in caps.lower(), caps)
check("en live: no Thai in the captions", caps and not thai(caps), caps)
st = wait_done(mid, 900)
check("en: processed to done", st["status"] == "done", st.get("status", "") + " " + str(st.get("error")))
tr = (c.get(f"/api/meetings/{mid}").json().get("transcript") or {}) if st["status"] == "done" else {}
txt = " ".join(s["text"] for s in tr.get("segments", []))
check("en transcript: English Whisper engine", "turbo-en" in (tr.get("engine") or ""), tr.get("engine"))
check("en transcript: the sentence, no Thai", "third quarter" in txt.lower() and "forecast" in txt.lower() and not thai(txt), txt)
smd = (c.get(f"/api/meetings/{mid}").json().get("summary") or "") if st["status"] == "done" else ""
check("en summary: English headings", "## Overview" in smd and "## Decisions" in smd and "## Action items" in smd, smd[:300])
check("en summary: written in English", smd and not thai(smd) and ("budget" in smd.lower() or "forecast" in smd.lower()), smd[:400])
hist = " ".join(st.get("history", []))
check("en: no 'has no English' switch note (the English model was used)", "has no English" not in hist, hist[-300:])
put({"auto_summary": False})
# 2. Thai + English, record only: the Thai-only choice is replaced by a Whisper model, English kept
m = asyncio.run(meeting("mixed", "after", "th-en", f"{S}/mixed.wav"))
mid2 = next((x["id"] for x in m if x["type"] == "started"), None)
st2 = wait_done(mid2, 900, ("transcribed", "done"))
check("th-en: transcribed", st2["status"] in ("transcribed", "done"), st2.get("status"))
check("th-en: switched to Typhoon Whisper Large v3 (logged)", "Large v3 is used instead" in " ".join(st2.get("history", [])), st2.get("history", [])[-3:])
tr2 = c.get(f"/api/meetings/{mid2}").json().get("transcript") or {}; txt2 = " ".join(s["text"] for s in tr2.get("segments", []))
check("th-en transcript: Thai and English both present", thai(txt2) and ("budget" in txt2.lower() or "forecast" in txt2.lower() or "revenue" in txt2.lower()), txt2)
check("th-en: Large v3 engine", "large-v3" in (tr2.get("engine") or ""), tr2.get("engine"))
# 3. Thai: unchanged path
m = asyncio.run(meeting("thai", "after", "th", f"{S}/thai.wav"))
mid3 = next((x["id"] for x in m if x["type"] == "started"), None)
st3 = wait_done(mid3, 900, ("transcribed", "done"))
tr3 = c.get(f"/api/meetings/{mid3}").json().get("transcript") or {}; txt3 = " ".join(s["text"] for s in tr3.get("segments", []))
check("th transcript: Thai, FastConformer", thai(txt3) and "ต้อนรับ" in txt3 and "rnnt" in (tr3.get("engine") or "").lower(), (tr3.get("engine"), txt3))
check("th: language stored on the meeting", c.get(f"/api/meetings/{mid3}").json()["state"].get("language") == "th")
print(f"\n{ok} passed, {fail} failed")
