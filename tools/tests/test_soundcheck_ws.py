"""Watchdog test over the recording WebSocket: silence -> notice, sound -> cleared; level messages carry a number in Record-only mode."""
import asyncio, json, ssl, sys, time, wave, numpy as np, httpx, websockets
S = sys.argv[1]; B = "https://127.0.0.1:8445"
c = httpx.Client(base_url=B, verify=False, timeout=60, headers={"Origin": B})
assert c.post("/api/auth/login", json={"username": "boss", "password": "adminpassword123"}).status_code == 200
cookie = "; ".join(f"{k}={v}" for k, v in c.cookies.items())
speech = np.frombuffer(wave.open(S + "/speech.wav").readframes(10**7), np.int16)
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
ok = fail = 0
def check(name, cond, info=""):
    global ok, fail
    ok += cond; fail += (not cond); print(("PASS " if cond else "FAIL ") + name, info)
async def main():
    got = {"notice": [], "levels": [], "stopped": None, "started": None}
    async with websockets.connect("wss://127.0.0.1:8445/ws/record?mode=after&title=watchdog", ssl=ctx, additional_headers={"Cookie": cookie, "Origin": B}) as ws:
        async def reader():
            try:
                async for m in ws:
                    m = json.loads(m)
                    if m["type"] == "notice": got["notice"].append((round(time.time() - t0, 1), m))
                    elif m["type"] == "level": got["levels"].append(m["db"])
                    elif m["type"] == "started": got["started"] = m
                    elif m["type"] == "stopped": got["stopped"] = m; return
            except Exception: pass
        t0 = time.time(); rt = asyncio.create_task(reader())
        async def stream(src, secs):  # 20 ms packets in real time
            pos = 0
            for i in range(int(secs * 50)):
                pk = src[pos:pos + 320]; pos = (pos + 320) % max(320, len(src) - 320)
                if len(pk) < 320: pk = np.zeros(320, np.int16)
                await ws.send(pk.tobytes()); await asyncio.sleep(0.02)
        await stream(np.zeros(320, np.int16), 10)       # 10 s of digital silence (SECRET_SILENT_S=3 on the scratch instance)
        n1 = len(got["notice"])
        await stream(speech, 6)                          # then speech
        await asyncio.sleep(0.5)
        check("started", got["started"] is not None)
        check("Record only: level messages carry a number", bool(got["levels"]) and all(isinstance(v, (int, float)) for v in got["levels"]), f"{len(got['levels'])} levels, last {got['levels'][-3:]}")
        check("silence -> no-sound notice within 10 s", n1 >= 1 and "No sound" in got["notice"][0][1].get("message", ""), str(got["notice"][:1]))
        check("sound again -> notice cleared", any(m.get("clear") for _, m in got["notice"][n1:]), str(got["notice"][n1:]))
        check("levels: silence low, speech high", min(got["levels"]) < -60 and max(got["levels"]) > -30, f"min {min(got['levels'])} max {max(got['levels'])}")
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(rt, 30)
        check("stopped cleanly", got["stopped"] is not None, str(got["stopped"]))
    mid = got["started"]["id"]; ev = c.get(f"/api/meetings/{mid}").json()
    hist = json.dumps(ev.get("history") or ev.get("events") or ev, ensure_ascii=False)
    check("event logged in the meeting", "no sound from the microphone" in hist, "")
asyncio.run(main()); print(f"\n{ok} passed, {fail} failed")
