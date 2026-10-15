"""Phone microphone: POST /join/<token>/sound (speech / 429 / silence / bad token), the guest watchdog over /ws/join, the host's panel flags."""
import asyncio, json, sys, time, wave, numpy as np, httpx, websockets
sys.path.insert(0, sys.argv[1]); from hostmeet import Host, B, ctx
S = sys.argv[1]; speech = np.frombuffer(wave.open(S + "/speech.wav").readframes(10**7), np.int16)
ok = fail = 0
def check(name, cond, info=""):
    global ok, fail
    ok += cond; fail += (not cond); print(("PASS " if cond else "FAIL ") + name, info)
async def main():
    async with Host() as h:
        g = httpx.Client(base_url=B, verify=False, timeout=120, headers={"Origin": B}); H = {"Content-Type": "application/octet-stream"}
        r = g.post(f"/join/{h.token}/sound", headers=H, content=speech.tobytes()); j = r.json()
        st = {x["id"]: x for x in j["checks"]}
        check("phone speech: 200 ok + heard", r.status_code == 200 and st["speech"]["status"] == "ok" and "heard" in st["speech"]["detail"], st["speech"]["detail"][:90])
        check("phone advice is short (no engine name)", "Pathumma" not in st["speech"]["detail"] and "transcript after" not in st["speech"]["detail"])
        check("second check within 5 s: 429", g.post(f"/join/{h.token}/sound", headers=H, content=speech.tobytes()).status_code == 429)
        check("bad token: 404", g.post("/join/nope/sound", headers=H, content=speech.tobytes()).status_code == 404)
        await asyncio.sleep(5.2)
        r = g.post(f"/join/{h.token}/sound", headers=H, content=np.zeros(16000 * 4, np.int16).tobytes()); st = {x["id"]: x for x in r.json()["checks"]}
        check("phone silence: fail with phone advice", st["sound"]["status"] == "fail" and "phone" in st["sound"]["detail"], st["sound"]["detail"][:90])
        # the guest socket: sound status to the host, then the watchdog
        got = []
        async with websockets.connect(f"wss://127.0.0.1:8445/ws/join?token={h.token}&name=Phone%20One", ssl=ctx, additional_headers={"Origin": B}) as gw:
            async def rd():
                try:
                    async for m in gw: got.append(json.loads(m))
                except Exception: pass
            rt = asyncio.create_task(rd()); await asyncio.sleep(1)
            check("guest hello with mic allowed", any(m["type"] == "hello" and m.get("mic") for m in got))
            await gw.send(json.dumps({"type": "sound", "status": "ok"})); await asyncio.sleep(1)
            hg = [m for m in h.msgs if m["type"] == "guests"]
            check("host told: phone passed the sound check", hg and any(x.get("sound") == "ok" and x["name"] == "Phone One" for x in hg[-1]["guests"]), str(hg[-1]["guests"] if hg else None))
            async def stream(src, secs):  # 100 ms messages like join.html
                pos = 0
                for _ in range(int(secs * 10)):
                    pk = src[pos:pos + 1600]; pos = (pos + 1600) % max(1600, len(src) - 1600)
                    if len(pk) < 1600: pk = np.zeros(1600, np.int16)
                    await gw.send(pk.tobytes()); await asyncio.sleep(0.1)
            await stream(np.zeros(1600, np.int16), 10)
            n1 = len([m for m in got if m["type"] == "notice"])
            hg = [m for m in h.msgs if m["type"] == "guests"]
            check("phone silent 10 s -> notice to the phone", n1 >= 1 and "No sound from your microphone" in [m for m in got if m["type"] == "notice"][0]["message"], str([m for m in got if m["type"] == "notice"][:1]))
            check("host told: phone silent", hg and any(x.get("silent") and x["name"] == "Phone One" for x in hg[-1]["guests"]), str(hg[-1]["guests"] if hg else None))
            await stream(speech, 4); await asyncio.sleep(0.5)
            hg = [m for m in h.msgs if m["type"] == "guests"]
            check("sound again -> notice cleared", any(m.get("clear") for m in got if m["type"] == "notice"))
            check("host told: phone no longer silent", hg and all(not x.get("silent") for x in hg[-1]["guests"]), str(hg[-1]["guests"] if hg else None))
            rt.cancel()
        hist = json.dumps(h.c.get(f"/api/meetings/{h.mid}").json(), ensure_ascii=False)
        check("event logged", "no sound from Phone One's microphone" in hist)
        await h.stop()
asyncio.run(main()); print(f"\n{ok} passed, {fail} failed")
