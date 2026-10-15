"""join.html in headless Chrome with a WAV as the phone's microphone: tap 🎤 -> the sound check runs -> live (speech) or refused (silence)."""
import asyncio, json, subprocess, sys, httpx, websockets
sys.path.insert(0, sys.argv[1]); from hostmeet import Host, B
S, WAV, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = fail = 0
def check(name, cond, info=""):
    global ok, fail
    ok += cond; fail += (not cond); print(("PASS " if cond else "FAIL ") + name, info)
async def main():
    prof = f"{S}/chrome-{PORT}"; subprocess.run(["rm", "-rf", prof])
    chrome = subprocess.Popen(["google-chrome", "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", "--no-first-run", "--ignore-certificate-errors",
                               "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", f"--use-file-for-fake-audio-capture={WAV}", "--autoplay-policy=no-user-gesture-required",
                               "--window-size=420,800", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with Host("phone-ui") as h:
            for _ in range(50):
                try: tgt = [t for t in httpx.get(f"http://127.0.0.1:{PORT}/json").json() if t["type"] == "page"][0]; break
                except Exception: await asyncio.sleep(0.3)
            async with websockets.connect(tgt["webSocketDebuggerUrl"], max_size=2**26) as ws:
                seq = [0]
                async def cdp(method, **params):
                    seq[0] += 1; await ws.send(json.dumps({"id": seq[0], "method": method, "params": params}))
                    while True:
                        m = json.loads(await ws.recv())
                        if m.get("id") == seq[0]: return m.get("result", m)
                async def js(expr, await_=True):
                    r = await cdp("Runtime.evaluate", expression=expr, awaitPromise=await_, returnByValue=True)
                    if "exceptionDetails" in r: raise RuntimeError(json.dumps(r["exceptionDetails"])[:300])
                    return r.get("result", {}).get("value")
                await cdp("Page.enable"); await cdp("Page.navigate", url=f"{B}/join/{h.token}"); await asyncio.sleep(2)
                await js('$("#name").value = "Phone UI"; $("#go").click()', False)
                await js('(async()=>{for(let i=0;i<200&&$("#mic").classList.contains("hide");i++)await new Promise(r=>setTimeout(r,100));})()')
                hid = await js('$("#mic").classList.contains("hide")'); print("   status:", await js('$("#status").textContent'), "| mic hidden:", hid, "| worklet:", await js('!!(navigator.mediaDevices && window.AudioWorkletNode)'))
                check("joined, mic button shown", hid is False)
                await js('window.__snaps=[]; window.__iv=setInterval(()=>__snaps.push($("#micInfo").textContent),150); $("#mic").click();', False)
                await asyncio.sleep(1)
                check("button says checking", "Checking" in await js('$("#mic").textContent'))
                await js('(async()=>{for(let i=0;i<400&&$("#mic").textContent.includes("Checking");i++)await new Promise(r=>setTimeout(r,100));})()')
                await js('clearInterval(window.__iv)', False)
                if "silence" not in WAV: check("'checking the sound' shown after the countdown", any("Checking the sound" in s for s in await js('window.__snaps')))
                snaps = await js('window.__snaps'); info = await js('$("#micInfo").textContent'); btn = await js('$("#mic").textContent'); bad = await js('$("#micInfo").classList.contains("err")')
                check("prompt + level bar shown", any("Say a short sentence" in s and ("█" in s or "silence" in WAV) for s in snaps), f"{len(snaps)} snapshots")
                if "silence" in WAV:
                    check("silent phone refused", "not used yet" in info and "silent" in info and bad, info[:120])
                    check("button back to Use my microphone", "Use my microphone" in btn, btn)
                else:
                    check("sound check passed + heard", "Sound check passed" in info and "heard" in info and not bad, info[:140])
                    check("microphone live", "Stop my microphone" in btn, btn)
                    await asyncio.sleep(2)
                    hg = [m for m in h.msgs if m["type"] == "guests"]
                    check("host sees 🎤 + sound ok", bool(hg) and any(x["mic"] and x.get("sound") == "ok" for x in hg[-1]["guests"]), str(hg[-1]["guests"] if hg else None))
                    lvl = await js('(document.querySelector("#micInfo .lvl i")||{style:{}}).style.width')
                    check("level bar moves while live", lvl not in ("", "0%", "0px", None), str(lvl))
                    await js('stream_ended = (mic.stream.getAudioTracks()[0].onended(), $("#micInfo").textContent)', False); await asyncio.sleep(0.5)
                    info2 = await js('$("#micInfo").textContent'); btn2 = await js('$("#mic").textContent')
                    check("track ended -> off with advice", "disconnected" in info2 and "Use my microphone" in btn2, info2[:80])
            await h.stop()
    finally:
        chrome.terminate()
asyncio.run(main()); print(f"\n{ok} passed, {fail} failed")
