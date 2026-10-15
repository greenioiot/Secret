"""Start asks for the mode of this meeting (live captions / record only) before the system check; Cancel; Check system unchanged."""
import asyncio, base64, json, subprocess, sys, httpx, websockets
S, WAV, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3]); B = "https://127.0.0.1:8445"
ok = fail = 0
def check(name, cond, info=""):
    global ok, fail
    ok += bool(cond); fail += (not cond); print(("PASS " if cond else "FAIL ") + name, info)
async def main():
    prof = f"{S}/chrome-{PORT}"; subprocess.run(["rm", "-rf", prof])
    chrome = subprocess.Popen(["google-chrome", "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", "--no-first-run", "--ignore-certificate-errors",
                               "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", f"--use-file-for-fake-audio-capture={WAV}", "--window-size=1200,1000", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
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
            async def shot(name, w):
                await cdp("Emulation.setDeviceMetricsOverride", width=w, height=1000, deviceScaleFactor=1, mobile=w < 500); await asyncio.sleep(0.4)
                open(f"{S}/{name}.png", "wb").write(base64.b64decode((await cdp("Page.captureScreenshot", format="png"))["data"]))
            await cdp("Page.enable"); await cdp("Page.navigate", url=B + "/login"); await asyncio.sleep(1.5)
            await js('fetch("/api/auth/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:"boss",password:"adminpassword123"})}).then(r=>r.status)')
            await cdp("Page.navigate", url=B + "/"); await asyncio.sleep(3)
            await js('(async()=>{for(let i=0;i<50&&!Object.keys(settings).length;i++)await new Promise(r=>setTimeout(r,100));})()')
            # 1. Start -> the choice
            await js('setMode("live"); $("#recBtn").click()', False); await asyncio.sleep(1.5)
            txt = await js('$("#preflight").textContent'); n = await js('document.querySelectorAll("#preflight .opt").length'); cur = await js('document.querySelector("#preflight .opt.cur b").textContent')
            check("Start shows the mode choice", "How should this meeting run" in txt and n == 2, f"{n} options")
            check("current default highlighted (live)", cur.startswith("● Live captions"), cur)
            check("model names shown", "Live model:" in txt and "Transcript afterwards:" in txt, txt[:300])
            check("not recording yet", not await js('recording'))
            await js('$("#preflight").scrollIntoView()', False); await shot("mode_1200", 1200); await shot("mode_390", 390); await cdp("Emulation.clearDeviceMetricsOverride")
            # 2. Cancel closes it, nothing starts
            await js('[...document.querySelectorAll("#preflight button")].find(b=>b.textContent==="Cancel").click()', False); await asyncio.sleep(0.5)
            check("Cancel: hidden, not recording", await js('$("#preflight").classList.contains("hide") && !recording'))
            # 3. choose Record only -> the check runs (sentence prompt) -> the meeting starts in Record-only mode
            await js('$("#recBtn").click()', False); await asyncio.sleep(1.2)
            await js('window.__snaps=[]; window.__iv=setInterval(()=>__snaps.push($("#preflight").textContent),150); document.querySelectorAll("#preflight .opt")[1].click()', False)
            await js('(async()=>{for(let i=0;i<400&&!meetingId;i++)await new Promise(r=>setTimeout(r,100));})()'); await js('clearInterval(window.__iv)', False)
            snaps = await js('window.__snaps')
            if not await js('meetingId'):  # warnings on the scratch box: press Start anyway
                await js('(()=>{const b=[...document.querySelectorAll("#preflight button")].find(b=>b.textContent.startsWith("Start anyway")); b&&b.click()})()', False)
                await js('(async()=>{for(let i=0;i<100&&!meetingId;i++)await new Promise(r=>setTimeout(r,100));})()')
            check("check ran after the choice", any("Say a short sentence" in s for s in snaps))
            check("switch follows the choice", await js('mode') == "after" and await js('$("#modeSeg button.on").dataset.m') == "after")
            check("recording started", await js('recording && !!meetingId'))
            await asyncio.sleep(1)
            cap_hidden = await js('$("#capCard").classList.contains("hide")'); strip = await js('$("#engStrip").textContent')
            check("meeting is Record only (captions card hidden, strip says off)", cap_hidden and "off (Record only)" in strip, strip[:80])
            await js('$("#recBtn").click()', False); await js('(async()=>{for(let i=0;i<200&&recording;i++)await new Promise(r=>setTimeout(r,100));})()')
            check("stopped", not await js('recording'))
            # 4. Check system: no choice, straight to the check
            await js('$("#pfBtn").click()', False); await asyncio.sleep(1.0)
            txt = await js('$("#preflight").textContent')
            check("Check system goes straight to the check", "System check" in txt and "How should" not in txt, txt[:60])
            await js('(async()=>{for(let i=0;i<400&&pfRunning;i++)await new Promise(r=>setTimeout(r,100));})()')
            check("admin default saved as after", await js('api("/settings").then(s=>s.mode)') == "after")
    finally:
        chrome.terminate()
asyncio.run(main()); print(f"\n{ok} passed, {fail} failed")
