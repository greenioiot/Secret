"""Headless Chrome (fake microphone from a WAV) through the real UI: the sound check in 'Check system', Start with the check, the meter in
Record-only mode, the microphone-replaced path, the watchdog banner."""
import asyncio, json, subprocess, sys, time, httpx, websockets
S = sys.argv[1]; WAV = sys.argv[2]; PORT = int(sys.argv[3]); B = "https://127.0.0.1:8445"
prof = f"{S}/chrome-{PORT}"; subprocess.run(["rm", "-rf", prof])
chrome = subprocess.Popen(["google-chrome", "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", "--no-first-run", "--ignore-certificate-errors",
                           "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", f"--use-file-for-fake-audio-capture={WAV}", "--autoplay-policy=no-user-gesture-required",
                           "--window-size=1200,900", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ok = fail = 0
def check(name, cond, info=""):
    global ok, fail
    ok += cond; fail += (not cond); print(("PASS " if cond else "FAIL ") + name, info)
async def main():
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
        async def nav(url):
            await cdp("Page.enable"); await cdp("Page.navigate", url=url); await asyncio.sleep(1.5)
        await nav(B + "/login")
        r = await js(f'fetch("/api/auth/login",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{username:"boss",password:"adminpassword123"}})}}).then(r=>r.status)')
        check("login", r == 200, str(r))
        await nav(B + "/"); await asyncio.sleep(2)
        await js('(async()=>{for(let i=0;i<50&&!Object.keys(settings).length;i++)await new Promise(r=>setTimeout(r,100));})()')
        if "silence" in WAV:   # only the watchdog banner (a silent microphone cannot pass the check, so the recording is started directly)
            await js('setMode("after"); $("#title").value="ui50-silent"; startRec();', False)
            await js('(async()=>{for(let i=0;i<100&&!meetingId;i++)await new Promise(r=>setTimeout(r,100));})()')
            await asyncio.sleep(10)
            w = await js('({t:$("#recWarn").textContent, hidden:$("#recWarn").classList.contains("hide")})')
            check("watchdog banner shown for a silent microphone", "No sound" in w["t"] and not w["hidden"], w["t"][:100])
            await js('$("#recBtn").click()', False); await js('(async()=>{for(let i=0;i<200&&recording;i++)await new Promise(r=>setTimeout(r,100));})()')
            return w
        # 1. Check system: the sound check asks for a sentence, then reports what it heard
        await js('window.__snaps=[]; window.__pf=preflight(false); window.__iv=setInterval(()=>__snaps.push($("#preflight").textContent),150);', False)
        await js('(async()=>{await window.__pf; clearInterval(window.__iv);})()')
        snaps = await js('window.__snaps'); txt = await js('$("#preflight").textContent')
        check("prompt shown while listening", any("Say a short sentence" in s for s in snaps), f"{len(snaps)} snapshots")
        check("level bar moved", any("█" in s for s in snaps))
        check("sound check item ok", "Sound check" in txt and ("above the noise" in txt), txt[:400].replace("\n", " "))
        check("speech recognised with text", "heard: “" in txt, "")
        check("storage item present", "Recording storage" in txt and "writable" in txt)
        check("head says ready", "Everything is ready" in txt or "Ready, with warnings" in txt, txt[:80])
        # 2. Start in Record-only mode: check first, then recording; meter moves; status shows what was heard
        await js('setMode("after"); $("#title").value="ui50"; window.__pf2=preflight(true);', False)
        await js('(async()=>{await window.__pf2;})()')
        full = await js('[...document.querySelectorAll("#preflight .ck")].map(c=>c.textContent).join(" | ")'); print("   items:", full[:600])
        warn = await js('[...document.querySelectorAll("#preflight .warn b")].map(b=>b.textContent)'); print("   warnings:", warn)
        clicked = await js('(()=>{const b=[...document.querySelectorAll("#preflight button")].find(b=>b.textContent.startsWith("Start anyway")); if(b){b.click();return true} return false})()', False)
        print("   Start anyway clicked:", clicked)
        await js('(async()=>{for(let i=0;i<100&&!meetingId;i++)await new Promise(r=>setTimeout(r,100));})()')
        await asyncio.sleep(2.5)
        rec = await js('({rec:recording, mid:meetingId, meter:$("#meter").style.width, msg:$("#recMsg").textContent, warn:$("#recWarn").classList.contains("hide"), onended: !!(stream&&stream.getAudioTracks()[0].onended)})')
        check("recording started after the check", rec["rec"] and bool(rec["mid"]), str(rec["mid"]))
        check("meter moves in Record-only mode", rec["meter"] not in ("", "0", "0%", "0px"), rec["meter"])
        check("status shows the heard text", "heard" in rec["msg"], rec["msg"][:120])
        check("no warning while sound flows", rec["warn"])
        check("microphone track is watched", rec["onended"])
        # 3. the microphone disappears: another one takes over, the recording goes on
        await js('stream.getAudioTracks()[0].onended()'); await asyncio.sleep(2)
        w = await js('({t:$("#recWarn").textContent, rec:recording, live: stream.getAudioTracks()[0].readyState})')
        check("microphone replaced, warning shown", "Microphone changed" in w["t"] and w["rec"] and w["live"] == "live", w["t"][:100])
        # 4. stop
        await js('$("#recBtn").click()', False)
        await js('(async()=>{for(let i=0;i<200&&recording;i++)await new Promise(r=>setTimeout(r,100));})()')
        st = await js('({rec:recording, msg:$("#recMsg").textContent, warn:$("#recWarn").classList.contains("hide")})')
        check("stopped and saved", not st["rec"] and "Saved" in st["msg"], st["msg"][:80]); check("warning cleared on stop", st["warn"])
        return None
    return None
try:
    w = asyncio.run(main())
finally:
    chrome.terminate()
print(json.dumps({"ok": ok, "fail": fail, "banner": w}))
