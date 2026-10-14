#!/usr/bin/env python3
"""Measure what a live meeting and each joined guest cost on THIS device (RAM, CPU), using a throwaway server on port 8446.
Usage: .venv/bin/python tools/bench_capacity.py --wav some_16k_mono_speech.wav [--steps 0,40,100]
The numbers go into app/capacity.py (COST) and decide how many people the box lets in."""
import argparse, asyncio, json, os, shutil, ssl, subprocess, sys, tempfile, time, urllib.request, wave
import numpy as np, websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--wav", required=True); ap.add_argument("--steps", default="0,40,100"); ap.add_argument("--window", type=float, default=15.0)
args = ap.parse_args()
steps = [int(x) for x in args.steps.split(",")]
raw = np.frombuffer(wave.open(args.wav).readframes(10**9), np.int16).tobytes()
frames = [raw[i:i + 640] for i in range(0, len(raw), 640)]
tmp = tempfile.mkdtemp(prefix="secrets-bench-"); PORT = 8446; TOK = "benchtoken123"
os.makedirs(tmp + "/meetings"); open(tmp + "/access.token", "w").write(TOK)
json.dump({"allow_join": True, "max_guests": 500, "reserve_ram_gb": 0.5, "cpu_limit_pct": 100}, open(tmp + "/settings.json", "w"))
env = {**os.environ, "SECRET_DATA": tmp, "SECRET_PORT": str(PORT), "SECRET_JOIN_PER_IP": "100000", "SECRET_BENCH_TOKEN": TOK}  # all simulated guests share one address
srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "app/server.py")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def http(method, path, body=None):
    r = urllib.request.Request(f"https://127.0.0.1:{PORT}{path}", method=method, data=json.dumps(body).encode() if body is not None else None, headers={"X-Secret-Token": TOK, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, context=ctx, timeout=30).read())
def rss_mb(pid): return next(int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS:")) / 1024
def cpu_s(pid):
    f = open(f"/proc/{pid}/stat").read(); v = f[f.rindex(")") + 2:].split(); return (int(v[11]) + int(v[12])) / os.sysconf("SC_CLK_TCK")
async def main():
    for _ in range(60):
        try: http("GET", "/api/status"); break
        except Exception: await asyncio.sleep(1)
    cores = os.cpu_count() or 8; results = []
    host = await websockets.connect(f"wss://127.0.0.1:{PORT}/ws/record?mode=live&title=bench", ssl=ctx, additional_headers={"X-Secret-Token": TOK}, max_size=None)
    msgs = []
    async def reader():
        async for m in host: msgs.append(json.loads(m))
    asyncio.create_task(reader())
    async def stream():
        t0, i = time.time(), 0
        while True:
            await host.send(frames[i % len(frames)]); i += 1
            await asyncio.sleep(max(0, t0 + i * 0.02 - time.time()))
    for _ in range(50):
        if any(m["type"] == "started" for m in msgs): break
        await asyncio.sleep(0.2)
    mid = next(m["id"] for m in msgs if m["type"] == "started"); st = asyncio.create_task(stream())
    url = http("POST", f"/api/meetings/{mid}/join", {"open": True})["urls"][0]; token = url.rsplit("/", 1)[1]
    guests, got, live = [], [0], [0]
    async def guest(i):
        try:
            async with websockets.connect(f"wss://127.0.0.1:{PORT}/ws/join?token={token}&name=G{i}", ssl=ctx, max_size=None) as ws:
                async for m in ws:
                    j = json.loads(m)
                    if j["type"] == "hello": live[0] += 1
                    elif j["type"] == "error": return
                    elif j["type"] == "caption": got[0] += 1
        except Exception: pass
    await asyncio.sleep(12)  # warm-up: models load, captions start
    n = 0
    for target in steps:
        while n < target:
            guests.append(asyncio.create_task(guest(n))); n += 1
            if n % 20 == 0: await asyncio.sleep(0.5)
        await asyncio.sleep(6)
        r0, c0, t0 = rss_mb(srv.pid), cpu_s(srv.pid), time.time(); await asyncio.sleep(args.window); r1, c1, t1 = rss_mb(srv.pid), cpu_s(srv.pid), time.time()
        results.append({"guests": n, "rss_mb": round(r1, 1), "cpu_pct_of_all_cores": round(100 * (c1 - c0) / (t1 - t0) / cores, 2), "cpu_cores": round((c1 - c0) / (t1 - t0), 3)})
        results[-1]["connected"] = live[0]; print(results[-1], flush=True)
    base = results[0]; last = results[-1]
    out = {"cores": cores, "steps": results, "guest_mb": round((last["rss_mb"] - base["rss_mb"]) / max(1, last["guests"] - base["guests"]), 3) if last["guests"] > base["guests"] else None,
           "guest_cpu_pct": round(max(0.0, last["cpu_pct_of_all_cores"] - base["cpu_pct_of_all_cores"]) / max(1, last["guests"] - base["guests"]), 4) if last["guests"] > base["guests"] else None,
           "live_cpu_pct": base["cpu_pct_of_all_cores"], "captions_delivered": got[0]}
    print("RESULT", json.dumps(out)); st.cancel(); await host.send(json.dumps({"type": "stop"})); await asyncio.sleep(2)
try: asyncio.run(main())
finally:
    srv.terminate()
    try: srv.wait(15)
    except Exception: srv.kill()
    shutil.rmtree(tmp, ignore_errors=True)
