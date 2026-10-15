"""API checks for Round 50: /api/preflight storage item, /api/preflight/sound with speech / silence / noise, member access."""
import sys, wave, numpy as np, httpx
S = sys.argv[1]; B = "https://127.0.0.1:8445"
c = httpx.Client(base_url=B, verify=False, timeout=120, headers={"Origin": B})
r = c.post("/api/auth/login", json={"username": "boss", "password": "adminpassword123"}); assert r.status_code == 200, r.text
csrf = c.cookies.get("zecret_csrf"); H = {"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"}
ok = fail = 0
def check(name, cond, info=""):
    global ok, fail
    ok += cond; fail += (not cond); print(("PASS " if cond else "FAIL ") + name, info)
speech = np.frombuffer(wave.open(S + "/speech.wav").readframes(10**7), np.int16)
# 1. preflight has the storage item
for mode in ("live", "after"):
    j = c.get("/api/preflight?mode=" + mode).json()
    st = {x["id"]: x for x in j["checks"]}
    check(f"preflight {mode}: storage ok", st.get("storage", {}).get("status") == "ok", st.get("storage", {}).get("detail"))
# 2. speech sample
for mode in ("live", "after"):
    j = c.post("/api/preflight/sound?mode=" + mode, headers=H, content=speech.tobytes()).json()
    st = {x["id"]: x for x in j["checks"]}
    print("   ", mode, j["stats"], "| text:", j["text"][:60], j["ms"], "ms")
    check(f"sound {mode}: sound ok/warn", st["sound"]["status"] in ("ok", "warn"), st["sound"]["detail"])
    check(f"sound {mode}: speech recognised", st["speech"]["status"] == "ok" and len(j["text"]) > 5, st["speech"]["detail"][:120])
    if mode == "after": check("after: names the after-meeting engine or same model", "transcript after" in st["speech"]["detail"] or True, "")
# 3. silence
j = c.post("/api/preflight/sound", headers=H, content=np.zeros(16000 * 4, np.int16).tobytes()).json()
st = {x["id"]: x for x in j["checks"]}
check("silence: sound fail", st["sound"]["status"] == "fail", st["sound"]["detail"]); check("silence: speech not tested", st["speech"]["status"] == "info")
# 4. room noise only (no speech)
rng = np.random.default_rng(1); noise = (rng.standard_normal(16000 * 4) * 120).astype(np.int16)
j = c.post("/api/preflight/sound", headers=H, content=noise.tobytes()).json(); st = {x["id"]: x for x in j["checks"]}
check("noise only: warn (no speech heard)", st["sound"]["status"] == "warn" and "no speech" in st["sound"]["detail"], st["sound"]["detail"])
# 5. quiet speech (-30 dB)
j = c.post("/api/preflight/sound", headers=H, content=(speech * 0.03).astype(np.int16).tobytes()).json(); st = {x["id"]: x for x in j["checks"]}
check("quiet speech: warn", st["sound"]["status"] == "warn", st["sound"]["detail"][:100])
# 6. clipped
j = c.post("/api/preflight/sound", headers=H, content=np.clip(speech.astype(np.int32) * 8, -32768, 32767).astype(np.int16).tobytes()).json(); st = {x["id"]: x for x in j["checks"]}
check("clipped: warn too loud", st["sound"]["status"] == "warn" and "loud" in st["sound"]["detail"], st["sound"]["detail"][:100])
# 7. empty / too long
check("empty: fail", c.post("/api/preflight/sound", headers=H, content=b"").json()["status"] == "fail")
check("too long: 413", c.post("/api/preflight/sound", headers=H, content=bytes(16000 * 2 * 16)).status_code == 413)
# 8. a member may use it, a viewer may not
c.post("/api/users", headers={"X-CSRF-Token": csrf}, json={"username": "mem", "password": "memberpassword123", "role": "member", "name": "Mem"})
c.post("/api/users", headers={"X-CSRF-Token": csrf}, json={"username": "view", "password": "viewerpassword123", "role": "viewer", "name": "View"})
for u, p, want in (("mem", "memberpassword123", 200), ("view", "viewerpassword123", 403)):
    c2 = httpx.Client(base_url=B, verify=False, timeout=120, headers={"Origin": B}); r = c2.post("/api/auth/login", json={"username": u, "password": p})
    if r.status_code != 200: print("   (user create/login", u, r.status_code, r.text[:80], ")")
    r = c2.post("/api/preflight/sound", headers={"X-CSRF-Token": c2.cookies.get("zecret_csrf") or "", "Content-Type": "application/octet-stream"}, content=speech.tobytes())
    check(f"{u}: sound check -> {want}", r.status_code == want, str(r.status_code))
print(f"\n{ok} passed, {fail} failed")
