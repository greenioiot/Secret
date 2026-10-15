"""Round 55: e-mail recipients per account (To / CC), the CC header, members sending meetings they can read, auto-send to the owner.
Needs a scratch instance on 8445 (see README) whose data folder holds a meeting with a summary (any copied meeting folder works;
the test picks the first "done" meeting). It runs its own plain SMTP server on 127.0.0.1:8025 (security "none", no sign-in).
Run: .venv/bin/python -W ignore tools/tests/test_email_accounts.py <scratch data folder>"""
import asyncio, json, os, sys, threading, time
from email import message_from_bytes
import httpx

S = sys.argv[1]; B = "https://127.0.0.1:8445"; PORT = 8025
ok = fail = 0
def check(name, cond, info=""):
    global ok, fail
    ok += cond; fail += (not cond); print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{str(info)[:300]}]"))

# ---- a tiny SMTP server (plain, enough for smtplib): records (from, rcpt list, raw message)
GOT = []
async def smtp_session(r, w):
    w.write(b"220 fake ESMTP\r\n"); await w.drain(); frm, rcpt, data = "", [], None
    while True:
        line = await r.readline()
        if not line: break
        cmd = line.decode(errors="replace").strip(); u = cmd.upper()
        if u.startswith(("EHLO", "HELO")): w.write(b"250-fake\r\n250 8BITMIME\r\n")
        elif u.startswith("MAIL FROM:"): frm = cmd[10:].strip(); w.write(b"250 ok\r\n")
        elif u.startswith("RCPT TO:"): rcpt.append(cmd[8:].strip().strip("<>")); w.write(b"250 ok\r\n")
        elif u == "DATA":
            w.write(b"354 go\r\n"); await w.drain(); buf = b""
            while True:
                l = await r.readline(); buf += l
                if l == b".\r\n" or not l: break
            GOT.append((frm, rcpt, buf[:-3].replace(b"\r\n..", b"\r\n."))); frm, rcpt = "", []; w.write(b"250 queued\r\n")
        elif u == "QUIT": w.write(b"221 bye\r\n"); await w.drain(); break
        elif u == "RSET": frm, rcpt = "", []; w.write(b"250 ok\r\n")
        else: w.write(b"250 ok\r\n")
        await w.drain()
    w.close()
def serve():
    async def main():
        srv = await asyncio.start_server(smtp_session, "127.0.0.1", PORT)
        async with srv: await srv.serve_forever()
    asyncio.run(main())
threading.Thread(target=serve, daemon=True).start(); time.sleep(0.5)

def client(user, pw):
    c = httpx.Client(base_url=B, verify=False, timeout=60, headers={"Origin": B})
    assert c.post("/api/auth/login", json={"username": user, "password": pw}).status_code == 200, user
    c.headers["X-CSRF-Token"] = c.cookies.get("zecret_csrf"); return c
boss = client("boss", "adminpassword123")
for u, role in (("mem", "member"), ("view", "viewer")):
    r = boss.post("/api/users", json={"username": u, "password": "memberpassword123", "role": role, "name": u})
    assert r.status_code in (200, 400), r.text   # 400 = already there from an earlier run
mem = client("mem", "memberpassword123"); view = client("view", "memberpassword123")
mids = [m["id"] for m in boss.get("/api/meetings").json() if m.get("status") == "done"]
assert mids, "copy a finished meeting folder into the scratch data first"; mid = mids[0]

# 1. e-mail off: /api/me says so for everyone, sending answers 409
boss.put("/api/email", json={"enabled": False})
me = mem.get("/api/me").json()
check("/api/me carries email {enabled,to,cc,shared_to,attach}", set(me.get("email", {})) == {"enabled", "to", "cc", "shared_to", "attach"} and me["email"]["enabled"] is False, me.get("email"))
check("send while e-mail is off -> 409", mem.post(f"/api/meetings/{mid}/email", json={}).status_code == 409)

# 2. per-account recipients
r = mem.post("/api/me/email", json={"to": "a@example.com, b@example.com", "cc": "c@example.com; A@example.com"}).json()
check("member saves own To/CC (dup of a To address dropped from CC)", r.get("to") == ["a@example.com", "b@example.com"] and r.get("cc") == ["c@example.com"], r)
check("bad address -> 400", mem.post("/api/me/email", json={"to": "nope", "cc": ""}).status_code == 400)
check("too many -> 400", mem.post("/api/me/email", json={"to": ", ".join(f"u{i}@example.com" for i in range(21))}).status_code == 400)
check("viewer may keep recipients too", view.post("/api/me/email", json={"to": "v@example.com", "cc": ""}).status_code == 200)
check("boss has none, sees the member's not at all", boss.get("/api/me").json()["email"]["to"] == [])
check("member's prefs persist", mem.get("/api/me").json()["email"] | {} and mem.get("/api/me").json()["email"]["cc"] == ["c@example.com"])
users = boss.get("/api/users").json()
check("/api/users does not leak recipients", "email_to" not in json.dumps(users))

# 3. turn e-mail on (plain local server) with shared defaults
r = boss.put("/api/email", json={"enabled": True, "host": "127.0.0.1", "port": PORT, "security": "none", "username": "", "from_addr": "zecrets@example.com",
                                  "from_name": "Zecrets", "default_to": "shared@example.com", "verify_cert": False, "auto_send": False, "attach_summary": True, "attach_report": False, "attach_transcript": False})
check("e-mail on", r.status_code == 200 and r.json().get("enabled") is True, r.text)
me = boss.get("/api/me").json()["email"]
check("admin with no own To sees the shared default", me["enabled"] and me["to"] == [] and me["shared_to"] == ["shared@example.com"], me)

# 4. member sends with their own To/CC (no fields sent = "my recipients")
GOT.clear(); r = mem.post(f"/api/meetings/{mid}/email", json={"attach": {"summary": True}})
check("member may e-mail a meeting it does not own", r.status_code == 200, r.text)
j = r.json() if r.status_code == 200 else {}
check("answer lists to + cc", j.get("to") == ["a@example.com", "b@example.com"] and j.get("cc") == ["c@example.com"], j)
time.sleep(0.3)
check("one message delivered", len(GOT) == 1, len(GOT))
if GOT:
    frm, rcpt, raw = GOT[0]; m = message_from_bytes(raw)
    check("envelope: To + CC recipients", sorted(rcpt) == ["a@example.com", "b@example.com", "c@example.com"], rcpt)
    check("headers: To and Cc", m["To"] == "a@example.com, b@example.com" and m["Cc"] == "c@example.com", (m["To"], m["Cc"]))
    check("subject [Zecrets]", (m["Subject"] or "").startswith("[Zecrets]"), m["Subject"])
    check("summary attached", any(p.get_filename() == f"{mid}-summary.md" for p in m.walk()), [p.get_filename() for p in m.walk()])
log = (j.get("log") or [])[-1] if j else {}
check("email_log entry has cc + by", log.get("cc") == ["c@example.com"] and log.get("by") == "mem" and log.get("ok") is True, log)

# 5. explicit To/CC in the dialog, save=true updates the account
GOT.clear(); r = mem.post(f"/api/meetings/{mid}/email", json={"to": "x@example.com", "cc": "y@example.com, x@example.com", "save": True, "attach": {}})
check("explicit To/CC sent", r.status_code == 200 and r.json()["to"] == ["x@example.com"] and r.json()["cc"] == ["y@example.com"], r.text)
check("save=true remembered", mem.get("/api/me").json()["email"]["to"] == ["x@example.com"] and mem.get("/api/me").json()["email"]["cc"] == ["y@example.com"])
GOT.clear(); r = mem.post(f"/api/meetings/{mid}/email", json={"to": "z@example.com", "cc": "", "save": False, "attach": {}})
check("save=false leaves the account alone", r.status_code == 200 and mem.get("/api/me").json()["email"]["to"] == ["x@example.com"], r.text)
check("empty To -> 400", mem.post(f"/api/meetings/{mid}/email", json={"to": "", "cc": "q@example.com"}).status_code == 400)
check("viewer cannot send", view.post(f"/api/meetings/{mid}/email", json={}).status_code == 403)

# 6. admin with no own recipients falls back to the shared default
GOT.clear(); r = boss.post(f"/api/meetings/{mid}/email", json={"attach": {}})
check("admin falls back to the shared default recipients", r.status_code == 200 and r.json()["to"] == ["shared@example.com"] and r.json()["cc"] == [], r.text)

# 7. the automatic send picks the owner's recipients (mailer.recipients_for, used by worker.auto_email)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app")); os.environ["ZECRET_DATA"] = S
import mailer
to, cc = mailer.recipients_for("mem"); check("auto: owner's own To/CC", to == ["x@example.com"] and cc == ["y@example.com"], (to, cc))
to, cc = mailer.recipients_for("boss"); check("auto: owner without own -> shared default", to == ["shared@example.com"] and cc == [], (to, cc))
to, cc = mailer.recipients_for(None); check("auto: no owner -> shared default", to == ["shared@example.com"], (to, cc))
boss.put("/api/email", json={"enabled": False})
print(f"\n{ok} passed, {fail} failed"); sys.exit(1 if fail else 0)
