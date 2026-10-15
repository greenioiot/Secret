"""Secrets: on-device Thai meeting recorder. Browser mic -> WebSocket -> crash-safe disk -> (live captions) ->
worker process (accurate Thai ASR, summary). Everything stays on this machine."""
from __future__ import annotations

import asyncio
import hmac
import io
import itertools
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import asr
import auth
import bot as botmod
import guide
import handsfree
import whois
import tts
import capacity
import engines
import config
import kb
import learn
import mailer
import mics
import mindex
import vocab
import voicelib
import wctc
import report
import separation
import speakers
import stats
import store

log = logging.getLogger("secret")
HERE = Path(__file__).resolve().parent
app = FastAPI(title="Secrets")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

PARTIAL_POOL = ThreadPoolExecutor(1)  # interim captions never wait behind final captions / speaker labels
PARTIAL_EVERY_S = 1.0
GUEST_CAN_ASK = False  # participants get the transcript view only
CAPTION_POOL = ThreadPoolExecutor(1)  # one Typhoon call at a time, in order
_emb: speakers.Embedder | None = None
_ctc: asr.ThaiCTC | None = None
_ctc_lock = threading.Lock()
active: dict[str, dict] = {}          # meeting id -> {"started": ts}
worker_proc: subprocess.Popen | None = None


_ctc_used = 0.0
_summary_lock = threading.Lock()


def ctc() -> asr.ThaiCTC:
    global _ctc, _ctc_used
    with _ctc_lock:
        _ctc_used = time.time()
        if _ctc is None:
            st_ = store.settings()
            if st_.get("live_engine") == "pathumma-rnnt":
                import rnnt

                if rnnt.available():
                    try:
                        _ctc = rnnt.PathummaLive(threads=4)
                    except Exception as e:  # noqa: BLE001
                        log.warning("Pathumma FastConformer could not start (%s): live captions use the Typhoon realtime model", str(e)[:120])
            elif st_.get("live_engine") == "wctc" and wctc.available():  # Whisper-medium CTC on the NPU (falls back to the Typhoon realtime model)
                try:
                    _ctc = wctc.WhisperCTC()
                except Exception as e:  # noqa: BLE001
                    log.warning("Whisper CTC could not load (%s): live captions use the Typhoon realtime model", str(e)[:120])
            if _ctc is None:
                _ctc = asr.ThaiCTC(threads=2, npu=bool(st_.get("asr_npu", True)))
        return _ctc


PARTIAL_GAP = {"wctc": 2.0, "pathumma-rnnt": 3.0}  # seconds between interim texts (Pathumma runs on the CPU: about 1 s per call)
SILENT_WARN_S = int(os.environ.get("SECRET_SILENT_S", "60"))  # recording: no sound for this long -> the recorder is warned (muted / unplugged microphone)
SOUND_DB = -50.0  # a 20 ms packet whose peak is above this counts as sound (room noise through an open microphone is around -45; a muted one -90 or less)


def peak_db(data: bytes) -> float:
    """Peak level of a packet of 16-bit audio in dB (0 = full scale)."""
    x = np.frombuffer(data[: len(data) & ~1], np.int16)
    return 20 * np.log10(int(np.abs(x).max()) / 32768) if len(x) and np.abs(x).max() else -120.0


def sound_stats(pcm: np.ndarray) -> dict:
    """What the sound check measures on a few seconds from the microphone (30 ms frames): the loud and the quiet parts, how far apart they
    are (signal to noise), how much of it is speech by the live segmenter's own rule (a frame 3x the noise floor, at least 220 RMS), clipping."""
    fr = pcm[: len(pcm) // 480 * 480].astype(np.float32).reshape(-1, 480)
    if not len(fr):
        return {"seconds": 0.0, "peak_db": -120.0, "noise_db": -120.0, "speech_db": -120.0, "snr_db": 0.0, "speech_s": 0.0, "clip_pct": 0.0}
    rms = np.sqrt((fr ** 2).mean(1)) + 1e-3
    noise, loud = float(np.percentile(rms, 10)), float(np.percentile(rms, 90))
    db = lambda v: round(20 * float(np.log10(v / 32768)), 1)  # noqa: E731
    return {"seconds": round(len(pcm) / config.SR, 1), "peak_db": db(float(np.abs(fr).max()) + 1e-3), "noise_db": db(noise), "speech_db": db(loud),
            "snr_db": round(20 * float(np.log10(loud / noise)), 1), "speech_s": round(float((rms > max(220.0, noise * 3.0)).sum()) * 0.03, 1),
            "clip_pct": round(100 * float((np.abs(fr) >= 32000).mean()), 2)}


def _drop_ctc():
    """Forget the live model (call with _ctc_lock held); a model in a child process (Pathumma) is stopped."""
    global _ctc
    old, _ctc = _ctc, None
    if old is not None and hasattr(old, "close"):
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass


def release_idle_ctc():
    """No recording for 90 s: give the NPU speech model back (it reloads in 0.3 s thanks to the graph cache), so the NPU memory is free
    for the summary model while meetings are being processed."""
    global _ctc
    while True:
        time.sleep(15)
        with _ctc_lock:
            # Pathumma FastConformer (CPU, 2.1 GB) stays loaded: it needs no NPU memory and takes 15 s to start (the check before a meeting is instant)
            if _ctc is not None and not active and time.time() - _ctc_used > 90 and getattr(_ctc, "kind", "") != "pathumma-rnnt":
                _drop_ctc()
                import gc

                gc.collect()
                log.info("idle: NPU speech model released")


def emb() -> speakers.Embedder:
    global _emb
    with _ctc_lock:
        if _emb is None:
            _emb = speakers.Embedder(threads=2)
        return _emb


# ---------------------------------------------------------------- accounts, sessions, roles (app/auth.py)
def token() -> str:
    """The old shared access code. With accounts it is only the one-time SETUP CODE that proves the first admin is the box's owner."""
    store.init()
    if not config.TOKEN_FILE.exists():
        config.TOKEN_FILE.write_text(secrets.token_urlsafe(9))
        os.chmod(config.TOKEN_FILE, 0o600)
    return config.TOKEN_FILE.read_text().strip()


SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
PUBLIC_PATHS = {"/login", "/sso", "/static/fonts.css", "/static/theme.css", "/static/pcm-worklet.js", "/static/login.html", "/api/auth/state", "/api/auth/login", "/api/auth/setup", "/api/auth/logout"}
ADMIN_GET = ("/api/whois", "/api/email", "/api/users", "/api/enroll", "/guide", "/api/guide", "/api/meetings-unowned", "/api/voices/drafts", "/api/tts")
ADMIN_WRITE = ("/api/queue/", "/api/whois", "/api/settings", "/api/vocab-suggestions", "/api/vocab", "/api/words", "/api/voices", "/api/enroll", "/api/email", "/api/bot/", "/api/kb", "/api/users", "/api/meetings/assign", "/api/tts")
MEMBER_WRITE = ("/api/meetings/plan", "/api/table/delete", "/api/kb/search", "/api/me/", "/api/preflight")
MEETING_RE = re.compile(r"^/api/meetings/([0-9]{8}-[0-9]{6}(?:-[0-9]+)?)(/.*)?$")
BENCH_TOKEN = os.environ.get("SECRET_BENCH_TOKEN", "")  # tools/bench_capacity.py on its own throwaway instance only: header X-Secret-Token = admin


def is_loopback(host: str | None) -> bool:
    return host in ("127.0.0.1", "::1", "localhost") or str(host or "").startswith("127.")


def same_origin(headers, url) -> bool:
    """Blocks cross-site form posts / WebSocket hijacking: a browser always sends Origin on these; it must be this server."""
    o = headers.get("origin")
    if not o:
        return True  # not a browser cross-site request (curl, the benchmark)
    from urllib.parse import urlsplit

    return urlsplit(o).netloc.lower() == (headers.get("host") or url.netloc).lower()


def user_of(cookies, headers) -> dict | None:
    if BENCH_TOKEN and hmac.compare_digest(headers.get("x-secret-token") or "", BENCH_TOKEN):
        return {"user": "bench", "name": "bench", "role": "admin", "csrf": "", "bench": True}
    return auth.session(cookies.get(auth.COOKIE))


def write_problem(u: dict, path: str) -> str | None:
    """Why this signed-in user may not make this change (None = allowed). Meeting changes: owner only."""
    if path.startswith("/api/me/"):
        return None
    if u["role"] == "viewer":
        return "your account is read-only (Viewer)"
    if path.startswith(ADMIN_WRITE):
        return None if u["role"] == "admin" else "only an administrator can change this"
    if path.startswith(MEMBER_WRITE):
        return None
    m = MEETING_RE.match(path)
    if m:
        if (m.group(2) or "") == "/owner":
            return None if u["role"] == "admin" else "only an administrator can assign an owner"
        try:
            st = store.load_state(m.group(1))
        except ValueError:
            st = None
        return auth.edit_problem(u, st) if st else None  # unknown meeting: the endpoint answers 404
    return None if u["role"] == "admin" else "only an administrator can do this"


@app.middleware("http")
async def require_auth(request: Request, call_next):
    p, method = request.url.path, request.method
    if method not in SAFE_METHODS and not same_origin(request.headers, request.url):
        return JSONResponse({"detail": "request from another site refused"}, status_code=403)
    if p in PUBLIC_PATHS or p.startswith(("/static/fonts/", "/join/", "/enroll/")):
        return await call_next(request)
    u = user_of(request.cookies, request.headers)
    if not u:
        if p.startswith("/api/"):
            return JSONResponse({"error": "not signed in", "detail": "not signed in"}, status_code=401)
        return RedirectResponse("/login")
    if method in SAFE_METHODS:
        if (p.startswith(ADMIN_GET) and not p.startswith("/api/enroll-qr")) and u["role"] != "admin":
            return JSONResponse({"detail": "only an administrator can open this"}, status_code=403)
    else:
        if not u.get("bench") and not hmac.compare_digest(request.headers.get("x-csrf-token") or "", u["csrf"]):
            return JSONResponse({"detail": "security check failed: reload the page and try again"}, status_code=403)
        why = write_problem(u, p)
        if why:
            return JSONResponse({"detail": why}, status_code=403)
    request.state.user = u
    return await call_next(request)


def who(request: Request) -> dict:
    return getattr(request.state, "user", None) or {"user": None, "name": "?", "role": "viewer"}


async def ws_user(ws: WebSocket, roles=("admin", "member")) -> dict | None:
    """Signed-in user of a WebSocket with one of these roles, else the socket is closed (4401 not signed in, 4403 not allowed)."""
    if not same_origin(ws.headers, ws.url):
        await ws.close(code=4403)
        return None
    u = user_of(ws.cookies, ws.headers)
    if not u:
        await ws.close(code=4401)
        return None
    if u["role"] not in roles:
        await ws.close(code=4403)
        return None
    return u


def _set_session_cookies(r, request: Request, sid: str, csrf: str):
    https = request.url.scheme == "https"
    r.set_cookie(auth.COOKIE, sid, max_age=auth.SESSION_S, httponly=True, secure=https, samesite="strict")
    r.set_cookie(auth.CSRF_COOKIE, csrf, max_age=auth.SESSION_S, httponly=False, secure=https, samesite="strict")


def _login_allowed(request: Request) -> str | None:
    if request.url.scheme != "https" and not is_loopback(request.client.host if request.client else None):
        return "sign-in from another device needs HTTPS: open https://… instead of http://"
    return None


@app.get("/login")
def login_page():
    return FileResponse(HERE / "static/login.html", headers={"Cache-Control": "no-store"})


@app.get("/sso")
def sso_page():
    return HTMLResponse("""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Company sign-in · Secrets</title>
<link rel=stylesheet href=/static/fonts.css><body style="font:16px/1.6 Inter,'Noto Sans Thai',system-ui;margin:0;display:grid;place-items:center;min-height:100vh;background:#f5f6f8;color:#1c2430">
<div style="max-width:460px;padding:28px;background:#fff;border:1px solid #e3e6eb;border-radius:14px"><h2 style=margin-top:0>Company single sign-on (SSO)</h2>
<p>Enterprise sign-in (SAML 2.0 / OpenID Connect, e.g. Microsoft Entra ID, Google Workspace, Okta) is <b>not connected yet</b>. This page is a placeholder for it.</p>
<p>เข้าสู่ระบบด้วยบัญชีองค์กร (SSO) ยังไม่เปิดใช้งาน กรุณาใช้บัญชีภายในเครื่อง</p><p><a href=/login>← Back to sign-in</a></p></div>""")


@app.get("/api/auth/state")
def api_auth_state(request: Request):
    u = user_of(request.cookies, request.headers)
    return {"setup_needed": not auth.has_users(), "signed_in": bool(u), "min_password": auth.MIN_PASSWORD, "https": request.url.scheme == "https",
            "remote_needs_https": bool(_login_allowed(request)), "sso": {"configured": False}}


@app.post("/api/auth/login")
async def api_login(request: Request):
    why = _login_allowed(request)
    if why:
        raise HTTPException(403, why)
    b = await request.json()
    ip = request.client.host if request.client else "?"
    uname = str(b.get("username", "")).strip().lower()[:64]
    wait = auth.throttled(ip, uname)
    if wait:
        raise HTTPException(429, f"too many failed attempts: try again in {max(1, wait // 60)} min")
    name = await asyncio.to_thread(auth.authenticate, uname, str(b.get("password", ""))[:256])
    if not name:
        auth.note_failure(ip, uname)
        await asyncio.sleep(1.0)
        raise HTTPException(401, "wrong user name or password (or the account is disabled)")
    sid, csrf = auth.new_session(name, ip)
    log.warning("sign-in: %s from %s", name, ip)
    r = JSONResponse({"ok": True})
    _set_session_cookies(r, request, sid, csrf)
    return r


@app.post("/api/auth/setup")
async def api_setup(request: Request):
    """First administrator. Needs the setup code (data/access.token, printed by the server at start) so nobody else on the network can claim the box."""
    why = _login_allowed(request)
    if why:
        raise HTTPException(403, why)
    if auth.has_users():
        raise HTTPException(409, "the first administrator already exists: sign in")
    b = await request.json()
    ip = request.client.host if request.client else "?"
    if auth.throttled(ip, "#setup"):
        raise HTTPException(429, "too many wrong setup codes: try again later")
    if not hmac.compare_digest(str(b.get("code", "")).strip(), token()):
        auth.note_failure(ip, "#setup")
        await asyncio.sleep(1.0)
        raise HTTPException(401, "wrong setup code (see data/access.token on this box)")
    try:
        u = auth.create_user(b.get("username", ""), str(b.get("password", "")), "admin", b.get("name", ""))
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    sid, csrf = auth.new_session(u["username"], ip)
    log.warning("first administrator created: %s", u["username"])
    r = JSONResponse({"ok": True})
    _set_session_cookies(r, request, sid, csrf)
    return r


@app.post("/api/auth/logout")
def api_logout(request: Request):
    auth.end_session(request.cookies.get(auth.COOKIE))
    r = JSONResponse({"ok": True})
    r.delete_cookie(auth.COOKIE)
    r.delete_cookie(auth.CSRF_COOKIE)
    return r


@app.get("/api/me")
def api_me(request: Request):
    u = who(request)
    return {"user": u["user"], "name": u["name"], "role": u["role"], "csrf": u.get("csrf", ""), "expires": u.get("expires"),
            "theme": auth.theme_of(u["user"], u["role"]), "themes": auth.themes_for(u["role"])}


@app.post("/api/me/theme")
async def api_my_theme(request: Request):
    u = who(request)
    b = await request.json()
    try:
        return {"theme": auth.set_theme(u["user"], u["role"], str(b.get("theme", "")))}
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e).strip("'")) from None


@app.post("/api/me/password")
async def api_my_password(request: Request):
    u = who(request)
    b = await request.json()
    rec = auth.users().get(u["user"] or "")
    if not rec or not auth.check_pw(str(b.get("current", "")), rec["pw"]):
        await asyncio.sleep(1.0)
        raise HTTPException(400, "the current password is wrong")
    try:
        auth.update_user(u["user"], u["user"], password=str(b.get("password", "")))
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    sid, csrf = auth.new_session(u["user"], request.client.host if request.client else "?")  # other sessions end; this browser stays signed in
    r = JSONResponse({"ok": True})
    _set_session_cookies(r, request, sid, csrf)
    return r


@app.get("/api/users")
def api_users():
    return {"users": auth.list_users(), "roles": list(auth.ROLES), "min_password": auth.MIN_PASSWORD}


@app.post("/api/users")
async def api_user_create(request: Request):
    b = await request.json()
    try:
        out = auth.create_user(b.get("username", ""), str(b.get("password", "")), str(b.get("role", "member")), b.get("name", ""))
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    log.warning("account %s (%s) created by %s", out["username"], out["role"], who(request)["user"])
    return out


@app.patch("/api/users/{username}")
async def api_user_update(username: str, request: Request):
    b = await request.json()
    try:
        out = auth.update_user(username, who(request)["user"], role=b.get("role"), enabled=b.get("enabled"), name=b.get("name"),
                               password=str(b["password"]) if b.get("password") else None)
    except KeyError:
        raise HTTPException(404, "no such account") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    log.warning("account %s changed by %s: %s", username, who(request)["user"], sorted(k for k in b if k != "password") + (["password"] if b.get("password") else []))
    return out


@app.post("/api/users/{username}/signout")
def api_user_signout(username: str):
    auth.revoke_user(username)
    return {"ok": True}


@app.delete("/api/users/{username}")
def api_user_delete(username: str, request: Request):
    try:
        auth.delete_user(username, who(request)["user"])
    except KeyError:
        raise HTTPException(404, "no such account") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"ok": True}


@app.get("/api/meetings-unowned")
def api_unowned():
    return [{"id": s["id"], "title": s.get("title"), "created": s.get("created"), "status": s.get("status")} for s in store.list_meetings() if not s.get("owner")]


def _assign(mid: str, username: str, actor: str) -> str | None:
    try:
        st = store.load_state(mid)
    except ValueError:
        return "bad id"
    if not st:
        return "not found"
    if st.get("owner"):
        return f"already owned by {st['owner']}"
    store.update_state(mid, owner=username)
    store.log_event(mid, f"owner set to {username} by {actor}")
    return None


@app.post("/api/meetings/assign")
async def api_assign_many(request: Request):
    """Admin gives meetings that have no owner yet (made before accounts existed) to an eligible member/admin. Owned meetings are never moved."""
    b = await request.json()
    username = str(b.get("user", ""))
    if not auth.can_own(username):
        raise HTTPException(400, "the new owner must be an enabled Member or Admin")
    ids = [s["id"] for s in store.list_meetings() if not s.get("owner")] if b.get("all_unowned") else [str(i) for i in (b.get("ids") or [])][:5000]
    done, skipped = [], {}
    for mid in ids:
        why = _assign(mid, username, who(request)["user"])
        if why:
            skipped[mid] = why
        else:
            done.append(mid)
    return {"assigned": done, "skipped": skipped}


@app.post("/api/meetings/{mid}/owner")
async def api_assign_one(mid: str, request: Request):
    username = str((await request.json()).get("user", ""))
    if not auth.can_own(username):
        raise HTTPException(400, "the new owner must be an enabled Member or Admin")
    why = _assign(mid, username, who(request)["user"])
    if why:
        raise HTTPException(409 if why.startswith("already") else 404, why)
    return store.load_state(mid)


@app.get("/")
def index():
    return FileResponse(HERE / "static/index.html", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- worker supervision
def supervise_worker():
    global worker_proc
    while True:
        if worker_proc is None or worker_proc.poll() is not None:
            if worker_proc is not None:
                log.warning("worker exited (%s), restarting in 5 s", worker_proc.returncode)
                time.sleep(5)
            logf = open(config.DATA / "worker.log", "ab")
            worker_proc = subprocess.Popen([sys.executable, str(HERE / "worker.py")], stdout=logf, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, cwd=str(HERE))
        time.sleep(3)


@app.on_event("startup")
def startup():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store.init()
    token()
    # a recording that was open when the server died is interrupted, not lost: its audio.pcm is on disk
    for st in store.list_meetings():
        if st["status"] == "recording":
            secs = store.pcm_seconds(st["id"])
            store.update_state(st["id"], status="interrupted", duration_s=secs)
            store.log_event(st["id"], f"server restarted while recording; {secs:.0f}s of audio recovered")
    capacity.MONITOR.start()
    threading.Thread(target=supervise_worker, daemon=True).start()
    threading.Thread(target=release_idle_ctc, daemon=True).start()

    def housekeeping():  # give the embedding model back when nobody has asked for 10 minutes
        while True:
            time.sleep(60)
            try:
                kb.EMB.maybe_unload()
                tts.release_idle()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=housekeeping, daemon=True).start()
    threading.Thread(target=lambda: ctc(), daemon=True).start()  # warm the live model


@app.on_event("shutdown")
def shutdown():
    tts.stop()
    for b in list(BOTS.values()):
        b["task"].cancel()
    BOT_LLM.stop()  # never leave the bot's language model running
    if worker_proc and worker_proc.poll() is None:
        worker_proc.terminate()


# ---------------------------------------------------------------- live recording
class LiveSegmenter:
    """Energy VAD with an adaptive noise floor. Thai pauses mid-sentence, so an utterance ends only after silence_ms."""

    FRAME = 320  # 20 ms

    def __init__(self, silence_ms: int = 1000, max_s: float = 14.0):
        self.silence = max(3, silence_ms // 20)
        self.max_frames = int(max_s * 50)
        self.buf = np.zeros(0, np.int16)
        self.nf = 150.0
        self.pre: list[np.ndarray] = []
        self.cur: list[np.ndarray] = []
        self.speaking = False
        self.run = 0
        self.quiet = 0
        self.pos = 0       # samples consumed
        self.start = 0
        self.level_db = -90.0
        self.utt = 0       # number of the utterance being spoken (changes when one ends)
        self.soft_frames = 9 * 50  # after 9 s a short dip (120 ms) already ends the utterance: video / music rarely has 1 s of silence

    def feed(self, data: bytes):
        self.buf = np.concatenate([self.buf, np.frombuffer(data[: len(data) & ~1], np.int16)])
        out = []
        while len(self.buf) >= self.FRAME:
            fr, self.buf = self.buf[: self.FRAME], self.buf[self.FRAME:]
            rms = float(np.sqrt(np.mean(fr.astype(np.float32) ** 2))) + 1e-3
            self.level_db = 20 * np.log10(rms / 32768)
            thr = max(220.0, self.nf * 3.0)
            loud = rms > thr
            if not self.speaking:
                self.nf = 0.97 * self.nf + 0.03 * min(rms, 4000)
                self.pre.append(fr)
                self.pre = self.pre[-15:]
                self.run = self.run + 1 if loud else 0
                if self.run >= 3:
                    self.speaking, self.quiet = True, 0
                    self.cur = list(self.pre)
                    self.start = self.pos + self.FRAME - len(self.cur) * self.FRAME
                    self.pre = []
            else:
                self.cur.append(fr)
                self.quiet = 0 if loud else self.quiet + 1
                if self.quiet >= self.silence or len(self.cur) >= self.max_frames or (len(self.cur) >= self.soft_frames and self.quiet >= 6):
                    audio = np.concatenate(self.cur)
                    if len(audio) >= 0.4 * config.SR:
                        # why it ended: a real pause, a short dip in long speech, or the hard length cap (may cut a word)
                        out.append((self.start, audio, "pause" if self.quiet >= self.silence else "dip" if self.quiet >= 6 else "cap"))
                    self.speaking, self.cur, self.run = False, [], 0
                    self.utt += 1
            self.pos += self.FRAME
        return out

    def current(self):
        """(utterance number, start sample, audio so far) while someone is speaking, else None: used for interim captions."""
        if self.speaking and len(self.cur) >= 40:  # at least 0.8 s
            return self.utt, self.start, np.concatenate(self.cur)
        return None

    def flush(self):
        if self.speaking and self.cur:
            audio = np.concatenate(self.cur)
            self.speaking, self.cur = False, []
            if len(audio) >= 0.4 * config.SR:
                return [(self.start, audio, "stop")]
        return []


@app.websocket("/ws/record")
async def ws_record(ws: WebSocket):
    me = await ws_user(ws)
    if not me:
        return
    await ws.accept()
    q = ws.query_params
    cfg = store.settings()
    mode = q.get("mode") if q.get("mode") in ("live", "after") else cfg["mode"]
    if store.free_gb() < 1.5:
        await ws.send_json({"type": "error", "message": f"Only {store.free_gb():.1f} GB of disk left, free some space first."})
        await ws.close()
        return
    mid = q.get("resume")
    if mid:
        try:
            st = store.load_state(mid)
        except ValueError:
            st = None
        why = auth.edit_problem(me, st) if st else None
        if why:  # checked BEFORE taking over a live session: only the owner may continue it
            await ws.send_json({"type": "error", "message": f"This recording cannot be continued: {why}."})
            await ws.close()
            return
        if mid in active:  # the browser reconnected before the server noticed the old link died: take the session over
            try:
                await active[mid]["ws"].close(code=4000)
            except Exception:  # noqa: BLE001
                pass
            for _ in range(300):
                if mid not in active:
                    break
                await asyncio.sleep(0.1)
            st = store.load_state(mid)
        if not st or st["status"] not in ("interrupted", "recording") or mid in active:
            await ws.send_json({"type": "error", "message": "This recording cannot be continued."})
            await ws.close()
            return
        pcm = store.pcm_path(mid)
        if pcm.exists() and pcm.stat().st_size % 2:
            with open(pcm, "ab") as f:
                f.truncate(pcm.stat().st_size - 1)
        store.update_state(mid, status="recording")
        store.log_event(mid, "recording continued")
    elif q.get("plan"):
        try:
            st = store.load_state(q["plan"])
        except ValueError:
            st = None
        if not st or st.get("status") != "planned":
            await ws.send_json({"type": "error", "message": "This planned meeting cannot be started (it may already have been started or deleted)."})
            await ws.close()
            return
        if auth.edit_problem(me, st):
            await ws.send_json({"type": "error", "message": f"This planned meeting cannot be started: {auth.edit_problem(me, st)}."})
            await ws.close()
            return
        mid = st["id"]
        upd = {"status": "recording", "mode": mode, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "started_from_plan": True}
        if q.get("title", "").strip():
            upd["title"] = q["title"].strip()[:120]
        st = store.update_state(mid, **upd)
        store.log_event(mid, "recording started from the plan")
    else:
        st = store.new_meeting(q.get("title", "").strip()[:120], mode, owner=me["user"])
        if q.get("participants"):
            store.update_state(st["id"], participants=store.clean_participants(q["participants"].split("|")))
        mid = st["id"]
    if q.get("packs") is not None and not q.get("resume") and not q.get("plan"):
        st = store.update_state(mid, vocab_packs=[x for x in q["packs"].split("|") if x])  # the packs chosen on the Record tab
    cfg = {**cfg, "vocab_rules": vocab.rules(vocab.active_ids(st))}  # Thai spellings -> terms of the meeting's vocabulary packs (live captions)
    base = store.pcm_seconds(mid)
    active[mid] = {"started": time.time(), "ws": ws, "terminate": False, "cancel": False, "cap": {"lat": [], "captions": 0, "partials": 0},
                   "mode": mode, "cfg": cfg, "ring": mics.Ring(), "merge": mics.LiveMerge(), "tracks": {}, "ncap": itertools.count(1)}
    stats.record(mid, "recording", started_at=time.strftime("%F %T"), started_ts=round(time.time(), 1), live_engine=cfg.get("live_engine", "typhoon"), accuracy=cfg["accuracy"],
                 live_partial=bool(cfg["live_partial"]), live_speakers=bool(cfg["live_speakers"]), mode=mode, vocab_packs=vocab.active_ids(st))
    seg = LiveSegmenter(cfg["silence_ms"])
    live_spk = speakers.LiveSpeakers(emb(), cfg["speaker_threshold"], cfg["voice_match_threshold"]) if (
        mode == "live" and cfg["live_speakers"] and speakers.available()) else None
    loop = asyncio.get_running_loop()
    send_lock = asyncio.Lock()
    live_file = open(store.mdir(mid) / "live.jsonl", "a", encoding="utf-8")
    pcm_file = open(store.pcm_path(mid), "ab")
    active[mid]["live_file"] = live_file
    active[mid]["pos"] = lambda: base + (pcm_file.tell() // 2) / config.SR  # the recording's position now (seconds): phones are kept on it
    stopped, last_sync, last_level, last_disk, tasks = False, time.time(), 0.0, time.time(), set()
    snd = {"quiet_since": time.time(), "warned": False, "db": -120.0}  # the no-sound watchdog: a muted or unplugged microphone during a meeting

    async def send(obj):
        async with send_lock:
            try:
                await ws.send_json(obj)
            except Exception:  # noqa: BLE001 - client gone; the audio is still safe on disk
                pass

    async def caption(start, audio, why="pause", t_ready=None):
        try:
            text, conf = await loop.run_in_executor(CAPTION_POOL, ctc().transcribe_long, audio)
        except Exception as e:  # noqa: BLE001
            await send({"type": "error", "message": f"live captions failed: {e}"})
            return
        text = asr.polish(text, cfg)
        if not text:
            return
        t0 = base + start / config.SR
        rec = {"t0": round(t0, 2), "t1": round(t0 + len(audio) / config.SR, 2), "text": text, "conf": round(conf, 3), "end": why}
        if live_spk:
            try:
                rec["speaker"] = await loop.run_in_executor(CAPTION_POOL, live_spk.label, audio)
            except Exception:  # noqa: BLE001 - captions matter more than labels
                pass
        if cfg.get("separate_overlaps", True) and separation.overlap_available():
            try:  # two voices at once in this sentence: marked ⇄ now, separated after the meeting (about 0.04 s per sentence)
                ov = await loop.run_in_executor(CAPTION_POOL, separation.find_overlaps, audio, float(cfg.get("overlap_threshold", 0.3)), separation.OVL_MIN_S)
                if ov:
                    rec["overlap"] = round(sum(b - a for a, b in ov) / config.SR, 1)
            except Exception:  # noqa: BLE001 - a mark, never a reason to lose the caption
                pass
        a = active.get(mid) or {}
        rec["id"] = f"r{next(a['ncap'])}" if a else None
        if a and a["tracks"]:
            rec = a["merge"].add(rec)  # the same words already captioned from a participant's own phone: not shown twice
        live_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        live_file.flush()
        cp = a.get("cap")
        if cp is not None:
            cp["captions"] += 1
            if t_ready:
                cp["lat"].append(time.time() - t_ready)  # from the end of the utterance (after the pause) to the caption being sent
        if rec.get("dup"):
            return
        await send({"type": "caption", **rec})
        broadcast(mid, {"type": "caption", "id": rec["id"], "t0": rec["t0"], "text": rec["text"], "speaker": rec.get("speaker"), "overlap": rec.get("overlap")})
        if mid in BOTS:
            BOTS[mid]["bot"].feed({"t0": rec["t0"], "t1": rec["t1"], "text": rec["text"], "speaker": rec.get("speaker")})

    last_partial, partial_busy = [0.0], [False]

    async def partial(utt, start, audio):
        try:
            text, _ = await loop.run_in_executor(PARTIAL_POOL, ctc().transcribe, audio[-int(14.5 * config.SR):])
            text = asr.polish(text, cfg)
            if text and seg.utt == utt:  # the sentence is still going on (otherwise the final caption is on its way)
                t0 = round(base + start / config.SR, 2)
                cp = active.get(mid, {}).get("cap")
                if cp is not None:
                    cp["partials"] += 1
                await send({"type": "partial", "t0": t0, "text": text})
                broadcast(mid, {"type": "partial", "t0": t0, "text": text})
        except Exception:  # noqa: BLE001 - interim text is a bonus, never a reason to disturb the recording
            pass
        finally:
            partial_busy[0] = False

    def spawn(items):
        for start, audio, why in items:
            t = asyncio.create_task(caption(start, audio, why, time.time()))
            tasks.add(t)
            t.add_done_callback(tasks.discard)

    active[mid]["send"] = send

    async def silence_watch():  # also fires when no audio packets arrive at all (the browser's microphone track ended)
        while True:
            await asyncio.sleep(5)
            quiet = time.time() - snd["quiet_since"]
            if not snd["warned"] and quiet > SILENT_WARN_S:
                snd["warned"] = True
                store.log_event(mid, f"no sound from the microphone for {int(quiet)} s")
                await send({"type": "notice", "kind": "silent", "message": f"No sound for {int(quiet)} s: is the microphone muted or unplugged? Say something to test it."})

    watch_task = asyncio.create_task(silence_watch())
    await ensure_bot(mid)  # a bot configured for this meeting survives a reconnect or a server restart
    if mode == "live":
        await loop.run_in_executor(None, ctc)  # (re)load the speech model now: 0.3 s from the graph cache, so the first caption is not delayed
    eng = None
    if mode == "live":
        try:
            eng = await loop.run_in_executor(CAPTION_POOL, ctc)  # load now: the Record tab shows the model really in use
        except Exception as e:  # noqa: BLE001
            log.warning("live caption model: %s", e)
    await send({"type": "started", "id": mid, "mode": mode, "base": base, "title": st["title"], "engines": engines.plan(cfg, eng)})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            now = time.time()
            if msg.get("bytes"):
                pcm_file.write(msg["bytes"])
                active[mid]["ring"].add(msg["bytes"], base + (pcm_file.tell() // 2) / config.SR)
                snd["db"] = peak_db(msg["bytes"])
                if snd["db"] > SOUND_DB:
                    snd["quiet_since"] = now
                    if snd["warned"]:
                        snd["warned"] = False
                        await send({"type": "notice", "kind": "silent", "clear": True})
                if mode == "live":
                    spawn(seg.feed(msg["bytes"]))
                    if cfg["live_partial"] and not partial_busy[0] and now - last_partial[0] >= PARTIAL_GAP.get(getattr(_ctc, "kind", ""), PARTIAL_EVERY_S) and _ctc is not None and (_ctc.backend == "npu" or getattr(_ctc, "kind", "") == "pathumma-rnnt"):
                        cur = seg.current()  # interim text of the sentence being spoken (the NPU makes this cheap)
                        if cur:
                            last_partial[0], partial_busy[0] = now, True
                            t = asyncio.create_task(partial(*cur))
                            tasks.add(t)
                            t.add_done_callback(tasks.discard)
                if now - last_sync > 1.0:
                    pcm_file.flush()
                    os.fsync(pcm_file.fileno())
                    config.HEARTBEAT.touch()
                    last_sync = now
                if now - last_level > 0.2:
                    last_level = now
                    await send({"type": "level", "db": round(seg.level_db if mode == "live" else snd["db"] - 10, 1),  # Record only: from the packet peak
                                "seconds": round(base + (pcm_file.tell() // 2) / config.SR, 1)})
                if now - last_disk > 30:
                    last_disk = now
                    if store.free_gb() < 0.5:
                        await send({"type": "error", "message": "Disk almost full: stopping the recording."})
                        stopped = True
                        break
            elif msg.get("text"):
                try:
                    if json.loads(msg["text"]).get("type") == "stop":
                        stopped = True
                        break
                except ValueError:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        cancelled = bool(active.get(mid, {}).get("cancel"))
        if active.get(mid, {}).get("terminate"):
            stopped = True  # ended from another device: finish and process it exactly like a normal stop
        watch_task.cancel()
        if mode == "live":
            spawn(seg.flush())
        if tasks:
            await asyncio.wait(tasks, timeout=30)
        for t in (active.get(mid, {}).get("tracks") or {}).values():
            t.close()
        pcm_file.flush()
        os.fsync(pcm_file.fileno())
        pcm_file.close()
        live_file.close()
        if not cancelled:
            cp = active.get(mid, {}).get("cap") or {}
            stats.record(mid, "recording", ended=time.strftime("%F %T"), ended_ts=round(time.time(), 1), duration_s=round(store.pcm_seconds(mid), 1), captions=cp.get("captions"),
                         partials=cp.get("partials"), latency=stats.latency_summary(cp.get("lat", [])), live_asr_backend=getattr(_ctc, "backend", None), live_asr=getattr(_ctc, "kind", "typhoon"))
        try:
            if cancelled:  # the host cancelled the meeting: nothing is kept, nothing is processed
                close_join(mid, "closed")
                await drop_bot(mid)
                store.update_state(mid, status="interrupted")  # (the delete helper only accepts meetings that are not in use)
                await loop.run_in_executor(None, mindex.delete_meetings, [mid], set())
                log.warning("meeting %s was cancelled and deleted", mid)
            elif stopped:
                close_join(mid, "ended")
                await stop_bot(mid)  # the closing recap is written in the background, after the recording is safe
                await loop.run_in_executor(None, finish_recording, mid)
            else:
                store.update_state(mid, status="interrupted", duration_s=store.pcm_seconds(mid))
                store.log_event(mid, "connection lost; audio so far is saved")
        finally:
            active.pop(mid, None)  # only now: a client taking over must see the final status
            if not active:
                config.HEARTBEAT.unlink(missing_ok=True)
    await send({"type": "cancelled", "id": mid} if cancelled else {"type": "stopped", "id": mid, "status": (store.load_state(mid) or {}).get("status")})
    try:
        await ws.close()
    except Exception:  # noqa: BLE001
        pass


@app.post("/api/meetings/{mid}/terminate")
async def api_terminate(mid: str):
    """End a meeting for everyone, from any device: the recording is finished and processed, guests are told, links are closed."""
    st = _meeting(mid)
    a = active.get(mid)
    if a:
        a["terminate"] = True
        try:
            await a["ws"].send_json({"type": "terminated", "message": "The meeting was ended by the host."})
            await a["ws"].close()
        except Exception:  # noqa: BLE001 - the recorder's connection may already be gone: the flag still finishes the meeting
            pass
        for _ in range(200):  # wait for the recorder's handler to finalize the audio
            if mid not in active:
                break
            await asyncio.sleep(0.1)
        close_join(mid, "ended")
        return {"ok": True, "status": (store.load_state(mid) or {}).get("status"), "ended": mid not in active}
    if st["status"] == "interrupted":
        await asyncio.get_running_loop().run_in_executor(None, finish_recording, mid)
        return {"ok": True, "status": (store.load_state(mid) or {}).get("status"), "ended": True}
    raise HTTPException(409, "This meeting is not being recorded.")


@app.post("/api/meetings/{mid}/cancel")
async def api_cancel(mid: str):
    """Cancel a meeting: the recording stops and the meeting (audio, captions, everything) is DELETED, nothing is transcribed. Also removes a
    meeting that was planned or interrupted. A meeting that is already being processed or is finished must be deleted from the Meetings list."""
    st = _meeting(mid)
    a = active.get(mid)
    if a:
        a["cancel"] = True
        try:
            await a["ws"].send_json({"type": "cancelled", "id": mid})
            await a["ws"].close()
        except Exception:  # noqa: BLE001 - the recorder's connection may already be gone: the flag still cancels the meeting
            pass
        for _ in range(300):
            if mid not in active:
                break
            await asyncio.sleep(0.1)
        return {"ok": True, "cancelled": mid not in active}
    if st["status"] not in ("planned", "interrupted", "recorded") or st.get("queue"):
        raise HTTPException(409, "This meeting is already finished or being processed: delete it from the Meetings list.")
    r = await asyncio.to_thread(mindex.delete_meetings, [mid], set(active))
    if not r["deleted"]:
        raise HTTPException(409, "The meeting is in use and cannot be cancelled now.")
    return {"ok": True, "cancelled": True}


def finish_recording(mid: str):
    secs = store.finalize_audio(mid)
    cfg = store.settings()
    store.update_state(mid, status="recorded", duration_s=secs, rush=False, queue=("transcribe" if cfg["auto_process"] and secs > 1 else None))
    store.log_event(mid, f"recording finished, {secs:.0f}s")


# ---------------------------------------------------------------- REST API
def _meeting(mid: str) -> dict:
    try:
        st = store.load_state(mid)
    except ValueError:
        st = None
    if not st:
        raise HTTPException(404, "meeting not found")
    return st


@app.get("/api/status")
def api_status():
    w = worker_proc is not None and worker_proc.poll() is None
    return {"worker": w, "recording": list(active), "free_gb": round(store.free_gb(), 1), "mem_available_gb": round(store.mem_available_gb(), 1),
            "models": {"typhoon_asr": (config.TYPHOON_ASR / "typhoon_asr_ctc.onnx").exists(), "whisper_th": (config.WHISPER_TH / "model.bin").exists(),
                       "summary_llm": config.LLM_GGUF.exists() and (config.LLAMA_BUILD / "bin/llama-server").exists()},
            "asr": {"npu_setting": bool(store.settings().get("asr_npu", True)), "backend": _ctc.backend if _ctc else "not loaded", "npu_error": _ctc.npu_error if _ctc else ""},
            "settings": {**store.settings(), "enroll_text_default": config.ENROLL_TEXT}}


@app.get("/api/preflight")
async def api_preflight(mode: str = "live"):
    """Checked before a meeting starts (Record tab): every part the meeting needs, each {id, label, status ok|warn|fail|info, detail}.
    The live caption model is really loaded and given a second of audio, so a broken model shows up now and not mid-meeting."""
    cfg = store.settings()
    mode = mode if mode in ("live", "after") else cfg["mode"]
    checks = []

    def add(id_, label, status, detail):
        checks.append({"id": id_, "label": label, "status": status, "detail": detail})

    add("server", "Server", "ok", "reachable")
    w = worker_proc is not None and worker_proc.poll() is None
    add("worker", "Processing worker", "ok" if w else "warn",
        "running" if w else "not running: the audio is still saved, but transcription waits until the worker is back (restart the app)")
    free = store.free_gb()
    add("disk", "Disk space", "fail" if free < 1.5 else "warn" if free < 5 else "ok",
        f"{free:.1f} GB free (about {int(free / 0.115)} hours of audio)" + (": free some space first, recording is refused below 1.5 GB" if free < 1.5 else ""))
    ram, need = store.mem_available_gb(), float(cfg.get("reserve_ram_gb", 2.0))
    add("ram", "Memory", "warn" if ram < need else "ok", f"{ram:.1f} GB available" + (f" (below the {need:g} GB kept for recording)" if ram < need else ""))
    try:  # the recording really can be written where meetings live (a read-only or missing disk shows up now, not when Start is pressed)
        config.MEETINGS.mkdir(parents=True, exist_ok=True)
        probe = config.MEETINGS / f".write-test-{os.getpid()}"
        with open(probe, "wb") as f:
            f.write(b"ok")
            f.flush()
            os.fsync(f.fileno())
        probe.unlink()
        add("storage", "Recording storage", "ok", f"writable ({config.MEETINGS})")
    except OSError as e:
        add("storage", "Recording storage", "fail", f"cannot write to {config.MEETINGS} ({e.strerror or e}): no recording is possible until this is fixed")
    loop = asyncio.get_running_loop()
    eng = None
    if mode == "live":
        try:
            t_load = time.time()
            eng = await loop.run_in_executor(CAPTION_POOL, ctc)
            load_s = time.time() - t_load
            t0 = time.time()
            noise = (np.random.default_rng(0).standard_normal(config.SR) * 30).astype(np.int16)  # one second of quiet noise
            await loop.run_in_executor(CAPTION_POOL, eng.transcribe_long, noise)
            ms = (time.time() - t0) * 1000
            p = engines.plan(cfg, eng)
            note = p["live"].get("note", "")
            add("live", "Live captions model", "warn" if note or ms > 3000 else "ok",
                f"{p['live']['name']} · answered in {ms:.0f} ms" + (f" (loaded in {load_s:.0f} s)" if load_s > 2 else "") + (f" · {note}" if note else "") + (" · slow: the device is busy" if ms > 3000 else ""))
        except Exception as e:  # noqa: BLE001
            add("live", "Live captions model", "fail", f"could not run ({type(e).__name__}: {str(e)[:120]}): choose “Record only”, or check Settings → Live caption model")
    else:
        add("live", "Live captions", "info", "off (Record only): the transcript is made after the meeting")
    p = engines.plan(cfg, eng)
    a = p["after"]
    add("after", "Transcript after the meeting", "ok" if a["ok"] else "warn", a["uses"] + (f" · {a['note']}" if a["note"] else ""))
    s_ = p["summary"]
    add("summary", "Summary model", "ok" if s_["ok"] else "warn", s_["uses"] + (f" · {s_['note']}" if s_["note"] else ""))
    if cfg.get("identify_speakers") or cfg.get("live_speakers"):
        ok = speakers.available()
        add("speakers", "Participant voices", "ok" if ok else "warn", "speaker model ready" if ok else "the speaker model is missing: participants will not be told apart")
        if cfg.get("identify_speakers") and cfg.get("separate_overlaps", True):
            import separation

            ok = separation.overlap_available() and separation.separation_available()
            add("overlaps", "People talking at once", "ok" if ok else "warn", "overlap detection + voice separation ready (after the meeting)" if ok
                else "voice separation is not installed (models/separation): overlapping speech stays on one line")
    others = [m for m in store.list_meetings() if m.get("status") == "recording"]
    if others:
        add("busy_rec", "Other recordings", "warn", "already recording: " + ", ".join(f"“{m.get('title') or m['id']}”" for m in others[:3]))
    jobs = [m for m in store.list_meetings() if m.get("status") in ("transcribing", "analysing", "summarizing")]
    if jobs:
        add("busy_job", "Background processing", "warn", f"“{jobs[0].get('title') or jobs[0]['id']}” is being processed now ({jobs[0]['status']}): it shares the processor, captions may lag")
    worst = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {"status": worst, "mode": mode, "checks": checks, "engines": p}


@app.post("/api/preflight/sound")
async def api_preflight_sound(request: Request, mode: str = "live"):
    """The sound check before a meeting. The browser records a few seconds from the real microphone through the same path a meeting uses
    (16 kHz 16-bit mono) and sends it here. Measured (level, room noise, speech) and then recognised by the live speech model, so
    "the microphone works, the sound arrives and the words come out" is proven before the meeting instead of discovered after it."""
    raw = await request.body()
    if len(raw) > 2 * config.SR * 15:
        raise HTTPException(413, "sample too long (15 s at most)")
    pcm = np.frombuffer(raw[: len(raw) & ~1], np.int16)
    cfg = store.settings()
    mode = mode if mode in ("live", "after") else cfg["mode"]
    checks, text, ms, eng = [], "", 0, None

    def add(id_, label, status, detail):
        checks.append({"id": id_, "label": label, "status": status, "detail": detail})

    st = sound_stats(pcm)
    if st["seconds"] < 0.5:
        add("sound", "Sound check", "fail", "no audio arrived from the browser: reload the page and check the microphone permission")
        return {"status": "fail", "checks": checks, "stats": st, "text": ""}
    lvl = f"speech {st['speech_db']:.0f} dB · room noise {st['noise_db']:.0f} dB · {st['snr_db']:.0f} dB above the noise"
    if st["peak_db"] < -60:
        add("sound", "Sound check", "fail", f"silent (peak {st['peak_db']:.0f} dB): nothing reaches the app. Is the microphone muted, or is another one chosen in Settings?")
    elif st["speech_s"] < 0.5 and st["speech_db"] - st["noise_db"] >= 10:
        add("sound", "Sound check", "warn", f"too quiet ({lvl}): speech at {st['speech_db']:.0f} dB is below the level the app listens for. Move closer to the microphone or raise its gain")
    elif st["speech_s"] < 0.5:
        add("sound", "Sound check", "warn", f"the microphone delivers sound (noise {st['noise_db']:.0f} dB) but no speech was heard in {st['seconds']:.0f} s: say a sentence while checking")
    elif st["clip_pct"] > 0.5:
        add("sound", "Sound check", "warn", f"too loud / distorted ({st['clip_pct']:.1f} % clipped): move the microphone away or lower its gain")
    elif st["speech_db"] < -35:
        add("sound", "Sound check", "warn", f"quiet speech ({lvl}): move closer to the microphone or raise its gain")
    elif st["snr_db"] < 10:
        add("sound", "Sound check", "warn", f"noisy ({lvl}): a loud room or the microphone far from the people; captions will have more errors")
    else:
        add("sound", "Sound check", "ok", lvl)
    if st["speech_s"] >= 0.5:
        loop = asyncio.get_running_loop()
        try:
            eng = await loop.run_in_executor(CAPTION_POOL, ctc)
            t0 = time.time()
            text, _ = await loop.run_in_executor(CAPTION_POOL, eng.transcribe_long, pcm)
            ms = int((time.time() - t0) * 1000)
            text = asr.polish(text or "", cfg).strip()
            p = engines.plan(cfg, eng)
            after = f" · the transcript after the meeting is made by {p['after']['uses']}" if mode != "live" and p["after"]["uses"] not in p["live"]["name"] else ""
            if text:
                add("speech", "Speech recognised", "ok", f"heard: “{text[:100]}” ({p['live']['name']} · {ms} ms){after}")
            else:
                add("speech", "Speech recognised", "warn", f"sound arrived but no words were recognised ({p['live']['name']} · {ms} ms): speak closer and clearly, or check that the right microphone is chosen{after}")
        except Exception as e:  # noqa: BLE001
            add("speech", "Speech recognised", "fail", f"the speech model could not run ({type(e).__name__}: {str(e)[:120]}): restart the app, or choose another live caption model in Settings")
    else:
        add("speech", "Speech recognised", "info", "not tested (nothing was said)")
    worst = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {"status": worst, "mode": mode, "checks": checks, "stats": st, "text": text, "ms": ms}


@app.get("/api/settings")
def get_settings():
    return {**store.settings(), "enroll_text_default": config.ENROLL_TEXT}


@app.put("/api/settings")
async def put_settings(request: Request):
    global _ctc
    before = (bool(store.settings().get("asr_npu", True)), store.settings().get("live_engine"))
    out = store.save_settings(await request.json())
    if (bool(out.get("asr_npu", True)), out.get("live_engine")) != before:
        with _ctc_lock:
            _drop_ctc()  # reloaded (NPU / CPU / Pathumma) at the next caption
    return {**out, "enroll_text_default": config.ENROLL_TEXT}


@app.get("/api/meetings")
def api_meetings():
    return store.list_meetings()


@app.get("/api/meetings/{mid}")
def api_meeting(mid: str):
    st = _meeting(mid)
    d = store.mdir(mid)
    st = {**st, "recording_now": mid in active}
    tr = store.read_json(d / "transcript.json")
    live = []
    if (d / "live.jsonl").exists():
        for ln in (d / "live.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                live.append(json.loads(ln))
            except ValueError:
                pass
    summary = (d / "summary.md").read_text(encoding="utf-8") if (d / "summary.md").exists() else None
    rep_ = store.read_json(d / "report.json")
    spk = public_speakers(mid, tr)
    nk = report.names_key(mid)
    stale = {"summary": bool(summary) and bool(spk) and st.get("summary_names_key") != nk,
             "report": bool(rep_ and rep_.get("narrative")) and rep_.get("narrative_names_key") != nk}
    return {"state": st, "transcript": tr, "live": live, "summary": summary, "has_audio": store.wav_path(mid).exists(),
            "speakers": spk, "report": rep_, "stale": stale, "summary_has_ai": (d / "summary.ai.md").exists(), "summary_rev": store.text_rev(summary),
            "summary_history": len(store.read_json(d / "summary-history.json", []) or [])}


def public_speakers(mid: str, tr: dict | None) -> list[dict]:
    spk = store.speakers(mid)
    out = []
    for sid, v in spk.items():
        segs = [s for s in (tr or {}).get("segments", []) if s.get("spk") == sid]
        best = max(segs, key=lambda s: s["t1"] - s["t0"], default=None)
        out.append({"id": sid, "name": v.get("name"), "voice_id": v.get("voice_id"), "score": v.get("score"), "suggest": v.get("suggest"),
                    "locked": bool(v.get("name_locked")), "talk_s": v.get("talk_s"), "segments": len(segs),
                    "sample": {"t0": best["t0"], "t1": best["t1"]} if best else None})
    return out


@app.get("/api/meetings/{mid}/audio")
def api_audio(mid: str):
    _meeting(mid)
    if not store.wav_path(mid).exists():
        raise HTTPException(404, "audio not ready")
    return FileResponse(store.wav_path(mid), media_type="audio/wav", filename=f"{mid}.wav")


@app.get("/api/meetings/{mid}/export.md")
def api_export(mid: str):
    st = _meeting(mid)
    d = store.mdir(mid)
    parts = []
    if (d / "summary.md").exists():
        parts.append((d / "summary.md").read_text(encoding="utf-8"))
    if (d / "report.md").exists():
        parts.append("\n---\n\n" + (d / "report.md").read_text(encoding="utf-8"))
    if (d / "transcript.md").exists():
        parts.append("\n---\n\n## Transcript\n\n" + (d / "transcript.md").read_text(encoding="utf-8").split("\n", 5)[-1])
    return PlainTextResponse("\n".join(parts) or f"# {st['title']}\n\n(nothing processed yet)\n", media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{mid}.md"'})


@app.patch("/api/meetings/{mid}")
async def api_patch(mid: str, request: Request):
    _meeting(mid)
    body = await request.json()
    upd = {}
    if isinstance(body.get("title"), str) and body["title"].strip():
        upd["title"] = body["title"].strip()[:120]
    if isinstance(body.get("remark"), str):
        upd["remark"] = body["remark"].strip()[:4000]
    if "participants" in body:
        if not isinstance(body["participants"], list):
            raise HTTPException(400, "participants must be a list of names")
        upd["participants"] = store.clean_participants(body["participants"])
    if "vocab_packs" in body:
        vp = body["vocab_packs"]
        if vp is not None and not (isinstance(vp, list) and all(isinstance(x, str) for x in vp)):
            raise HTTPException(400, "vocab_packs must be a list of pack ids (or null for the defaults)")
        upd["vocab_packs"] = None if vp is None else [x for x in vp if x in {p["id"] for p in vocab.list_packs()}]
    if "planned_at" in body:
        pa = str(body["planned_at"] or "").strip()[:16]
        if pa and not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", pa):
            raise HTTPException(400, "planned_at must look like 2026-10-06T14:30")
        upd["planned_at"] = pa
    if "agenda" in body:
        if not isinstance(body["agenda"], list):
            raise HTTPException(400, "agenda must be a list")
        upd["agenda"] = botmod.resolve({"agenda": body["agenda"]})["agenda"]
    if upd:
        store.update_state(mid, **upd)
        _regen(mid)  # transcript header and the report (who attended but stayed silent) follow the details
        if ("participants" in upd or "agenda" in upd) and (store.mdir(mid) / "materials").exists():
            await asyncio.to_thread(learn.refresh_terms, mid)  # names and agenda words are key terms too
    return store.load_state(mid)


# ---------------------------------------------------------------- planned meetings and their documents ("learn before the meeting")
@app.post("/api/meetings/plan")
async def api_plan(request: Request):
    body = await request.json()
    pa = str(body.get("planned_at") or "").strip()[:16]
    if pa and not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", pa):
        raise HTTPException(400, "planned_at must look like 2026-10-06T14:30")
    st = store.new_meeting(str(body.get("title") or "New meeting").strip()[:120] or "New meeting", "live", owner=who(request)["user"])
    store.update_state(st["id"], status="planned", planned_at=pa, participants=store.clean_participants(body.get("participants") or []), remark=str(body.get("remark") or "").strip()[:4000],
                       agenda=botmod.resolve({"agenda": body.get("agenda") or []})["agenda"])
    store.log_event(st["id"], "meeting planned")
    return store.load_state(st["id"])


def _materials_payload(mid: str) -> dict:
    d = store.read_json(learn._terms_path(mid)) or {"auto": [], "extra": [], "removed": []}
    cov = store.read_json(store.mdir(mid) / "agenda_coverage.json")
    return {"docs": kb.materials(mid).list_docs(), "terms": {"auto": d.get("auto", []), "extra": d.get("extra", []), "removed": d.get("removed", [])},
            "prompt": learn.asr_prompt(learn.trusted_terms(mid)), "correction_terms": learn.correction_terms(mid), "coverage": cov, "embedder": kb.EMB.available()}


@app.get("/api/meetings/{mid}/materials")
def api_materials(mid: str):
    st = _meeting(mid)
    if (store.mdir(mid) / "materials").exists() or st.get("agenda") or st.get("participants"):
        return _materials_payload(mid)
    return {"docs": [], "terms": {"auto": [], "extra": [], "removed": []}, "prompt": "", "correction_terms": [], "coverage": None, "embedder": kb.EMB.available()}


@app.post("/api/meetings/{mid}/materials")
async def api_materials_add(mid: str, request: Request):
    st = _meeting(mid)
    if st.get("status") in ("transcribing", "summarizing", "analysing"):
        raise HTTPException(409, "This meeting is being processed: add documents when it has finished.")
    body = await request.json()
    name = " ".join(str(body.get("name", "document.txt")).split())[:100] or "document.txt"
    try:
        if body.get("data_b64"):
            import base64

            data: bytes | str = base64.b64decode(body["data_b64"], validate=False)
        else:
            data = str(body.get("text", ""))
        d = await asyncio.to_thread(kb.materials(mid).add_document, name, data)
        await asyncio.to_thread(learn.refresh_terms, mid)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"This file could not be read: {type(e).__name__}")
    return {"doc": d, **_materials_payload(mid)}


@app.delete("/api/meetings/{mid}/materials/{did}")
async def api_materials_delete(mid: str, did: str):
    _meeting(mid)
    if not kb.materials(mid).delete_document(did):
        raise HTTPException(404, "not found")
    await asyncio.to_thread(learn.refresh_terms, mid)
    return _materials_payload(mid)


@app.put("/api/meetings/{mid}/terms")
async def api_terms(mid: str, request: Request):
    _meeting(mid)
    body = await request.json()
    learn.save_user_terms(mid, body.get("extra") or [], body.get("removed") or [])
    if body.get("refresh"):
        await asyncio.to_thread(learn.refresh_terms, mid)
    return _materials_payload(mid)


@app.get("/api/people")
def api_people():
    """Names for auto-complete: enrolled voices + people listed in earlier meetings (most used first)."""
    count: dict[str, int] = {}
    for st in store.list_meetings():
        for n in st.get("participants") or []:
            count[n] = count.get(n, 0) + 1
    for v in speakers.load_voices():
        count.setdefault(v["name"], 0)
    return sorted(count, key=lambda n: (-count[n], n.casefold()))


def _seg_payload(seg: dict) -> dict:
    return {k: seg.get(k) for k in ("text", "src", "edit_from", "v", "history", "reviewed", "orig")}


def _transcript_changed(mid: str, st: dict, tr: dict):
    d = store.mdir(mid)
    store.write_json(d / "transcript.json", tr)
    store.atomic_write(d / "transcript.md", store.transcript_md(st, tr["segments"], store.speaker_names(mid)))


async def _seg_change(mid: str, idx: int, request: Request, change):
    """Read-modify-write of one transcript line under the transcript lock. The client sends the line version it saw ("v"):
    if the line changed since (another tab, another person), the change is refused with 409 and the current line, never overwritten."""
    st = _meeting(mid)
    body = await request.json()
    me = who(request)
    with store.transcript_lock(mid):
        tr = store.read_json(store.mdir(mid) / "transcript.json")
        if not tr or not 0 <= idx < len(tr["segments"]):
            raise HTTPException(404, "segment not found")
        seg = tr["segments"][idx]
        if "v" in body and body["v"] is not None and int(body["v"]) != int(seg.get("v", 0)):
            return JSONResponse({"detail": "this line was changed in another tab or by someone else: it has been reloaded, check it and edit again",
                                 "segment": _seg_payload(seg)}, status_code=409)
        out = change(seg, body, me)
        seg["v"] = int(seg.get("v", 0)) + 1
        _transcript_changed(mid, st, tr)
    report.refresh(mid)
    return {"ok": True, **_seg_payload(seg), **(out or {})}


def _note_change(seg: dict, before: str, after: str, me: dict, kind: str):
    seg.setdefault("history", []).append({"from": before, "to": after, "by": me["user"], "at": time.strftime("%F %T"), "kind": kind})
    seg["history"] = seg["history"][-50:]


@app.put("/api/meetings/{mid}/segments/{idx}")
async def api_edit_segment(mid: str, idx: int, request: Request):
    """Correct a transcript line. Kept: the recognised wording (edit_from), every change (history: from, to, who, when). An edited line counts as
    reviewed and is protected from automatic passes. Corrections apply to THIS meeting only; a possible term goes to the admin's suggestion list."""
    def change(seg, body, me):
        before, text = seg["text"], " ".join(str(body.get("text", "")).split())[:5000]
        if text == before:
            return {"suggest": None}
        first = seg.setdefault("edit_from", {"text": before, "src": seg.get("src")})  # the recognised line, kept for good
        seg["text"] = text
        seg["src"] = (first.get("src") or "whisper") if text == first["text"] else "edited"
        _note_change(seg, before, text, me, "edit")
        seg["reviewed"] = {"by": me["user"], "at": time.strftime("%F %T")}
        sug = asr.suggest_rule(before, text)
        if sug and re.search(r"[A-Za-z]", sug["right"]) and re.search(r"[\u0e00-\u0e7f]", sug["wrong"]) and len(sug["right"]) <= 40 and len(sug["right"].split()) <= 4:
            # only a Thai spelling replaced by an English term / acronym looks like domain vocabulary; other wording fixes stay in this meeting
            try:
                vocab.add_suggestion(sug["right"], sug["wrong"], mid, me["user"])
            except ValueError:
                pass
        if sug and (me["role"] != "admin" or any(r == (sug["wrong"], sug["right"]) for r in asr.parse_rules(store.settings()["corrections"]))):
            sug = None  # the "remember this word everywhere?" bar is for administrators (shared word collection)
        return {"suggest": sug}
    return await _seg_change(mid, idx, request, change)


@app.post("/api/meetings/{mid}/segments/{idx}/undo")
async def api_undo_segment(mid: str, idx: int, request: Request):
    """Undo the latest text change of a line (one step; repeat for earlier ones). The undo itself is written to the history."""
    def change(seg, body, me):
        hist = seg.get("history") or []
        last = next((h for h in reversed(hist) if h["kind"] == "edit" and not h.get("undone")), None)
        if not last:
            if seg.get("src") == "edited" and seg.get("edit_from"):  # an edit made before histories existed: back to the recognised text
                last = {"from": seg["edit_from"]["text"], "to": seg["text"]}
            else:
                raise HTTPException(409, "nothing to undo on this line")
        before = seg["text"]
        seg["text"] = last["from"]
        last["undone"] = True
        first = seg.get("edit_from") or {}
        seg["src"] = (first.get("src") or "whisper") if seg["text"] == first.get("text") else "edited"
        _note_change(seg, before, seg["text"], me, "undo")
        return {}
    return await _seg_change(mid, idx, request, change)


@app.post("/api/meetings/{mid}/segments/{idx}/review")
async def api_review_segment(mid: str, idx: int, request: Request):
    """Mark a line as checked (or not). A reviewed line is never changed by automatic passes (saved words, writing style, re-transcription)."""
    def change(seg, body, me):
        if body.get("reviewed", True):
            seg["reviewed"] = {"by": me["user"], "at": time.strftime("%F %T")}
        else:
            seg.pop("reviewed", None)
        return {}
    return await _seg_change(mid, idx, request, change)


@app.post("/api/meetings/{mid}/suggest-term")
async def api_suggest_term(mid: str, request: Request):
    """Reviewer proposes a short Thai/English term (and the Thai spelling heard) for a domain pack. Inactive until an admin approves it."""
    _meeting(mid)
    b = await request.json()
    try:
        s = vocab.add_suggestion(str(b.get("term", "")), str(b.get("spelling", "")).strip(), mid, who(request)["user"], pack=b.get("pack") or None, source="reviewer")
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if not s:
        raise HTTPException(429, "the suggestion list is full: ask an administrator to review it")
    return s


@app.get("/api/vocab-suggestions")
def api_suggestions(status: str = "pending"):
    return {"items": vocab.suggestions(status if status in ("pending", "approved", "rejected", "") else "pending")}


@app.post("/api/vocab-suggestions/{sid}")
async def api_suggestion_decide(sid: str, request: Request):
    b = await request.json()
    try:
        return vocab.decide_suggestion(sid, b.get("action") == "approve", who(request)["user"], pack=b.get("pack"), term=b.get("term"), variants=b.get("variants"))
    except KeyError:
        raise HTTPException(404, "suggestion not found (already decided?)") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.put("/api/meetings/{mid}/summary")
async def api_edit_summary(mid: str, request: Request):
    """Hand-edited summary (Markdown). The text the model wrote is kept once in summary.ai.md so it can be restored."""
    st = _meeting(mid)
    if st["status"] in ("summarizing", "analysing") or st.get("queue") in ("summarize", "report", "speakers"):
        raise HTTPException(409, "the summary is being written right now: edit it when it is finished")
    body = await request.json()
    text = str(body.get("text", "")).replace("\r\n", "\n").strip()
    if not text:
        raise HTTPException(400, "the summary cannot be empty (use Restore AI version to go back)")
    if len(text) > 200_000:
        raise HTTPException(400, "the summary is too long")
    d = store.mdir(mid)
    me = who(request)
    with _summary_lock:
        cur = (d / "summary.md").read_text(encoding="utf-8") if (d / "summary.md").exists() else None
        if body.get("base") is not None and body["base"] != store.text_rev(cur):
            raise HTTPException(409, "the summary was changed (in another tab, by the AI or by someone else) since you opened the editor: "
                                     "copy your text, reload, and apply your change again")
        if cur is not None and not (d / "summary.ai.md").exists():
            store.atomic_write(d / "summary.ai.md", cur)
        store.atomic_write(d / "summary.md", text + "\n")
        store.add_summary_history(mid, "edit", me["user"], cur, text + "\n")
    store.update_state(mid, summary_edited=True, summary_edited_at=time.strftime("%F %T"), summary_edited_by=me["user"])
    store.log_event(mid, f"summary edited by hand ({me['user']})")
    return {"ok": True, "rev": store.text_rev(text + "\n")}


@app.post("/api/meetings/{mid}/summary/restore")
def api_restore_summary(mid: str, request: Request):
    _meeting(mid)
    d = store.mdir(mid)
    me = who(request)
    with _summary_lock:
        if not (d / "summary.ai.md").exists():
            raise HTTPException(404, "there is no AI version to go back to")
        cur = (d / "summary.md").read_text(encoding="utf-8") if (d / "summary.md").exists() else None
        ai = (d / "summary.ai.md").read_text(encoding="utf-8")
        store.atomic_write(d / "summary.md", ai)
        (d / "summary.ai.md").unlink()
        store.add_summary_history(mid, "restore", me["user"], cur, ai)
    store.update_state(mid, summary_edited=False, summary_edited_at=None, summary_edited_by=None)
    store.log_event(mid, f"summary restored to the AI version ({me['user']})")
    return {"ok": True}


@app.get("/api/meetings/{mid}/summary/history")
def api_summary_history(mid: str):
    _meeting(mid)
    return {"items": list(reversed(store.read_json(store.mdir(mid) / "summary-history.json", []) or []))}


@app.put("/api/meetings/{mid}/report/notes")
async def api_edit_notes(mid: str, request: Request):
    """Hand-edited 'contributions per person' note. body: {name, text}; text null = back to what the model wrote."""
    st = _meeting(mid)
    if st["status"] in ("summarizing", "analysing") or st.get("queue") in ("summarize", "report", "speakers"):
        raise HTTPException(409, "the report is being written right now: edit it when it is finished")
    b = await request.json()
    name, text = str(b.get("name", "")), b.get("text")
    d = store.mdir(mid)
    rep = store.read_json(d / "report.json")
    people = ((rep or {}).get("narrative") or {}).get("people")
    if not people or name not in people:
        raise HTTPException(404, "no AI note for this person")
    ai = rep.setdefault("narrative_ai", {})
    if text is None:
        if name in ai:
            people[name] = ai.pop(name)
    else:
        text = str(text).strip()
        if len(text) > 50_000:
            raise HTTPException(400, "the note is too long")
        ai.setdefault(name, people[name])
        people[name] = text
    store.write_json(d / "report.json", rep)
    store.atomic_write(d / "report.md", report.to_md(mid, rep))
    store.log_event(mid, f"AI note for {name} " + ("restored" if text is None else "edited by hand"))
    return {"ok": True}


# ---------------------------------------------------------------- statistics and comparison
@app.get("/api/stats/rows")
def api_stats_rows(date_from: str = "", date_to: str = ""):
    return {"rows": stats.rows(date_from, date_to), "metrics": {k: {"label": v[0], "unit": v[1], "better": v[2]} for k, v in stats.METRICS.items()}, "dimensions": stats.DIMENSIONS}


@app.get("/api/stats/compare")
def api_stats_compare(metric: str, by: str, date_from: str = "", date_to: str = ""):
    try:
        return stats.compare(metric, by, date_from, date_to)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/whois")
def api_whois(request: Request):
    """Admin 'Who is this?': unnamed participants with name suggestions (voices only when cross_meeting_match is on)."""
    return whois.listing(who(request), bool(store.settings().get("cross_meeting_match")))


@app.get("/api/meetings/{mid}/whois")
def api_meeting_whois(mid: str, request: Request):
    """Unnamed voices of one meeting with name hints for its Participants tab. Voice comparison only for administrators and only
    when cross_meeting_match is on; everyone else gets the meeting's participant list and what each voice said."""
    _meeting(mid)
    u = who(request)
    return whois.listing(u, u["role"] == "admin" and bool(store.settings().get("cross_meeting_match")), mid)


@app.post("/api/whois/reject")
async def api_whois_reject(request: Request):
    b = await request.json()
    mid, sid, cand = str(b.get("mid", "")), str(b.get("sid", "")), str(b.get("candidate", ""))
    _meeting(mid)
    if sid not in store.speakers(mid):
        raise HTTPException(404, "participant not found")
    whois.reject(mid, sid, cand)
    store.log_event(mid, f"'Who is this?': {sid} is not {cand}")
    return {"ok": True}


@app.get("/api/stats/participants")
def api_stats_participants(date_from: str = "", date_to: str = ""):
    return stats.participants(date_from, date_to)


@app.get("/api/stats/participants.csv")
def api_stats_participants_csv(date_from: str = "", date_to: str = ""):
    return PlainTextResponse(stats.participants_csv(date_from, date_to), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="secrets-participants.csv"'})


@app.get("/api/stats/export.csv")
def api_stats_export(date_from: str = "", date_to: str = ""):
    return PlainTextResponse(stats.export_csv(date_from, date_to), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="secrets-statistics.csv"'})


@app.get("/api/meetings/{mid}/stats")
def api_meeting_stats(mid: str):
    _meeting(mid)
    return {"stats": stats.backfill(mid), "row": stats.row(mid)}


@app.put("/api/meetings/{mid}/rating")
async def api_rating(mid: str, request: Request):
    _meeting(mid)
    j = await request.json()
    cur = (store.load_state(mid) or {}).get("rating") or {}
    for k in ("transcript", "summary"):
        if k in j:
            v = j[k]
            if v is None:
                cur.pop(k, None)
            elif isinstance(v, int) and 1 <= v <= 5:
                cur[k] = v
            else:
                raise HTTPException(400, "a rating is a whole number from 1 to 5")
    if isinstance(j.get("note"), str):
        cur["note"] = j["note"].strip()[:500]
    cur["at"] = time.strftime("%F %T")
    store.update_state(mid, rating=cur)
    return cur


# ---------------------------------------------------------------- vocabulary packs (domain vocab: Banking, Insurance, Regulator, Law, ...)
def _rev_ok(p: dict, j: dict):
    try:
        vocab.check_rev(p, j.get("rev"))
    except vocab.StaleEdit as e:
        raise HTTPException(409, str(e)) from None


def _pack(pid: str) -> dict:
    try:
        p = vocab.load(pid)
    except ValueError:
        p = None
    if not p:
        raise HTTPException(404, "pack not found")
    return p


@app.get("/api/vocab")
def api_vocab():
    return {"packs": vocab.list_packs()}


@app.post("/api/vocab")
async def api_vocab_create(request: Request):
    j = await request.json()
    try:
        return vocab.create(str(j.get("name", "")), str(j.get("desc", "")))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/vocab/restore-starters")
def api_vocab_restore():
    before = {p["id"] for p in vocab.list_packs()}
    store.write_json(vocab.DIR / ".removed", [])
    vocab.seed()
    return {"restored": sorted({p["id"] for p in vocab.list_packs()} - before)}


@app.get("/api/vocab/{pid}")
def api_vocab_get(pid: str, q: str = "", cat: str = "", page: int = 1, size: int = 50):
    p = _pack(pid)
    ents = p["entries"]
    if q.strip():
        k = q.strip().casefold()
        ents = [e for e in ents if k in e["term"].casefold() or k in e.get("note", "").casefold() or any(k in v.casefold() for v in e["variants"] + (e.get("aliases") or []))]
    if cat:
        ents = [e for e in ents if e["cat"] == cat]
    size = max(10, min(200, size))
    pages = max(1, -(-len(ents) // size))
    page = max(1, min(pages, page))
    return {"id": p["id"], "name": p["name"], "desc": p.get("desc", ""), "builtin": bool(p.get("builtin")), "enabled": bool(p.get("enabled")),
            "context": p.get("context", ""), "rev": int(p.get("rev") or 0), "total": len(p["entries"]), "matched": len(ents), "page": page, "pages": pages, "entries": ents[(page - 1) * size: page * size]}


@app.put("/api/vocab/{pid}")
async def api_vocab_update(pid: str, request: Request):
    p = _pack(pid)
    j = await request.json()
    _rev_ok(p, j)
    if isinstance(j.get("name"), str) and j["name"].strip():
        p["name"] = " ".join(j["name"].split())[:60]
    if isinstance(j.get("desc"), str):
        p["desc"] = j["desc"].strip()[:300]
    if isinstance(j.get("enabled"), bool):
        p["enabled"] = j["enabled"]
    if isinstance(j.get("context"), str):
        p["context"] = " ".join(j["context"].split())[:500]
    vocab.save(p)
    return {"ok": True, "rev": p["rev"]}


@app.delete("/api/vocab/{pid}")
def api_vocab_delete(pid: str, rev: int | None = None):
    _rev_ok(_pack(pid), {"rev": rev})
    vocab.delete_pack(pid)
    return {"ok": True}


@app.post("/api/vocab/{pid}/reset")
def api_vocab_reset(pid: str):
    _pack(pid)
    try:
        vocab.reset_builtin(pid)
    except KeyError:
        raise HTTPException(400, "only the starter packs can be reset")
    return {"ok": True}


@app.post("/api/vocab/{pid}/entries")
async def api_vocab_entry(pid: str, request: Request):
    j = await request.json()
    _rev_ok(_pack(pid), j)
    try:
        e = vocab.upsert(pid, j, old_term=j.get("old_term"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {**e, "rev": vocab.load(pid)["rev"]}


@app.post("/api/vocab/{pid}/entries/delete")
async def api_vocab_entry_delete(pid: str, request: Request):
    j = await request.json()
    _rev_ok(_pack(pid), j)
    return {"ok": vocab.delete_entry(pid, str(j.get("term", ""))), "rev": vocab.load(pid)["rev"]}


@app.post("/api/vocab/{pid}/import")
async def api_vocab_import(pid: str, request: Request):
    """body: {b64: file bytes} (a chosen file: size + UTF-8 checked on the bytes) or {text: pasted text}; optional rev, replace."""
    j = await request.json()
    _rev_ok(_pack(pid), j)
    try:
        if j.get("b64"):
            import base64
            import binascii

            try:
                data = base64.b64decode(str(j["b64"]), validate=True)
            except (binascii.Error, ValueError):
                raise ValueError("the file could not be read") from None
            r = vocab.import_bytes(pid, data, bool(j.get("replace")))
        else:
            r = vocab.import_text(pid, str(j.get("text", "")), bool(j.get("replace")))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {**r, "rev": vocab.load(pid)["rev"]}


@app.get("/api/vocab/{pid}/export.csv")
def api_vocab_export(pid: str):
    _pack(pid)
    return PlainTextResponse(vocab.export_csv(pid), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="vocab-{pid}.csv"'})


# ---------------------------------------------------------------- word collection: wrong word -> right word, applied to every transcript
def _words_payload():
    text = store.settings()["corrections"]
    return {"rules": [{"wrong": a, "right": b} for a, b in asr.parse_rules(text)], "text": text}


def _meetings_containing(wrong: str, limit: int = 400) -> int:
    n = 0
    for st in store.list_meetings()[:limit]:
        tr = store.read_json(store.mdir(st["id"]) / "transcript.json")
        if tr and any(wrong in s.get("text", "") for s in tr.get("segments", [])):
            n += 1
    return n


@app.get("/api/words")
def api_words():
    return _words_payload()


@app.post("/api/words")
async def api_words_add(request: Request):
    j = await request.json()
    wrong, right = str(j.get("wrong", "")).strip(), str(j.get("right", "")).strip()
    why = asr.rule_problem(wrong, right)
    if why:
        raise HTTPException(400, why)
    rules = dict(asr.parse_rules(store.settings()["corrections"]))
    rules[wrong] = right
    store.save_settings({"corrections": asr.rules_text(rules.items())})
    return {**_words_payload(), "meetings_with_wrong_word": _meetings_containing(wrong)}


@app.post("/api/words/delete")
async def api_words_delete(request: Request):
    wrong = str((await request.json()).get("wrong", "")).strip()
    rules = [(a, b) for a, b in asr.parse_rules(store.settings()["corrections"]) if a != wrong]
    store.save_settings({"corrections": asr.rules_text(rules)})
    return _words_payload()


@app.post("/api/meetings/{mid}/apply-words")
def api_apply_words(mid: str):
    """Run the word collection over a finished transcript (originals of changed lines are kept in seg['orig'])."""
    st = _meeting(mid)
    if st["status"] in mindex.ACTIVE or mid in active:
        raise HTTPException(409, "the meeting is still being recorded or processed")
    d = store.mdir(mid)
    tr = store.read_json(d / "transcript.json")
    if not tr:
        raise HTTPException(404, "no transcript yet")
    changed = 0
    pcfg = {**store.settings(), "vocab_rules": vocab.rules(vocab.active_ids(st))}
    protected = 0
    with store.transcript_lock(mid):
        tr = store.read_json(d / "transcript.json") or tr
        for seg in tr["segments"]:
            if store.review_count([seg]):  # a person checked / corrected this line: automatic passes leave it alone
                protected += 1
                continue
            new = asr.polish(seg["text"], pcfg)
            if new != seg["text"]:
                seg.setdefault("orig", seg["text"])
                seg["text"] = new
                seg["v"] = int(seg.get("v", 0)) + 1
                changed += 1
        if changed:
            _transcript_changed(mid, st, tr)
    if changed:
        report.refresh(mid)
    return {"ok": True, "changed": changed, "protected": protected}


@app.post("/api/meetings/{mid}/finish")
def api_finish(mid: str):
    st = _meeting(mid)
    if st["status"] != "interrupted" or mid in active:
        raise HTTPException(409, "not an interrupted recording")
    finish_recording(mid)
    return store.load_state(mid)


@app.post("/api/meetings/{mid}/reprocess")
async def api_reprocess(mid: str, request: Request):
    st = _meeting(mid)
    stage = (await request.json()).get("stage")
    if stage not in ("transcribe", "summarize", "report", "speakers") or mid in active or st["status"] in ("recording", "interrupted"):
        raise HTTPException(409, "cannot reprocess now")
    d = store.mdir(mid)
    if stage == "transcribe":
        n = store.review_count((store.read_json(d / "transcript.json") or {}).get("segments") or [])
        if n:
            raise HTTPException(409, f"{n} line(s) of this transcript were reviewed or corrected by hand: transcribing again would erase that review history, so it is blocked. "
                                     "Correct the remaining lines in Review instead.")
        for f in ("transcript.partial.jsonl", "chunks.json", "chunks.filled"):
            (d / f).unlink(missing_ok=True)
        new_status = "recorded"
    else:
        if not (d / "transcript.json").exists():
            raise HTTPException(409, "transcribe first")
        if stage == "summarize":
            (d / "summary.partial.json").unlink(missing_ok=True)
        elif stage == "speakers":
            (d / "summary.partial.json").unlink(missing_ok=True)
        else:  # report only: keep the summary checkpoints, redo the per-person notes and the overview
            part = store.read_json(d / "summary.partial.json", {}) or {}
            part.pop("notes", None)
            part.pop("overview", None)
            store.write_json(d / "summary.partial.json", part)
        new_status = "transcribed"
    # asked by an administrator: runs at once; by a member: waits for the processing window like a new meeting
    store.update_state(mid, status=new_status, queue=stage, error=None, attempts=0, progress=0.0, rush=who(request)["role"] == "admin")
    return store.load_state(mid)


@app.post("/api/queue/{mid}/now")
def api_process_now(mid: str, request: Request):
    """Administrator (ADMIN_WRITE, any owner's meeting): process this waiting meeting now instead of in the processing window (?rush=0: wait again)."""
    st = _meeting(mid)
    if not st.get("queue") and st["status"] not in ("transcribing", "analysing", "summarizing"):
        raise HTTPException(409, "this meeting is not waiting to be processed")
    store.update_state(mid, rush=bool(_body_rush(request)))
    store.log_event(mid, ("processing now (outside the window), asked by " if store.load_state(mid).get("rush") else "back to waiting for the processing window, asked by ") + who(request)["user"])
    return store.load_state(mid)


def _body_rush(request: Request) -> bool:
    return request.query_params.get("rush", "1") not in ("0", "false")


@app.get("/api/estimate")
def api_estimate(request: Request):
    """Settings page: how long the work after a 1-hour meeting takes with the choices on screen (not saved yet), step by step."""
    import schedule

    cfg = {**store.settings(), **{k: v for k, v in request.query_params.items() if k in ("accuracy", "summary_model")}}
    for k in ("identify_speakers", "speakers_after", "auto_summary", "batch_night", "batch_weekends"):
        if k in request.query_params:
            cfg[k] = request.query_params[k] in ("1", "true")
    for k in ("batch_start", "batch_end"):
        if schedule.valid_hm(request.query_params.get(k, "")):
            cfg[k] = request.query_params[k]
    dur = 3600.0
    steps = [[s, round(schedule.stage_secs(s, dur, cfg))] for s in schedule.stages_left({"status": "recorded", "queue": "transcribe"}, cfg)]
    total = sum(x[1] for x in steps)
    night = None
    if schedule.enabled(cfg):
        a, b = schedule._hm(cfg["batch_start"]), schedule._hm(cfg["batch_end"])
        night = (b - a) % 86400 or 86400
    return {"steps": steps, "total_s": total, "night_s": night, "per_night": int(night // total) if night and total else None}


@app.get("/api/compare")
def api_compare(summary_model: str = "", accuracy: str = ""):
    """Settings: every transcript mode and summary model side by side (errors, time for a 1-hour meeting, memory), for the user to choose."""
    import schedule

    cfg = store.settings()
    if summary_model in engines.SUMMARY:
        cfg["summary_model"] = summary_model
    if accuracy in engines.ACCURACY:
        cfg["accuracy"] = accuracy
    night = None
    if schedule.enabled(cfg):
        night = (schedule._hm(cfg["batch_end"]) - schedule._hm(cfg["batch_start"])) % 86400 or 86400  # one weeknight
    rest = sum(schedule.stage_secs(s, 3600, cfg) for s in ("speakers", "summarize") if s in schedule.stages_left({"queue": "transcribe"}, cfg))
    rows = []
    for mode, m in sorted(engines.MEASURED.items(), key=lambda kv: kv[1]["cer"]):
        t = schedule.stage_secs("transcribe", 3600, {**cfg, "accuracy": mode})
        rows.append({"mode": mode, "name": engines.ACCURACY.get(mode, mode), **m, "installed": engines.after_plan({**cfg, "accuracy": mode})["ok"],
                     "transcribe_s": round(t), "total_s": round(t + rest), "per_night": int(night // (t + rest)) if night else None})
    trs = schedule.stage_secs("transcribe", 3600, cfg) + schedule.stage_secs("speakers", 3600, cfg)
    sums = []
    for key, m in engines.SUMMARY_MEASURED.items():
        t = schedule.stage_secs("summarize", 3600, {**cfg, "summary_model": key})
        sums.append({"model": key, "name": engines.SUMMARY.get(key, key), **m, "installed": (config.SUMMARY_GGUF.get(key) or config.LLM_GGUF).exists(),
                     "summary_s": round(t), "total_s": round(trs + t)})
    return {"transcript": rows, "summary": sums, "night_s": night, "saved": {"accuracy": store.settings()["accuracy"], "summary_model": store.settings()["summary_model"]}}


@app.get("/api/queue")
def api_queue():
    """Every waiting / running meeting with its expected start and finish (schedule.estimate), and the processing window."""
    import schedule

    return schedule.estimate()


@app.delete("/api/meetings/{mid}")
def api_delete(mid: str):
    _meeting(mid)
    r = mindex.delete_meetings([mid], set(active))
    if r["skipped"]:
        raise HTTPException(409, "This meeting is being recorded or processed right now: wait until it finishes, then delete it.")
    return {"ok": True}


# ---------------------------------------------------------------- the Meetings table: search, filters, sorting, paging, CSV, bulk delete
def _table_params(q, fields, status, date_from, date_to, person, min_min, max_min, has, sort, dir, page, size) -> dict:
    return {"q": q.strip()[:200], "fields": fields, "status": status, "date_from": date_from[:16], "date_to": date_to[:16], "person": person.strip()[:60], "min_min": min_min, "max_min": max_min,
            "has": has, "sort": sort, "dir": dir, "page": page, "size": size}


@app.get("/api/table/meetings")
def api_table(q: str = "", fields: str = "all", status: str = "", date_from: str = "", date_to: str = "", person: str = "", min_min: str = "", max_min: str = "", has: str = "",
              sort: str = "created", dir: str = "desc", page: int = 1, size: int = 25):
    try:
        return mindex.query(_table_params(q, fields, status, date_from, date_to, person, min_min, max_min, has, sort, dir, page, size))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"bad filter: {e}")


@app.get("/api/table/ids")
def api_table_ids(q: str = "", fields: str = "all", status: str = "", date_from: str = "", date_to: str = "", person: str = "", min_min: str = "", max_min: str = "", has: str = ""):
    p = _table_params(q, fields, status, date_from, date_to, person, min_min, max_min, has, "created", "desc", 1, 25)
    return {"ids": mindex.ids(p)}


@app.get("/api/table/meetings.csv")
def api_table_csv(q: str = "", fields: str = "all", status: str = "", date_from: str = "", date_to: str = "", person: str = "", min_min: str = "", max_min: str = "", has: str = "",
                  sort: str = "created", dir: str = "desc"):
    p = _table_params(q, fields, status, date_from, date_to, person, min_min, max_min, has, sort, dir, 1, 25)
    return PlainTextResponse(mindex.export_csv(p), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="meetings.csv"'})


@app.post("/api/table/delete")
async def api_table_delete(request: Request):
    ids = [str(i) for i in ((await request.json()).get("ids") or [])][:5000]
    if not ids:
        raise HTTPException(400, "nothing selected")
    me, mine, not_mine = who(request), [], []
    for i in ids:
        try:
            st = store.load_state(i)
        except ValueError:
            st = None
        (mine if st and not auth.edit_problem(me, st) else not_mine).append(i)
    r = await asyncio.to_thread(mindex.delete_meetings, mine, set(active)) if mine else {"deleted": [], "skipped": []}
    r["skipped"] = list(r.get("skipped") or []) + [{"id": i, "reason": "not your meeting"} for i in not_mine]
    log.warning("deleted %d meeting(s), skipped %d", len(r["deleted"]), len(r["skipped"]))
    return r


# ---------------------------------------------------------------- voices (enrolled voice profiles)
def _voice_id(vid: str) -> str:
    import re

    if not re.fullmatch(r"v[0-9a-f]{8}", vid):
        raise HTTPException(404, "voice not found")
    return vid


@app.get("/api/voices")
def api_voices():
    return [voicelib.public(v) for v in speakers.load_voices()]


@app.patch("/api/voices/{vid}")
async def api_voice_rename(vid: str, request: Request):
    b = await request.json()
    try:
        return voicelib.set_notes(_voice_id(vid), b.get("name"), b.get("notes"))
    except KeyError:
        raise HTTPException(404, "voice not found") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.delete("/api/voices/{vid}")
def api_voice_delete(vid: str):
    if not speakers._vfile(_voice_id(vid)).exists():
        raise HTTPException(404, "voice not found")
    voicelib.delete_voice(vid)
    return {"ok": True}


@app.get("/api/voices/{vid}/samples/{n}.wav")
def api_voice_sample(vid: str, n: int, request: Request):
    if who(request)["role"] != "admin":  # retained voice samples are biometric data: administrators only
        raise HTTPException(403, "only an administrator can play voice samples")
    f = voicelib.sample_path(_voice_id(vid), n)
    if not f:
        raise HTTPException(404, "sample not found")
    return FileResponse(f, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.delete("/api/voices/{vid}/samples/{n}")
def api_voice_sample_delete(vid: str, n: int):
    try:
        return voicelib.delete_sample(_voice_id(vid), n)
    except KeyError:
        raise HTTPException(404, "sample not found") from None


@app.post("/api/voices/analyze")
async def api_voice_analyze(request: Request):
    """An uploaded recording (decoded to 16 kHz mono PCM16 by the browser) -> quality report + preview draft. Nothing is saved to the library yet."""
    b = await request.json()
    try:
        pcm = voicelib.pcm_from_b64(str(b.get("pcm_b64", "")))
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except Exception:  # noqa: BLE001 - bad base64
        raise HTTPException(400, "the audio could not be read") from None
    if not speakers.available():
        raise HTTPException(503, "speaker model missing")
    return await asyncio.get_running_loop().run_in_executor(ENROLL_POOL, voicelib.make_draft, pcm, emb(), "upload", who(request)["user"])


@app.get("/api/voices/drafts/{did}/audio")
def api_voice_draft_audio(did: str):
    f = voicelib.draft_wav(did)
    if not f or not f.exists():
        raise HTTPException(404, "this preview has expired: record again")
    return FileResponse(f, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.post("/api/voices/drafts/{did}/commit")
async def api_voice_commit(did: str, request: Request):
    b = await request.json()
    try:
        vid = b.get("voice_id") or None
        out = voicelib.commit(did, str(b.get("name", "")), who(request)["user"], voice_id=_voice_id(vid) if vid else None, replace=bool(b.get("replace")),
                              notes=b.get("notes"), consent=bool(b.get("consent")))
    except KeyError:
        raise HTTPException(404, "this preview has expired: record again") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    log.warning("voice saved for %s by %s", out["name"], who(request)["user"])
    return out


@app.delete("/api/voices/drafts/{did}")
def api_voice_draft_drop(did: str):
    voicelib.drop_draft(did)
    return {"ok": True}


@app.get("/api/tts/status")
def api_tts_status():
    return {"engine": tts.engine(), "chosen": store.settings().get("tts_engine", "piper"), "piper": tts.piper_available(), "pathumma_f5": tts.f5_available(), "pending": tts.pending()}


@app.post("/api/tts/prewarm")
async def api_tts_prewarm(request: Request):
    """Queue the fixed guidance sentences for the Pathumma voice (made in the background, never while recording)."""
    b = await request.json()
    texts = [str(t) for t in (b.get("texts") or []) if isinstance(t, str)]
    return {"queued": tts.prewarm(texts), "pending": tts.pending()}


@app.get("/api/tts")
async def api_tts(text: str):
    """Spoken guidance for hands-free enrolment (admin): a WAV made on this device by the installed Piper voice."""
    if not tts.available():
        raise HTTPException(503, "no local speech voice is installed")
    try:
        f = await asyncio.get_running_loop().run_in_executor(TTS_POOL, tts.wav, text[:300])
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"speech failed: {e}") from None
    return FileResponse(f, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})


TTS_POOL = ThreadPoolExecutor(1)


@app.websocket("/ws/handsfree")
async def ws_handsfree(ws: WebSocket):
    """EXPERIMENTAL hands-free enrolment, listening part (admin, only after Start in the browser). The browser streams PCM16 16 kHz;
    every utterance (ends at a ~0.7 s pause) is recognised on this device and interpreted for the current step:
    client -> {"type":"stage","stage":"wake|command|consent|name|confirm|retry","sensitivity":"strict|normal|loose"} / {"type":"mute","on":bool}
    server -> {"type":"level","db"} / {"type":"heard","text","conf","intent","name"?,"score"?}. Nothing is stored."""
    if not await ws_user(ws, roles=("admin",)):
        return
    await ws.accept()
    cfg = store.settings()
    if not cfg.get("handsfree_enroll"):
        await ws.send_json({"type": "error", "message": "Hands-free enrolment is switched off (Settings)."})
        await ws.close()
        return
    seg, stage, sens, muted, last = LiveSegmenter(silence_ms=700, max_s=8.0), "wake", "normal", False, 0.0
    loop = asyncio.get_running_loop()

    def recognise(audio):
        return ctc().transcribe(audio)  # Typhoon realtime on the NPU (CPU fallback); it pads the clip itself

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("text"):
                m = json.loads(msg["text"])
                if m.get("type") == "stage":
                    stage = str(m.get("stage", "wake"))
                    sens = m.get("sensitivity") if m.get("sensitivity") in handsfree.SENSITIVITY else sens
                elif m.get("type") == "mute":  # while the guide speaks: do not listen to ourselves
                    muted = bool(m.get("on"))
                    seg = LiveSegmenter(silence_ms=700, max_s=8.0)
                continue
            if not msg.get("bytes") or muted:
                continue
            for _start, audio, _why in seg.feed(msg["bytes"]):
                text, conf = await loop.run_in_executor(CAPTION_POOL, recognise, audio)
                r = handsfree.parse(text, stage, cfg.get("handsfree_wake") or "Secrets", sens)
                await ws.send_json({"type": "heard", "text": text, "conf": round(float(conf), 2), "stage": stage, **r})
            if time.time() - last > 0.15:
                last = time.time()
                await ws.send_json({"type": "level", "db": round(seg.level_db, 1), "speaking": seg.speaking})
    except WebSocketDisconnect:
        return


ENROLL_POOL = ThreadPoolExecutor(1)  # voice analysis for enrolment: never waits behind (or delays) live captions
CHECK_FROM_S, CHECK_EVERY_S = 22.0, 5.0  # hands-free capture: first quality check after ~22 s, then about every 5 s


@app.websocket("/ws/enroll")
async def ws_enroll(ws: WebSocket):
    """Enrol a voice from the microphone (admin): the browser streams 16 kHz PCM16, then sends {"type":"stop"}; the answer is a DRAFT
    (quality report + preview) that is saved only after confirmation. ?auto=1 (hands-free): quality is checked from ~22 s every ~5 s and the
    capture stops by itself as soon as it passes (or at 40 s)."""
    me = await ws_user(ws, roles=("admin",))
    if not me:
        return
    await ws.accept()
    if not speakers.available():
        await ws.send_json({"type": "error", "message": "Speaker model missing"})
        await ws.close()
        return
    auto = ws.query_params.get("auto") == "1"
    cap_s = voicelib.MAX_CAPTURE_S if auto else 90.0
    loop = asyncio.get_running_loop()
    buf, last, next_check, checking = bytearray(), 0.0, CHECK_FROM_S, None
    try:
        while len(buf) < cap_s * config.SR * 2:
            if checking is not None and checking.done():
                q = checking.result()
                checking = None
                await ws.send_json({"type": "quality", "quality": q})
                if q["good"]:
                    break
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                buf += msg["bytes"]
                secs = len(buf) / 2 / config.SR
                if time.time() - last > 0.2:
                    last = time.time()
                    fr = np.frombuffer(bytes(buf[-6400:]), np.int16).astype(np.float32)
                    db = 20 * np.log10(float(np.sqrt(np.mean(fr ** 2))) / 32768 + 1e-6) if len(fr) else -90
                    await ws.send_json({"type": "level", "db": round(db, 1), "seconds": round(secs, 1)})
                if auto and secs >= next_check and checking is None:
                    next_check = secs + CHECK_EVERY_S
                    pcm = np.frombuffer(bytes(buf[: len(buf) & ~1]), np.int16).copy()
                    checking = loop.run_in_executor(ENROLL_POOL, lambda pcm=pcm: {k: v for k, v in voicelib.quality(pcm, emb()).items() if not k.startswith("_")})
            elif msg.get("text") and json.loads(msg["text"]).get("type") == "stop":
                break
    except WebSocketDisconnect:
        return
    await ws.send_json({"type": "analysing"})
    pcm = np.frombuffer(bytes(buf[: len(buf) & ~1]), np.int16).copy()
    d = await loop.run_in_executor(ENROLL_POOL, voicelib.make_draft, pcm, emb(), "hands-free" if auto else "microphone", me["user"])
    await ws.send_json({"type": "draft", "draft": d})
    await ws.close()


# ---------------------------------------------------------------- meeting bot (conducts the meeting; RAG over the knowledge base)
BOTS: dict[str, dict] = {}          # meeting id -> {"bot": MeetingBot, "task": asyncio.Task}
BOT_LLM = botmod.LlmHandle()


def public_bot_msg(m: dict) -> dict:
    return {"type": "bot", "id": m["id"], "t": m["t"], "kind": m["kind"], "text": m["text"], "sources": m["sources"], "by": m["by"], "bot": m["bot"], "llm": m["llm"]}


async def _emit(mid: str, msg: dict):
    pm = public_bot_msg(msg)
    a = active.get(mid)
    if a and a.get("send"):
        await a["send"](pm)
    broadcast(mid, pm)


async def _bot_loop(mid: str, b):
    while not b.closed:
        await asyncio.sleep(3)
        try:
            if mid in active:  # nothing to conduct while the recording is interrupted
                await b.tick()
            if BOT_LLM.state == "ready" and time.time() - BOT_LLM.last_used > 1200:
                await asyncio.get_running_loop().run_in_executor(None, BOT_LLM.stop)  # unused for 20 minutes: free its memory (it restarts on demand)
        except Exception:  # noqa: BLE001 - the bot must never take the recorder down
            log.exception("bot tick failed")


async def start_bot(mid: str, cfg: dict, announce: bool = True):
    if mid in BOTS:
        BOTS[mid]["bot"].cfg = cfg
        return BOTS[mid]["bot"]

    async def emit(msg, mid=mid):
        await _emit(mid, msg)

    b = botmod.MeetingBot(mid, cfg, emit, BOT_LLM, participants=lambda: (store.load_state(mid) or {}).get("participants") or [],
                          enrolled=lambda: [v["name"] for v in speakers.load_voices()], cap_settings=store.settings)
    restored = (not announce) and b.restore()
    BOTS[mid] = {"bot": b, "task": asyncio.create_task(_bot_loop(mid, b))}
    if announce or not restored:
        await b.start()
    return b


async def ensure_bot(mid: str):
    cfg = (store.load_state(mid) or {}).get("bot")
    if cfg and cfg.get("enabled") and mid not in BOTS:
        await start_bot(mid, botmod.resolve(cfg), announce=False)


async def drop_bot(mid: str):
    """A cancelled meeting: the bot is dropped WITHOUT a closing recap (nothing may be written into a folder that is about to be deleted)."""
    b = BOTS.pop(mid, None)
    if b:
        b["task"].cancel()
        if not BOTS:
            await asyncio.get_running_loop().run_in_executor(None, BOT_LLM.stop)


async def stop_bot(mid: str):
    """The meeting ended: the bot says goodbye (recap written in the background), then its language model is released."""
    b = BOTS.pop(mid, None)
    if not b:
        return

    async def finish():
        try:
            await b["bot"].close()
        except Exception:  # noqa: BLE001
            log.exception("bot closing failed")
        b["task"].cancel()
        if not BOTS:
            await asyncio.get_running_loop().run_in_executor(None, BOT_LLM.stop)

    asyncio.get_running_loop().create_task(finish())


@app.get("/api/bot")
def api_bot_get():
    d = botmod.load_store()
    return {**d, "presets": {k: {"label": v["label"], "info": v["info"]} for k, v in botmod.PRESETS.items()}, "llm": {"state": BOT_LLM.state, "reason": BOT_LLM.reason}}


@app.put("/api/bot/defaults")
async def api_bot_defaults(request: Request):
    return botmod.save_store(defaults=await request.json())


@app.put("/api/bot/templates")
async def api_bot_templates(request: Request):
    return botmod.save_store(templates=(await request.json()).get("templates") or [])


@app.get("/api/meetings/{mid}/bot")
def api_meeting_bot(mid: str):
    st = _meeting(mid)
    d = store.mdir(mid)
    msgs = []
    try:
        msgs = [json.loads(ln) for ln in (d / "bot.jsonl").read_text(encoding="utf-8").splitlines()][-300:]
    except (OSError, ValueError):
        pass
    b = BOTS.get(mid)
    return {"config": st.get("bot"), "running": bool(b), "status": b["bot"].status() if b else None, "messages": msgs, "notes": store.read_json(d / "bot_notes.json"),
            "snapshot": store.read_json(d / "bot_state.json")}  # agenda progress is saved as it happens: it is there even before the closing notes are written


@app.put("/api/meetings/{mid}/bot/config")
async def api_meeting_bot_config(mid: str, request: Request):
    st = _meeting(mid)
    cfg = botmod.resolve(await request.json(), botmod.resolve({}, (st.get("bot") or botmod.load_store()["defaults"])))
    store.update_state(mid, bot=cfg)
    if mid in active and cfg["enabled"]:
        if mid in BOTS:
            BOTS[mid]["bot"].cfg = cfg
        else:
            await start_bot(mid, cfg)
    elif mid in BOTS and not cfg["enabled"]:
        await stop_bot(mid)
    return {"config": cfg, "running": mid in BOTS}


@app.post("/api/meetings/{mid}/bot/{action}")
async def api_meeting_bot_action(mid: str, action: str, request: Request):
    _meeting(mid)
    b = BOTS.get(mid)
    if not b:
        raise HTTPException(409, "The bot is not running for this meeting.")
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    bt = b["bot"]
    if action == "next":
        await bt.next()
    elif action == "prev":
        await bt.prev()
    elif action == "goto":
        await bt.goto(int(body.get("index", 0)))
    elif action == "pause":
        bt.paused = bool(body.get("paused"))
    elif action == "say":
        text = str(body.get("text", "")).strip()[:500]
        if text:
            await bt.say("info", text, force=True, by="host")
    elif action == "ask":
        asyncio.get_running_loop().create_task(bt.ask(str(body.get("question", ""))[:400], "host"))
    else:
        raise HTTPException(404, "unknown action")
    return {"ok": True, "status": bt.status()}


# ---- knowledge base (RAG)
@app.get("/api/kb")
def api_kb():
    return kb.status()


@app.post("/api/kb")
async def api_kb_add(request: Request):
    body = await request.json()
    name = " ".join(str(body.get("name", "document.txt")).split())[:100] or "document.txt"
    try:
        if body.get("data_b64"):
            import base64

            data: bytes | str = base64.b64decode(body["data_b64"], validate=False)
        else:
            data = str(body.get("text", ""))
        d = await asyncio.to_thread(kb.add_document, name, data)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001 - unreadable PDF etc.
        raise HTTPException(400, f"This file could not be read: {type(e).__name__}")
    return d


@app.delete("/api/kb/{did}")
def api_kb_delete(did: str):
    if not kb.delete_document(did):
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.patch("/api/kb/{did}")
async def api_kb_patch(did: str, request: Request):
    if not kb.set_enabled(did, bool((await request.json()).get("enabled"))):
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.post("/api/kb/search")
async def api_kb_search(request: Request):
    body = await request.json()
    hits = await asyncio.to_thread(kb.search, str(body.get("q", ""))[:400], min(8, int(body.get("k", 4))))
    return [{"name": h["name"], "sim": h["sim"], "text": h["text"][:500]} for h in hits]


# ---------------------------------------------------------------- system monitor + capacity
@app.get("/api/system")
def api_system():
    s = dict(capacity.MONITOR.snapshot())
    mid = next(iter(active), None)
    s["recording"] = [{"id": m, "title": (store.load_state(m) or {}).get("title", ""), "guests": len(GUESTS.get(m, {}))} for m in active]
    s["capacity"] = capacity.plan(store.settings(), len(GUESTS.get(mid, {})) if mid else 0, bool(active), _bot_wants_llm())
    return s


@app.get("/api/capacity")
def api_capacity(mid: str = "", max_guests: int | None = None, reserve_ram_gb: float | None = None, cpu_limit_pct: float | None = None):
    """What the box can take now; the three limits may be passed to try them out before saving."""
    cfg = store.settings()
    for k, v in (("max_guests", max_guests), ("reserve_ram_gb", reserve_ram_gb), ("cpu_limit_pct", cpu_limit_pct)):
        if v is not None:
            cfg[k] = v
    guests = len(GUESTS.get(mid, {})) if mid else 0
    return {"now": capacity.plan(cfg, guests, bool(active), _bot_wants_llm()), "snapshot": capacity.MONITOR.snapshot(), "guests": guests, "costs": capacity.COST}


# ---------------------------------------------------------------- e-mail through the user's own SMTP account (OPTIONAL)
@app.get("/api/email")
def api_email_get():
    return mailer.public()   # never contains the password


@app.put("/api/email")
async def api_email_put(request: Request):
    try:
        mailer.save(await request.json())
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    return mailer.public()


@app.post("/api/email/test")
async def api_email_test(request: Request):
    cfg = mailer.load()
    to = (await request.json()).get("to") or cfg["from_addr"]
    try:
        sent = await asyncio.to_thread(mailer.send_test, to)
    except Exception as e:  # noqa: BLE001 - shown to the user in plain words
        raise HTTPException(400, mailer.friendly_error(e, cfg))
    return {"ok": True, "to": sent}


@app.post("/api/meetings/{mid}/email")
async def api_email_meeting(mid: str, request: Request):
    _meeting(mid)
    cfg = mailer.load()
    if not cfg["enabled"]:
        raise HTTPException(409, "E-mail is turned off in Settings.")
    body = await request.json()
    opts = {f"attach_{k}": bool(v) for k, v in (body.get("attach") or {}).items() if k in ("summary", "report", "transcript")}
    try:
        sent = await asyncio.to_thread(mailer.send_meeting, mid, body.get("to") or cfg["default_to"], opts)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, mailer.friendly_error(e, cfg))
    return {"ok": True, "to": sent, "log": (store.load_state(mid) or {}).get("email_log", [])}


def _bot_wants_llm() -> bool:
    return any(b["bot"].cfg["llm"] == "auto" for b in BOTS.values())


# ---------------------------------------------------------------- guests: join from the local network (OPTIONAL)
# Off unless Settings -> "Allow participants to join" is on AND the host opens joining for a meeting that is being recorded.
# A guest holds only the secret link of that meeting: it gives a read-only stream of live captions, nothing else.
GUESTS: dict[str, dict[int, dict]] = {}   # meeting id -> {guest id: {name, ip, q, since}}
_gid = itertools.count(1)


def join_state(mid: str) -> dict:
    return (store.load_state(mid) or {}).get("join") or {}


ROOM_FILE = config.DATA / "room.json"


def room_token(new: bool = False) -> str:
    """The room's join link (printable QR on the Record tab): it always leads to the meeting being recorded with joining open, and before
    the meeting starts participants wait on it. Replaced (old printouts stop working) with POST /api/room/new."""
    r = store.read_json(ROOM_FILE) or {}
    if new or not r.get("token"):
        r = {"token": secrets.token_urlsafe(12), "created": time.strftime("%F %T")}
        store.write_json(ROOM_FILE, r)
    return r["token"]


def room_urls() -> list[str]:
    return [f"https://{ip}:{config.PORT}/join/{room_token()}" for ip in lan_ips()]


def is_room_token(token: str) -> bool:
    return bool(token) and hmac.compare_digest(room_token(), token)


def open_meeting_for(token: str) -> str | None:
    """The meeting a join link belongs to, only while it is being recorded and joining is open. The room link -> the newest such meeting."""
    if not token or not store.settings()["allow_join"]:
        return None
    if is_room_token(token):
        live = [mid for mid in active if join_state(mid).get("open")]
        return max(live, key=lambda m: active[m]["started"]) if live else None
    for mid in list(active):
        j = join_state(mid)
        if j.get("open") and hmac.compare_digest(str(j.get("token", "")), token):
            return mid
    return None


def broadcast_host(mid: str):
    """The host's join panel: who is sending their own microphone."""
    a = active.get(mid)
    if a and a.get("send"):
        asyncio.get_running_loop().create_task(a["send"]({"type": "guests", **_join_payload(mid)}))


async def guest_audio(mid: str, gid: int, name: str, phone: dict, data: bytes, queue):
    """A chunk of a participant's own microphone (16 kHz int16, from join.html): saved on the meeting's timeline, and in a live meeting
    captioned under their name when it is their own voice (mics.LiveGate)."""
    a = active.get(mid)
    if not a or len(data) > 64000:
        return
    if not phone.get("track"):
        if phone.get("refused"):
            return
        why = None if store.settings().get("guest_mic", True) else "The host does not take participants' microphones in this meeting."
        if not why and len(a["tracks"]) >= mics.MAX_MICS:
            why = f"Already {mics.MAX_MICS} microphones in this meeting: yours is not used."
        if why:
            phone["refused"] = True
            await queue.put({"type": "mic", "on": False, "message": why})
            return
        t = mics.Track(mid, f"g{gid}", name, a["pos"]() - (len(data) // 2) / config.SR)
        phone.update(track=t, seg=mics_segmenter(a["cfg"]), gate=mics.LiveGate(), n=itertools.count(1))
        a["tracks"][t.id] = t
        store.log_event(mid, f"{name} turned on their own microphone")
        await queue.put({"type": "mic", "on": True})
        broadcast_host(mid)
    t = phone["track"]
    pad = t.write(data, a["pos"]())
    if pad is None or a["mode"] != "live":
        return
    items = (phone["seg"].feed(b"\0\0" * pad) if pad else []) + phone["seg"].feed(data)
    for start, audio, _why in items:
        asyncio.get_running_loop().create_task(guest_caption(mid, name, phone, t.start_s + start / config.SR, audio))


def mics_segmenter(cfg: dict):
    return LiveSegmenter(cfg["silence_ms"])


async def guest_caption(mid: str, name: str, phone: dict, t0: float, audio):
    a = active.get(mid)
    if not a:
        return
    t1 = t0 + len(audio) / config.SR
    if not phone["gate"].own(audio, a["ring"].rms(t0, t1)):
        return  # someone else near this phone: the room microphone captions them
    loop = asyncio.get_running_loop()
    try:
        text, conf = await loop.run_in_executor(CAPTION_POOL, ctc().transcribe_long, audio)
    except Exception:  # noqa: BLE001 - the room captions go on
        return
    text = asr.polish(text, a["cfg"])
    if not text or mid not in active:
        return
    rec = a["merge"].add({"id": f"{phone['track'].id}-{next(phone['n'])}", "t0": round(t0, 2), "t1": round(t1, 2), "text": text, "conf": round(conf, 3),
                          "speaker": name, "mic": phone["track"].id})
    a["live_file"].write(json.dumps(rec, ensure_ascii=False) + "\n")
    a["live_file"].flush()
    if a.get("send"):
        await a["send"]({"type": "caption", **rec})
    broadcast(mid, {"type": "caption", "id": rec["id"], "t0": rec["t0"], "text": rec["text"], "speaker": name, "replaces": rec.get("replaces"), "mic": True})
    if mid in BOTS:
        BOTS[mid]["bot"].feed({"t0": rec["t0"], "t1": rec["t1"], "text": rec["text"], "speaker": name})


def broadcast(mid: str, msg: dict):
    """Never blocks the recorder: a slow phone just loses its oldest queued message."""
    for g in list(GUESTS.get(mid, {}).values()):
        try:
            g["q"].put_nowait(msg)
        except asyncio.QueueFull:
            try:
                g["q"].get_nowait()
                g["q"].put_nowait(msg)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass


def close_join(mid: str, reason: str = "closed"):
    j = join_state(mid)
    if j.get("open"):
        store.update_state(mid, join={**j, "open": False})
    broadcast(mid, {"type": reason})


def lan_ips() -> list[str]:
    """Addresses other devices on the network can reach this box at (no loopback / container bridges)."""
    ips: list[str] = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"], capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            f = line.split()
            if len(f) > 3 and not f[1].startswith(("docker", "br-", "veth", "virbr", "lo")):
                ips.append(f[3].split("/")[0])
    except (OSError, subprocess.SubprocessError):
        pass
    if not ips:
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.0.2.1", 9))
                ips.append(s.getsockname()[0])
        except OSError:
            pass
    return ips


def clean_guest_name(raw: str) -> str:
    return " ".join(re.sub(r"[\x00-\x1f\x7f<>]", " ", raw or "").split())[:40]


def last_captions(mid: str, n: int = 40) -> list[dict]:
    f = store.mdir(mid) / "live.jsonl"
    out = []
    try:
        rows = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines()[-n:]]
        gone = {r.get("replaces") for r in rows}
        for r in rows:
            if not r.get("dup") and r.get("id") not in gone:
                out.append({"type": "caption", "id": r.get("id"), "t0": r["t0"], "text": r["text"], "speaker": r.get("speaker"), "overlap": r.get("overlap")})
    except (OSError, ValueError):
        pass
    return out


def _join_payload(mid: str) -> dict:
    j = join_state(mid)
    live = bool(j.get("open")) and mid in active
    ips = lan_ips()
    return {"enabled": store.settings()["allow_join"], "open": live, "recording": mid in active,
            "urls": room_urls() if live else [],  # the room link (the same QR before, during and after the meeting)
            "guests": [{"name": g["name"], "since": g["since"], "mic": f"g{k}" in (active.get(mid, {}).get("tracks") or {})} for k, g in GUESTS.get(mid, {}).items()],
            "max": capacity.limits(store.settings())["max_guests"],
            "capacity": capacity.plan(store.settings(), len(GUESTS.get(mid, {})), mid in active, _bot_wants_llm())}


@app.get("/api/join")
def api_join_info():
    return {"enabled": store.settings()["allow_join"], "max": capacity.limits(store.settings())["max_guests"], "app_urls": [f"https://{ip}:{config.PORT}/" for ip in lan_ips()]}


@app.get("/api/meetings/{mid}/join")
def api_join_get(mid: str):
    _meeting(mid)
    return _join_payload(mid)


@app.post("/api/meetings/{mid}/join")
async def api_join_set(mid: str, request: Request):
    _meeting(mid)
    if not store.settings()["allow_join"]:
        raise HTTPException(409, "Joining from the network is turned off in Settings.")
    want = bool((await request.json()).get("open"))
    if want:
        if mid not in active:
            raise HTTPException(409, "Only a meeting that is being recorded can be joined.")
        store.update_state(mid, join={"open": True, "token": secrets.token_urlsafe(12), "opened": time.strftime("%F %T")})  # a new link every time
        store.log_event(mid, "joining from the network opened")
    else:
        close_join(mid, "closed")
        store.log_event(mid, "joining from the network closed")
    return _join_payload(mid)


@app.get("/api/meetings/{mid}/join-qr.svg")
def api_join_qr(mid: str, i: int = 0):
    import segno
    from fastapi.responses import Response

    _meeting(mid)
    urls = _join_payload(mid)["urls"]
    if not urls:
        raise HTTPException(404, "joining is not open")
    buf = io.BytesIO()
    segno.make(urls[min(max(i, 0), len(urls) - 1)], error="m").save(buf, kind="svg", scale=6, border=2, dark="#111", light="#fff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/api/room")
def api_room():
    r = store.read_json(ROOM_FILE) or {}
    return {"enabled": store.settings()["allow_join"], "urls": room_urls(), "created": r.get("created"), "wifi": store.settings().get("guide_wifi", "")}


@app.post("/api/room/new")
def api_room_new():
    room_token(new=True)
    return api_room()


@app.get("/api/room-qr.svg")
def api_room_qr(i: int = 0):
    import segno
    from fastapi.responses import Response

    urls = room_urls()
    if not urls:
        raise HTTPException(404, "no network address found")
    buf = io.BytesIO()
    segno.make(urls[min(max(i, 0), len(urls) - 1)], error="m").save(buf, kind="svg", scale=8, border=2, dark="#111", light="#fff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/api/app-qr.svg")
def api_app_qr(i: int = 0):
    """QR code of this app's address on the local network (Settings -> Join): other devices on the same Wi-Fi scan it to open Zecret."""
    import segno
    from fastapi.responses import Response

    urls = [f"https://{ip}:{config.PORT}/" for ip in lan_ips()]
    if not urls:
        raise HTTPException(404, "no network address found")
    buf = io.BytesIO()
    segno.make(urls[min(max(i, 0), len(urls) - 1)], error="m").save(buf, kind="svg", scale=6, border=2, dark="#111", light="#fff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/join/{token}")
def join_page(token: str):
    if not store.settings()["allow_join"]:
        return HTMLResponse("<meta charset=utf-8><p style='font:16px system-ui;padding:24px'>Joining is turned off.</p>", status_code=404)
    return FileResponse(HERE / "static/join.html", headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@app.get("/join/{token}/qr.svg")
async def join_qr(token: str, request: Request):
    """The QR code of this join link, shown on a participant's page so the people next to them can scan it and join too (public like the page;
    only for the room link or a meeting open for joining)."""
    import segno
    from fastapi.responses import Response

    if not store.settings()["allow_join"] or not (is_room_token(token) or open_meeting_for(token)):
        await asyncio.sleep(1.0)  # slows down anyone guessing links
        raise HTTPException(404, "not open for joining")
    urls = [f"https://{ip}:{config.PORT}/join/{token}" for ip in lan_ips()]
    here = f"https://{request.url.hostname}:{config.PORT}/join/{token}"
    url = here if here in urls or not urls else urls[0]  # the address this phone used, when it is one of the box's own
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=8, border=2, dark="#111", light="#fff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@app.websocket("/ws/join")
async def ws_join(ws: WebSocket):
    q = ws.query_params
    name = clean_guest_name(q.get("name", ""))
    mid = open_meeting_for(q.get("token", ""))
    await ws.accept()
    ip = ws.client.host if ws.client else "?"
    if not mid or not name:
        waiting = not mid and is_room_token(q.get("token", "")) and store.settings()["allow_join"]
        if not mid and not waiting:
            await asyncio.sleep(1.0)  # slows down anyone guessing links
        await ws.send_json({"type": "error", "code": "name" if mid else "waiting" if waiting else "closed",
                            "message": "Please type your name." if mid else "The meeting has not started yet. This page connects by itself when it starts." if waiting
                            else "This meeting is not open for joining (it may have ended)."})
        await ws.close()
        return
    room = GUESTS.setdefault(mid, {})
    if sum(1 for g in room.values() if g["ip"] == ip) >= config.JOIN_PER_IP:
        await ws.send_json({"type": "error", "code": "full", "message": "Too many connections from this device."})
        await ws.close()
        return
    ok, why = capacity.admit_guest(store.settings(), len(room), True, _bot_wants_llm())
    if not ok:  # the device protects the recording first: RAM / CPU / the host's own limit
        await ws.send_json({"type": "error", "code": "full", "message": why})
        await ws.close()
        return
    gid, queue = next(_gid), asyncio.Queue(maxsize=200)
    room[gid] = {"name": name, "ip": ip, "q": queue, "since": time.strftime("%H:%M:%S")}
    st = store.load_state(mid) or {}
    if store.settings()["join_add_attendee"] and name.casefold() not in [p.casefold() for p in st.get("participants") or []]:
        store.update_state(mid, participants=store.clean_participants((st.get("participants") or []) + [name]))
    store.log_event(mid, f"{name} joined from {ip}")
    # Participants see the transcript and nothing else: no bot messages, no questions, no summary, no other people's names.
    await ws.send_json({"type": "hello", "title": st.get("title", ""), "backlog": last_captions(mid), "count": len(room), "bot": None,
                        "mic": bool(store.settings().get("guest_mic", True)), "captions": (active.get(mid) or {}).get("mode") == "live"})
    broadcast(mid, {"type": "count", "n": len(room)})

    async def pump():
        while True:
            m = await queue.get()
            if m["type"] == "bot":
                continue
            await ws.send_json(m)
            if m["type"] in ("ended", "closed"):
                return

    last_ask = [0.0]

    phone = {}  # this participant's own microphone, when they turn it on (mics.py)

    async def listen():  # reading detects the disconnect; a guest may also ask the bot when the host allows it
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                await guest_audio(mid, gid, name, phone, msg["bytes"], queue)
                continue
            if msg.get("text") and len(msg["text"]) < 1200:
                try:
                    j = json.loads(msg["text"])
                except ValueError:
                    continue
                b = BOTS.get(mid)
                if GUEST_CAN_ASK and j.get("type") == "ask" and b and b["bot"].cfg["guest_questions"]:
                    if time.time() - last_ask[0] < 45:
                        await queue.put({"type": "info", "message": "Please wait a little before asking again."})
                    elif b["bot"].asking >= 2 or not capacity.admit_enroll(store.settings(), 0)[0]:
                        await queue.put({"type": "info", "message": "The bot is busy, please try again soon."})
                    else:
                        last_ask[0] = time.time()
                        asyncio.get_running_loop().create_task(b["bot"].ask(str(j.get("text", ""))[:300], name))

    tasks = [asyncio.create_task(pump()), asyncio.create_task(listen())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception:  # noqa: BLE001
        pass
    finally:
        for t in tasks:
            t.cancel()
        if phone.get("track"):
            phone["track"].close()
            (active.get(mid, {}).get("tracks") or {}).pop(phone["track"].id, None)
            broadcast_host(mid)
        room.pop(gid, None)
        if room:
            broadcast(mid, {"type": "count", "n": len(room)})
        else:
            GUESTS.pop(mid, None)
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- participants enrol their own voice (OPTIONAL, host approves)
# Off unless Settings -> "Let participants enrol their own voice". The host opens a time-limited secret link; a participant ticks
# consent, reads a text aloud, and the voiceprint waits as "pending" until the host approves it (so nobody can enrol under another's name).
ENROLL_POOL = ThreadPoolExecutor(1)
ENROLLING: dict[str, int] = {}


def enroll_session() -> dict | None:
    if not store.settings()["allow_enroll"]:
        return None
    s = store.read_json(config.ENROLL_FILE)
    return s if s and s.get("expires", 0) > time.time() else None


def valid_enroll_token(t: str) -> bool:
    s = enroll_session()
    return bool(s and t and hmac.compare_digest(str(s["token"]), t))


def _embeddings_from_pcm(raw: bytes, min_s: float):
    pcm = np.frombuffer(raw[: len(raw) & ~1], np.int16)
    units = speakers.vad_units(pcm)
    speech = np.concatenate([pcm[a:b] for a, b in units]) if units else pcm[:0]
    secs = len(speech) / config.SR
    if secs < min_s:
        return None, secs
    n = max(2, min(5, len(speech) // (6 * config.SR)))
    step = len(speech) // n
    return [emb()(speech[i * step:(i + 1) * step][: 10 * config.SR]) for i in range(n)], secs


def _enroll_payload() -> dict:
    s = enroll_session()
    have = {v["name"].casefold() for v in speakers.load_voices()}
    return {"enabled": store.settings()["allow_enroll"], "open": bool(s), "expires_in": int(s["expires"] - time.time()) if s else 0,
            "urls": [f"https://{ip}:{config.PORT}/enroll/{s['token']}" for ip in lan_ips()] if s else [],
            "pending": [{"id": p["id"], "name": p["name"], "samples": len(p["embeddings"]), "speech_s": p.get("speech_s"), "time": p["time"], "best": p.get("best"),
                         "exists": p["name"].casefold() in have} for p in speakers.load_pending()]}


@app.get("/api/enroll")
def api_enroll_get():
    return _enroll_payload()


@app.post("/api/enroll")
async def api_enroll_set(request: Request):
    if not store.settings()["allow_enroll"]:
        raise HTTPException(409, "Voice enrolment by participants is turned off in Settings.")
    body = await request.json()
    if body.get("open"):
        minutes = body.get("minutes") if body.get("minutes") in (15, 30, 60) else 30
        store.write_json(config.ENROLL_FILE, {"token": secrets.token_urlsafe(12), "expires": time.time() + minutes * 60, "opened": time.strftime("%F %T")})  # new link each time
        try:
            os.chmod(config.ENROLL_FILE, 0o600)
        except OSError:
            pass
    else:
        config.ENROLL_FILE.unlink(missing_ok=True)
    return _enroll_payload()


@app.post("/api/enroll/pending/{pid}/approve")
def api_enroll_approve(pid: str):
    v = speakers.approve_pending(pid)
    if not v:
        raise HTTPException(404, "not found")
    return {"voice": speakers.public_voice(v)}


@app.delete("/api/enroll/pending/{pid}")
def api_enroll_reject(pid: str):
    if not speakers.delete_pending(pid):
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.get("/api/enroll-qr.svg")
def api_enroll_qr(i: int = 0):
    import segno
    from fastapi.responses import Response

    urls = _enroll_payload()["urls"]
    if not urls:
        raise HTTPException(404, "enrolment is not open")
    buf = io.BytesIO()
    segno.make(urls[min(max(i, 0), len(urls) - 1)], error="m").save(buf, kind="svg", scale=6, border=2, dark="#111", light="#fff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/enroll/{token}")
def enroll_page(token: str):
    if not store.settings()["allow_enroll"]:
        return HTMLResponse("<meta charset=utf-8><p style='font:16px system-ui;padding:24px'>Voice enrolment is turned off.</p>", status_code=404)
    return FileResponse(HERE / "static/enroll.html", headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


# ---- the enrolment guide: explains the steps, tells the address, answers questions about enrolment only
GUIDE_ASKS: dict[str, list[float]] = {}


def _guide_payload() -> dict:
    e = _enroll_payload()
    app_urls = [f"https://{ip}:{config.PORT}/" for ip in lan_ips()]
    return {**guide.payload(app_urls, e["urls"], e["expires_in"] // 60, store.settings()), "enabled": e["enabled"], "open": e["open"], "expires_in": e["expires_in"]}


@app.get("/api/guide")
def api_guide():
    """For the host's display screen: everything the guide says, plus who is waiting for approval."""
    g = _guide_payload()
    g["pending"] = [p["name"] for p in speakers.load_pending()]
    g["enrolled"] = len(speakers.load_voices())
    return g


@app.get("/guide")
def guide_page():
    return FileResponse(HERE / "static/guide.html", headers={"Cache-Control": "no-store"})


@app.get("/enroll/{token}/guide")
def enroll_guide(token: str):
    if not valid_enroll_token(token):
        raise HTTPException(404, "This enrolment link is not open any more.")
    return {**_guide_payload(), "read_text": store.enroll_text()}


@app.post("/enroll/{token}/ask")
async def enroll_guide_ask(token: str, request: Request):
    if not valid_enroll_token(token):
        await asyncio.sleep(1.0)
        raise HTTPException(404, "This enrolment link is not open any more.")
    ip = request.client.host if request.client else "?"
    now = time.time()
    hist = [t for t in GUIDE_ASKS.get(ip, []) if now - t < 600]
    if hist and now - hist[-1] < 2 or len(hist) >= 40:
        raise HTTPException(429, "Please wait a moment before asking again.")
    GUIDE_ASKS[ip] = hist + [now]
    body = await request.json()
    g = _guide_payload()
    c = guide.context(g["app_urls"], g["enroll_urls"], g["expires_in"] // 60, g["wifi"], g["name"])
    return guide.answer(str(body.get("q", ""))[:300], c, body.get("lang") if body.get("lang") in ("th", "en") else None)


@app.websocket("/ws/enroll-guest")
async def ws_enroll_guest(ws: WebSocket):
    q = ws.query_params
    await ws.accept()
    ip = ws.client.host if ws.client else "?"

    async def fail(code: str, msg: str, slow: bool = False):
        if slow:
            await asyncio.sleep(1.0)  # slows down anyone guessing links
        await ws.send_json({"type": "error", "code": code, "message": msg})
        await ws.close()

    if not valid_enroll_token(q.get("token", "")):
        return await fail("closed", "This enrolment link is not open any more. Ask the host for a new one.", True)
    name = clean_guest_name(q.get("name", ""))
    if not name:
        return await fail("name", "Please type your name.")
    if q.get("consent") != "1":
        return await fail("consent", "Please tick the consent box first.")
    if not speakers.available():
        return await fail("unavailable", "Voice recognition is not available on this device.")
    if len(speakers.load_pending()) >= config.ENROLL_MAX_PENDING:
        return await fail("full", "Too many voices are waiting for the host. Please tell the host.")
    if ENROLLING.get(ip, 0) >= 3:
        return await fail("busy", "Too many recordings from this device at once.")
    ok, why = capacity.admit_enroll(store.settings(), sum(ENROLLING.values()))
    if not ok:
        return await fail("busy", why)
    ENROLLING[ip] = ENROLLING.get(ip, 0) + 1
    try:
        buf, last = bytearray(), 0.0
        while len(buf) < 75 * config.SR * 2:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                buf += msg["bytes"]
                if time.time() - last > 0.2:
                    last = time.time()
                    fr = np.frombuffer(bytes(buf[-6400:]), np.int16).astype(np.float32)
                    db = 20 * np.log10(float(np.sqrt(np.mean(fr ** 2))) / 32768 + 1e-6) if len(fr) else -90
                    await ws.send_json({"type": "level", "db": round(db, 1), "seconds": round(len(buf) / 2 / config.SR, 1)})
            elif msg.get("text") and json.loads(msg["text"]).get("type") == "stop":
                break
        if not valid_enroll_token(q.get("token", "")):
            return await fail("closed", "The enrolment link closed while you were recording. Ask the host for a new one.")
        embs, secs = await asyncio.get_running_loop().run_in_executor(ENROLL_POOL, _embeddings_from_pcm, bytes(buf), config.ENROLL_MIN_SPEECH_S)
        if embs is None:
            return await fail("short", f"Only {secs:.0f} s of speech was heard. Read the whole text aloud (about 20 s) in a quiet place, close to the microphone.")
        p = speakers.add_pending(name, embs, {"consent": True, "ip": ip, "speech_s": round(secs, 1), "best": speakers.best_match(embs)})
        await ws.send_json({"type": "submitted", "speech_s": round(secs, 1)})
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        ENROLLING[ip] = max(0, ENROLLING.get(ip, 1) - 1)


# ---------------------------------------------------------------- speakers of a meeting
_spk_lock = threading.Lock()


def _regen(mid: str):
    d = store.mdir(mid)
    tr = store.read_json(d / "transcript.json")
    if tr:
        store.atomic_write(d / "transcript.md", store.transcript_md(store.load_state(mid), tr["segments"], store.speaker_names(mid)))
        report.refresh(mid)


@app.put("/api/meetings/{mid}/speakers/{sid}")
async def api_speaker(mid: str, sid: str, request: Request):
    _meeting(mid)
    body = await request.json()
    with _spk_lock, store.transcript_lock(mid):
        spk = store.speakers(mid)
        if sid not in spk:
            raise HTTPException(404, "participant not found")
        v = spk[sid]
        if body.get("merge_into"):
            tgt = body["merge_into"]
            if tgt == sid or tgt not in spk:
                raise HTTPException(400, "bad merge target")
            t = spk[tgt]
            w1, w2 = float(t.get("talk_s") or 1), float(v.get("talk_s") or 1)
            c = speakers.centroid([np.asarray(t["centroid"]) * w1, np.asarray(v["centroid"]) * w2]) if t.get("centroid") and v.get("centroid") else t.get("centroid")
            t.update(centroid=c.round(5).tolist() if hasattr(c, "round") else c, talk_s=round(w1 + w2, 1), samples=(t.get("samples", []) + v.get("samples", []))[:8])
            if not t.get("name") and v.get("name"):
                t.update(name=v["name"], voice_id=v.get("voice_id"), name_locked=v.get("name_locked"))
            del spk[sid]
            tr = store.read_json(store.mdir(mid) / "transcript.json")
            if tr:
                for s in tr["segments"]:
                    if s.get("spk") == sid:
                        s["spk"] = tgt
                store.write_json(store.mdir(mid) / "transcript.json", tr)
        elif body.get("voice_id"):
            lib = {x["id"]: x for x in speakers.load_voices()}
            lv = lib.get(body["voice_id"])
            if not lv:
                raise HTTPException(404, "voice not found")
            score = float(speakers.voice_centroid(lv) @ speakers.unit(v["centroid"])) if v.get("centroid") else None
            v.update(name=lv["name"], voice_id=lv["id"], score=round(score, 3) if score is not None else None, name_locked=True, suggest=None)
        elif "name" in body:
            name = str(body["name"]).strip()[:60]
            if not name:
                v.update(name=None, voice_id=None, score=None, name_locked=False)
            else:
                v.update(name=name, name_locked=True, suggest=None)
                if v.get("voice_id"):
                    lib = {x["id"]: x for x in speakers.load_voices()}
                    if lib.get(v["voice_id"], {}).get("name", "").casefold() != name.casefold():
                        v.update(voice_id=None, score=None)
        if body.get("remember") and not body.get("consent"):
            raise HTTPException(400, "remembering a voice needs the person's consent: tick the consent box")
        if body.get("remember") and v.get("name") and v.get("samples") and sid in spk:
            vv = speakers.add_voice_samples(v["name"], v["samples"], voice_id=v.get("voice_id"), source=f"meeting {mid}")
            v["voice_id"] = vv["id"]
        store.write_json(store.mdir(mid) / "speakers.json", spk)
    _regen(mid)
    return public_speakers(mid, store.read_json(store.mdir(mid) / "transcript.json"))


@app.put("/api/meetings/{mid}/segments/{idx}/speaker")
async def api_segment_speaker(mid: str, idx: int, request: Request):
    """Move one transcript line to another speaker (fixes a wrongly recognised line)."""
    _meeting(mid)
    sid = (await request.json()).get("speaker")
    d = store.mdir(mid)
    with _spk_lock, store.transcript_lock(mid):
        tr, spk = store.read_json(d / "transcript.json"), store.speakers(mid)
        if not tr or not 0 <= idx < len(tr["segments"]) or sid not in spk:
            raise HTTPException(404, "line or participant not found")
        tr["segments"][idx]["spk"] = sid
        store.write_json(d / "transcript.json", tr)
        for k, v in spk.items():  # keep the per-speaker talk time in step with the lines
            v["talk_s"] = round(sum(g["t1"] - g["t0"] for g in tr["segments"] if g.get("spk") == k), 1)
        store.write_json(d / "speakers.json", spk)
    _regen(mid)
    return {"ok": True}


@app.post("/api/meetings/{mid}/speakers/match")
def api_speakers_match(mid: str):
    """Re-run voice matching against the enrolled voices (e.g. right after enrolling someone)."""
    _meeting(mid)
    with _spk_lock:
        spk = store.speakers(mid)
        speakers.apply_matches(spk, store.settings()["voice_match_threshold"])
        store.write_json(store.mdir(mid) / "speakers.json", spk)
    _regen(mid)
    return public_speakers(mid, store.read_json(store.mdir(mid) / "transcript.json"))


def install_loop_watchdog():
    """Safety net: the asyncio logger is rate-limited (a flood can fill the disk in minutes), and if the 'callback None()' spin ever happens
    again the server exits with code 75 (run.sh restarts it) - but only when nothing is being recorded."""
    import logging

    class Guard(logging.Filter):
        n, t0 = 0, time.time()

        def filter(self, rec):
            if "Exception in callback None()" not in rec.getMessage():
                return True
            now = time.time()
            if now - Guard.t0 > 10:
                Guard.n, Guard.t0 = 0, now
            Guard.n += 1
            if Guard.n == 200:
                if active:
                    log.error("event loop is spinning, but a meeting is being recorded: not restarting")
                else:
                    log.error("event loop is spinning: restarting the server")
                    logging.shutdown()
                    os._exit(75)
            return Guard.n <= 5

    logging.getLogger("asyncio").addFilter(Guard())


def main():
    import uvicorn

    store.init()
    kw = {}
    cert, key = HERE / "certs/cert.pem", HERE / "certs/key.pem"
    if cert.exists() and key.exists():
        kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    print(f"Secrets on https://{config.HOST}:{config.PORT}" + ("" if auth.has_users() else f"  first-admin setup code: {token()}"), flush=True)
    try:
        import uvloop  # noqa: F401 - its own TLS code: Python 3.12's asyncio SSL once spun the event loop at 100 % and wrote 5.7 GB of "Exception in callback None()" to the log

        kw["loop"] = "uvloop"
    except ImportError:
        pass
    install_loop_watchdog()
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning", **kw)


if __name__ == "__main__":
    main()
