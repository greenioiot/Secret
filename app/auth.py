"""Local accounts and role-based access (Admin / Member / Viewer).

data/users.json (0600)    : accounts, scrypt password hashes, role, enabled, session version
data/sessions.json (0600) : signed-in sessions (only a SHA-256 of the cookie value is stored), 8 h absolute lifetime

Rules (enforced by server.require_auth for HTTP and server.ws_user for WebSockets):
- every signed-in user may read every meeting (no per-meeting read ACL)
- member/admin may create meetings; a new meeting belongs to its creator (state.owner)
- only the owner, while still member/admin, may change, process, record into or delete a meeting; another admin cannot override it
- meetings without an owner (made before accounts existed) stay read-only until an admin assigns an eligible owner; an owned
  meeting is never reassigned
- admin manages accounts, roles and the shared settings/services (vocabulary, word collection, voice library, e-mail, bot, ...)
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import os
import re
import secrets
import threading
import time

import config
import store

USERS_FILE = config.DATA / "users.json"
SESSIONS_FILE = config.DATA / "sessions.json"
ROLES = ("admin", "member", "viewer")
SESSION_S = 8 * 3600
MIN_PASSWORD = 12
COOKIE = "secret_session"
CSRF_COOKIE = "secret_csrf"
_lock = threading.RLock()
_fails: dict[str, list[float]] = {}  # "ip:<addr>" / "user:<name>" -> failure times (login throttling, in memory)
FAIL_WINDOW_S = 15 * 60
FAIL_MAX_USER = 5
FAIL_MAX_IP = 20


# ---------------------------------------------------------------- storage
_cache: dict = {}  # path -> (mtime_ns, size, obj): every request checks the session, so do not parse the files each time


def _read(path, default):
    try:
        stt = path.stat()
    except FileNotFoundError:
        return default
    c = _cache.get(path)
    if not c or c[:2] != (stt.st_mtime_ns, stt.st_size):
        c = (stt.st_mtime_ns, stt.st_size, store.read_json(path) or default)
        _cache[path] = c
    return copy.deepcopy(c[2])


def _write(path, obj):
    store.write_json(path, obj)
    os.chmod(path, 0o600)


def users() -> dict:
    return _read(USERS_FILE, {}).get("users", {})


def _save_users(u: dict):
    _write(USERS_FILE, {"users": u})


def has_users() -> bool:
    return bool(users())


def public(name: str, u: dict) -> dict:
    return {"username": name, "name": u.get("name") or name, "role": u["role"], "enabled": u.get("enabled", True),
            "created": u.get("created"), "last_login": u.get("last_login")}


def list_users() -> list[dict]:
    return sorted((public(n, u) for n, u in users().items()), key=lambda x: x["username"])


def can_own(username: str | None) -> bool:
    u = users().get(username or "")
    return bool(u and u.get("enabled", True) and u["role"] in ("admin", "member"))


# ---------------------------------------------------------------- passwords
def hash_pw(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt$14$8$1${salt.hex()}${h.hex()}"


def check_pw(pw: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** int(n), r=int(r), p=int(p), dklen=len(h) // 2)
        return hmac.compare_digest(got.hex(), h)
    except (ValueError, TypeError):
        return False


_DUMMY = hash_pw(secrets.token_hex(8))  # unknown user names cost the same time as wrong passwords


def password_problem(pw: str) -> str | None:
    if len(pw) < MIN_PASSWORD:
        return f"the password needs at least {MIN_PASSWORD} characters"
    if len(pw) > 256:
        return "the password is too long"
    return None


def clean_username(raw: str) -> str:
    u = str(raw or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,31}", u):
        raise ValueError("user name: 2-32 characters, letters a-z, digits, . _ - (starts with a letter or digit)")
    return u


def clean_name(raw: str, fallback: str) -> str:
    n = re.sub(r"[\x00-\x1f<>]", "", str(raw or "")).strip()[:60]
    return n or fallback


# ---------------------------------------------------------------- accounts
def create_user(username: str, password: str, role: str, name: str = "") -> dict:
    username = clean_username(username)
    if role not in ROLES:
        raise ValueError("role must be admin, member or viewer")
    p = password_problem(password)
    if p:
        raise ValueError(p)
    with _lock:
        u = users()
        if username in u:
            raise ValueError("this user name already exists")
        u[username] = {"name": clean_name(name, username), "role": role, "enabled": True, "pw": hash_pw(password),
                       "created": time.strftime("%F %T"), "ver": 1}
        _save_users(u)
        return public(username, u[username])


def update_user(username: str, actor: str, *, role=None, enabled=None, name=None, password=None) -> dict:
    with _lock:
        u = users()
        if username not in u:
            raise KeyError(username)
        rec = u[username]
        revoke = False
        if role is not None and role != rec["role"]:
            if role not in ROLES:
                raise ValueError("role must be admin, member or viewer")
            if rec["role"] == "admin" and _admins(u) <= 1:
                raise ValueError("there must always be at least one enabled admin")
            rec["role"], revoke = role, True
        if enabled is not None and bool(enabled) != rec.get("enabled", True):
            if not enabled and rec["role"] == "admin" and _admins(u) <= 1:
                raise ValueError("there must always be at least one enabled admin")
            if not enabled and username == actor:
                raise ValueError("you cannot disable your own account")
            rec["enabled"], revoke = bool(enabled), True
        if name is not None:
            rec["name"] = clean_name(name, username)
        if password is not None:
            p = password_problem(password)
            if p:
                raise ValueError(p)
            rec["pw"], revoke = hash_pw(password), True
        if revoke:
            rec["ver"] = rec.get("ver", 1) + 1  # every session of this user ends
        _save_users(u)
        return public(username, rec)


THEMES = ("auto", "light", "dark")  # auto = follow the device; "light" is shown as White
ADMIN_THEMES = THEMES + ("gray",)


def themes_for(role: str) -> tuple:
    return ADMIN_THEMES if role == "admin" else THEMES


def theme_of(username: str | None, role: str) -> str:
    """The account's colour theme; a gray theme left over from a former admin falls back to auto."""
    t = (users().get(username or "") or {}).get("theme", "auto")
    return t if t in themes_for(role) else "auto"


def set_theme(username: str, role: str, theme: str) -> str:
    if theme not in themes_for(role):
        raise ValueError("the Gray theme is for administrators" if theme in ADMIN_THEMES else "unknown theme")
    with _lock:
        u = users()
        if username not in u:
            raise KeyError("this account cannot keep a theme")
        u[username]["theme"] = theme
        _save_users(u)
    return theme


def _admins(u: dict) -> int:
    return sum(1 for r in u.values() if r["role"] == "admin" and r.get("enabled", True))


def delete_user(username: str, actor: str):
    with _lock:
        u = users()
        if username not in u:
            raise KeyError(username)
        if username == actor:
            raise ValueError("you cannot delete your own account")
        if u[username]["role"] == "admin" and _admins(u) <= 1:
            raise ValueError("there must always be at least one enabled admin")
        del u[username]
        _save_users(u)
    revoke_user(username)


# ---------------------------------------------------------------- login throttling
def _recent(key: str) -> list[float]:
    now = time.time()
    lst = [t for t in _fails.get(key, []) if now - t < FAIL_WINDOW_S]
    _fails[key] = lst
    return lst


def throttled(ip: str, username: str) -> int:
    """Seconds to wait before another attempt is allowed (0 = go ahead)."""
    for key, mx in ((f"user:{username}", FAIL_MAX_USER), (f"ip:{ip}", FAIL_MAX_IP)):
        lst = _recent(key)
        if len(lst) >= mx:
            return int(FAIL_WINDOW_S - (time.time() - lst[-mx])) + 1
    return 0


def note_failure(ip: str, username: str):
    for key in (f"user:{username}", f"ip:{ip}"):
        _recent(key).append(time.time())


def authenticate(username: str, password: str) -> str | None:
    """Returns the user name when the password is right and the account enabled."""
    try:
        username = clean_username(username)
    except ValueError:
        check_pw(password, _DUMMY)
        return None
    rec = users().get(username)
    ok = check_pw(password, rec["pw"] if rec else _DUMMY)
    if not (rec and ok and rec.get("enabled", True)):
        return None
    _fails.pop(f"user:{username}", None)
    with _lock:
        u = users()
        u[username]["last_login"] = time.strftime("%F %T")
        _save_users(u)
    return username


# ---------------------------------------------------------------- sessions
def _sessions() -> dict:
    return _read(SESSIONS_FILE, {})


def _key(sid: str) -> str:
    return hashlib.sha256(sid.encode()).hexdigest()


def new_session(username: str, ip: str) -> tuple[str, str]:
    sid, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    with _lock:
        s = {k: v for k, v in _sessions().items() if v["expires"] > time.time()}
        s[_key(sid)] = {"user": username, "csrf": csrf, "created": time.time(), "expires": time.time() + SESSION_S,
                        "ver": users()[username].get("ver", 1), "ip": ip}
        _write(SESSIONS_FILE, s)
    return sid, csrf


def session(sid: str | None) -> dict | None:
    """The signed-in user for a session cookie: {user, name, role, csrf} or None (expired, revoked, disabled, role changed)."""
    if not sid:
        return None
    s = _sessions().get(_key(sid))
    if not s or s["expires"] < time.time():
        return None
    rec = users().get(s["user"])
    if not rec or not rec.get("enabled", True) or rec.get("ver", 1) != s["ver"]:
        return None
    return {"user": s["user"], "name": rec.get("name") or s["user"], "role": rec["role"], "csrf": s["csrf"], "expires": s["expires"]}


def end_session(sid: str | None):
    if not sid:
        return
    with _lock:
        s = _sessions()
        if s.pop(_key(sid), None) is not None:
            _write(SESSIONS_FILE, s)


def revoke_user(username: str):
    with _lock:
        s = {k: v for k, v in _sessions().items() if v["user"] != username and v["expires"] > time.time()}
        _write(SESSIONS_FILE, s)


# ---------------------------------------------------------------- meeting ownership
def may_edit_meeting(user: dict | None, st: dict | None) -> bool:
    return bool(user and st and user["role"] in ("admin", "member") and st.get("owner") == user["user"])


def edit_problem(user: dict | None, st: dict) -> str | None:
    if not user or user["role"] == "viewer":
        return "your account is read-only (Viewer)"
    if not st.get("owner"):
        return "this meeting has no owner yet: an admin must assign one before it can be changed"
    if st.get("owner") != user["user"]:
        return f"only the meeting's owner ({st['owner']}) can change it"
    return None
