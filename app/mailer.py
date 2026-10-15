"""Email the meeting summary through the user's own SMTP account (opt-in: nothing is sent unless Settings -> Email is enabled).

* The account password lives only in data/smtp.json (mode 0600). The API never returns it.
* Connections are encrypted (SSL/TLS or STARTTLS) and the server certificate is verified unless the user turns that off.
  A password is never sent over an unencrypted connection.
* Everything that comes from the meeting (title, transcript, AI text) is HTML-escaped; addresses cannot carry header injections.
* The audio recording is never attached.
"""
from __future__ import annotations

import html
import os
import re
import smtplib
import socket
import ssl
import time
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid

import config
import store

CFG_FILE = config.DATA / "smtp.json"
DEFAULTS = {"enabled": False, "host": "", "port": 587, "security": "starttls", "username": "", "password": "", "from_name": "Zecrets",
            "from_addr": "", "default_to": [], "verify_cert": True, "auto_send": False,
            "attach_summary": True, "attach_report": True, "attach_transcript": False}
SECURITY = ("starttls", "ssl", "none")
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+$")
TIMEOUT = 25
MAX_TO = 20


# ------------------------------------------------------------------ configuration
def load() -> dict:
    return {**DEFAULTS, **(store.read_json(CFG_FILE, {}) or {})}


def public(cfg: dict | None = None) -> dict:
    cfg = dict(cfg or load())
    cfg["password_set"] = bool(cfg.pop("password", ""))
    return cfg


def parse_recipients(raw) -> list[str]:
    items = re.split(r"[,;\s]+", raw) if isinstance(raw, str) else [str(x) for x in (raw or [])]
    out, bad = [], []
    for x in items:
        x = x.strip()
        if not x:
            continue
        if EMAIL_RE.match(x) and len(x) <= 254:
            if x.lower() not in [o.lower() for o in out]:
                out.append(x)
        else:
            bad.append(x[:40])
    if bad:
        raise ValueError("Not a valid e-mail address: " + ", ".join(bad))
    if len(out) > MAX_TO:
        raise ValueError(f"At most {MAX_TO} recipients.")
    return out


def save(new: dict) -> dict:
    cur = load()
    for k, d in DEFAULTS.items():
        if k not in new or k == "password":
            continue
        v = new[k]
        if isinstance(d, bool):
            cur[k] = bool(v)
        elif k == "port":
            cur[k] = int(v)
            if not 1 <= cur[k] <= 65535:
                raise ValueError("Port must be between 1 and 65535.")
        elif k == "default_to":
            cur[k] = parse_recipients(v)
        elif k == "security":
            if v not in SECURITY:
                raise ValueError("Unknown security mode.")
            cur[k] = v
        else:
            cur[k] = " ".join(str(v).split())[:200]
    if new.get("clear_password"):
        cur["password"] = ""
    elif isinstance(new.get("password"), str) and new["password"]:
        cur["password"] = new["password"][:300]   # kept as typed: only an empty value means "unchanged"
    if cur["from_addr"] and not EMAIL_RE.match(cur["from_addr"]):
        raise ValueError("The From address is not a valid e-mail address.")
    if cur["enabled"] and not (cur["host"] and (cur["from_addr"] or cur["username"])):
        raise ValueError("Enter the server and the From address before turning e-mail on.")
    config.DATA.mkdir(parents=True, exist_ok=True)
    tmp = CFG_FILE.with_name(CFG_FILE.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # private from the first byte
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        import json

        f.write(json.dumps(cur, ensure_ascii=False, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CFG_FILE)
    os.chmod(CFG_FILE, 0o600)
    return cur


# ------------------------------------------------------------------ sending
def friendly_error(e: Exception, cfg: dict) -> str:
    where = f"{cfg.get('host')}:{cfg.get('port')}"
    if isinstance(e, smtplib.SMTPAuthenticationError):
        return "The server rejected the user name or password. Gmail, Outlook and Yahoo usually need an app password (or SMTP sign-in enabled), not your normal password."
    if isinstance(e, ssl.SSLCertVerificationError):
        return "The server's certificate could not be verified. Check the server name; untick “Verify certificate” only if you trust this server (for example a private server)."
    if isinstance(e, ssl.SSLError):
        return "The secure connection failed. Try the other mode: SSL/TLS is normally port 465, STARTTLS port 587."
    if isinstance(e, (socket.gaierror,)):
        return f"The server name “{cfg.get('host')}” was not found. Check the spelling and the network."
    if isinstance(e, (ConnectionRefusedError, smtplib.SMTPConnectError)):
        return f"Could not connect to {where}. Check the port and that this box can reach the internet or the mail server."
    if isinstance(e, (TimeoutError, socket.timeout)):
        return f"Timed out talking to {where}. The port may be blocked, or the security mode may not match the port."
    if isinstance(e, smtplib.SMTPRecipientsRefused):
        return "The server refused the recipient address(es): " + ", ".join(e.recipients) if e.recipients else "The server refused the recipients."
    if isinstance(e, smtplib.SMTPSenderRefused):
        return "The server refused the From address. Use an address your account is allowed to send from."
    if isinstance(e, smtplib.SMTPServerDisconnected):
        return "The server closed the connection. The security mode probably does not match the port (SSL/TLS 465, STARTTLS 587)."
    if isinstance(e, smtplib.SMTPNotSupportedError):
        return "The server does not support this security mode or sign-in."
    if isinstance(e, smtplib.SMTPException):
        return "The mail server said: " + str(e)[:160]
    if isinstance(e, ValueError):
        return str(e)
    return f"{type(e).__name__}: {str(e)[:160]}"


def _connect(cfg: dict) -> smtplib.SMTP:
    if not cfg["host"]:
        raise ValueError("No SMTP server is set.")
    if cfg["username"] and cfg["security"] == "none":
        raise ValueError("Signing in over an unencrypted connection is not allowed: choose STARTTLS or SSL/TLS.")
    ctx = ssl.create_default_context()
    if not cfg["verify_cert"]:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    host, port = cfg["host"], int(cfg["port"])
    if cfg["security"] == "ssl":
        s: smtplib.SMTP = smtplib.SMTP_SSL(host, port, timeout=TIMEOUT, context=ctx)
    else:
        s = smtplib.SMTP(host, port, timeout=TIMEOUT)
    try:
        s.ehlo()
        if cfg["security"] == "starttls":
            if not s.has_extn("starttls"):
                raise smtplib.SMTPNotSupportedError("STARTTLS is not offered by this server")
            s.starttls(context=ctx)
            s.ehlo()
        if cfg["username"]:
            s.login(cfg["username"], cfg["password"])
    except Exception:
        try:
            s.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    return s


def send(to: list[str], subject: str, text: str, html_body: str | None = None, attachments: list[tuple[str, bytes]] | None = None, cfg: dict | None = None):
    cfg = cfg or load()
    to = parse_recipients(to)
    if not to:
        raise ValueError("Add at least one recipient.")
    sender = cfg["from_addr"] or cfg["username"]
    if not EMAIL_RE.match(sender or ""):
        raise ValueError("Set a valid From address in Settings → Email.")
    msg = EmailMessage()
    msg["Subject"] = " ".join(subject.split())[:200]          # no line breaks: no header injection
    msg["From"] = formataddr((" ".join((cfg["from_name"] or "Zecrets").split())[:60], sender))
    msg["To"] = ", ".join(to)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1])
    msg.set_content(text)
    if html_body:
        msg.add_alternative(html_body, subtype="html")
    for name, data in attachments or []:
        msg.add_attachment(data, maintype="text", subtype="markdown", filename=name, cte="8bit") if name.endswith(".md") else \
            msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=name)
    s = _connect(cfg)
    try:
        s.send_message(msg, from_addr=sender, to_addrs=to)
    finally:
        try:
            s.quit()
        except Exception:  # noqa: BLE001
            s.close()


def send_test(to: str) -> str:
    cfg = load()
    rcpt = parse_recipients(to)
    send(rcpt, "Zecrets: test e-mail", "This is a test message from Zecrets.\nIf you can read it, your e-mail settings work.\n\nนี่คือข้อความทดสอบจาก Zecrets\n",
         f"<p>This is a test message from <b>Zecrets</b>.<br>If you can read it, your e-mail settings work.</p><p>นี่คือข้อความทดสอบจาก Zecrets</p>", cfg=cfg)
    return ", ".join(rcpt)


# ------------------------------------------------------------------ the meeting e-mail
def _md_to_html(md: str) -> str:
    out, ul = [], False
    for ln in md.splitlines():
        s = ln.rstrip()
        if re.match(r"^\s*[-*] ", s) or re.match(r"^\s*\d+[.)] ", s):
            if not ul:
                out.append("<ul style='margin:4px 0;padding-left:22px'>")
                ul = True
            out.append("<li>" + html.escape(re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", s)) + "</li>")
            continue
        if ul:
            out.append("</ul>")
            ul = False
        if s.startswith("#"):
            out.append("<h3 style='margin:16px 0 4px;font-size:16px'>" + html.escape(s.lstrip("# ").strip()) + "</h3>")
        elif s.strip() and not s.startswith(">"):
            out.append("<p style='margin:4px 0'>" + html.escape(s.strip()) + "</p>")
    if ul:
        out.append("</ul>")
    return "\n".join(out)


def build_meeting(mid: str, opts: dict | None = None) -> tuple[str, str, str, list[tuple[str, bytes]]]:
    cfg = load()
    opts = {k: cfg[k] for k in ("attach_summary", "attach_report", "attach_transcript")} | (opts or {})
    st = store.load_state(mid) or {}
    d = store.mdir(mid)
    summary = (d / "summary.md").read_text(encoding="utf-8") if (d / "summary.md").exists() else ""
    rep = store.read_json(d / "report.json") or {}
    if not summary and not rep:
        raise ValueError("This meeting has no summary yet.")
    title = st.get("title", "Meeting")
    meta = [f"วันที่: {st.get('created', '').replace('T', ' ')}", f"ระยะเวลา: {store.fmt_ts(st.get('duration_s', 0))}"]
    if st.get("participants"):
        meta.append("ผู้เข้าร่วม: " + ", ".join(st["participants"]))
    if st.get("remark"):
        meta.append("หมายเหตุ: " + " ".join(st["remark"].split()))
    body_md = re.sub(r"^# .*\n+", "", summary, count=1)
    rows = ""
    txt_people = ""
    if rep.get("people"):
        import report as _r

        for p in rep["people"]:
            rows += (f"<tr><td style='padding:4px 10px;border-bottom:1px solid #e3e6eb'>{html.escape(p['name'])}</td>"
                     f"<td style='padding:4px 10px;border-bottom:1px solid #e3e6eb'>{_r.mmss(p['talk_s'])}</td>"
                     f"<td style='padding:4px 10px;border-bottom:1px solid #e3e6eb'>{p['share'] * 100:.0f}%</td>"
                     f"<td style='padding:4px 10px;border-bottom:1px solid #e3e6eb'>{p['turns']}</td>"
                     f"<td style='padding:4px 10px;border-bottom:1px solid #e3e6eb'>{p['questions']}</td></tr>")
            txt_people += f"  - {p['name']}: {_r.mmss(p['talk_s'])} นาที ({p['share'] * 100:.0f}%), {p['turns']} ครั้ง, {p['questions']} คำถาม\n"
    ov = rep.get("overview_text", "")
    note = "สรุปนี้สร้างโดย AI บนอุปกรณ์ในห้องประชุม อาจมีข้อผิดพลาด โปรดตรวจสอบกับบันทึกถอดเสียง / AI-generated, may contain mistakes."
    text = f"{title}\n" + "\n".join(meta) + "\n\n" + (body_md.strip() + "\n\n" if body_md.strip() else "")
    if txt_people:
        text += "การมีส่วนร่วม\n" + (ov + "\n" if ov else "") + txt_people + "\n"
    text += note + "\n"
    h = ("<div style=\"font-family:Arial,'Noto Sans Thai',sans-serif;font-size:15px;color:#1c2430;max-width:720px\">"
         f"<h2 style='margin:0 0 6px'>{html.escape(title)}</h2><div style='color:#667085;font-size:13px'>" + "<br>".join(html.escape(m) for m in meta) + "</div>"
         + _md_to_html(body_md))
    if rows:
        h += ("<h3 style='margin:18px 0 4px;font-size:16px'>การมีส่วนร่วม</h3>" + (f"<p style='margin:4px 0'>{html.escape(ov)}</p>" if ov else "")
              + "<table style='border-collapse:collapse;font-size:14px'><tr style='color:#667085;text-align:left'><th style='padding:4px 10px'>ชื่อ</th><th style='padding:4px 10px'>เวลาพูด</th>"
                "<th style='padding:4px 10px'>สัดส่วน</th><th style='padding:4px 10px'>ครั้ง</th><th style='padding:4px 10px'>คำถาม</th></tr>" + rows + "</table>")
    h += f"<p style='color:#667085;font-size:12px;margin-top:20px'>{html.escape(note)}</p></div>"
    att: list[tuple[str, bytes]] = []
    for flag, fname, out in (("attach_summary", "summary.md", f"{mid}-summary.md"), ("attach_report", "report.md", f"{mid}-report.md"),
                             ("attach_transcript", "transcript.md", f"{mid}-transcript.md")):
        if opts.get(flag) and (d / fname).exists():
            att.append((out, (d / fname).read_bytes()))
    return f"[Zecrets] {title}", text, h, att


def send_meeting(mid: str, to, opts: dict | None = None, auto: bool = False) -> list[str]:
    cfg = load()
    rcpt = parse_recipients(to)
    subject, text, h, att = build_meeting(mid, opts)
    entry = {"time": time.strftime("%F %T"), "to": rcpt, "attachments": [a[0].split("-", 2)[-1] for a in att], "auto": auto}
    try:
        send(rcpt, subject, text, h, att, cfg=cfg)
        entry["ok"] = True
        return rcpt
    except Exception as e:  # noqa: BLE001
        entry.update(ok=False, error=friendly_error(e, cfg))
        raise
    finally:
        st = store.load_state(mid) or {}
        store.update_state(mid, email_log=((st.get("email_log") or []) + [entry])[-30:])
        store.log_event(mid, f"e-mail to {len(rcpt)} recipient(s): " + ("sent" if entry["ok"] else "FAILED"))
