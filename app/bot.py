"""Meeting bot: conducts a meeting from an agenda (configurable), keeps time, watches the flow, answers questions from the knowledge base (RAG).

Two layers, so it stays useful when the device is short of memory:
  * RULES (no language model, instant, ~0 MB): opening, agenda items and timers, "time is nearly up" / "time is up", silence prompts,
    nudges for enrolled people who have not spoken, drift to another agenda item (embeddings), commitments spotted by keywords,
    extractive answers (the best passage of the knowledge base).
  * LLM (Typhoon 4B through llama.cpp, started only when needed and only if capacity.plan() says it fits, stopped when the meeting ends):
    answers in a sentence or two from retrieved passages + the latest minutes, recaps, action items, closing recap.
Everything the bot says goes to the host screen, to joined guests, and to data/meetings/<id>/bot.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import capacity
import config
import kb
import store

log = logging.getLogger("bot")
BOT_FILE = config.DATA / "bot.json"
LLM_PORT = config.LLM_PORT + 1  # the summary worker uses config.LLM_PORT: never share a port

PRESETS = {
    "facilitator": {"label": "Facilitator · ผู้ดำเนินการประชุม", "info": "Balanced: keeps to the agenda, invites quiet people, notes commitments, short recaps.", "style": "neutral",
                    "timekeeping": True, "silence_s": 60, "balance": True, "actions": True, "offtopic": True, "recap_min": 10, "cooldown_s": 25},
    "timekeeper": {"label": "Strict timekeeper · ผู้คุมเวลา", "info": "Watches the clock: warns early, says plainly when time is up, no small talk.", "style": "strict",
                   "timekeeping": True, "silence_s": 0, "balance": False, "actions": True, "offtopic": True, "recap_min": 0, "cooldown_s": 15},
    "brainstorm": {"label": "Brainstorm coach · ผู้กระตุ้นความคิด", "info": "Encourages ideas, fills silences, invites everyone, summarises themes.", "style": "encourage",
                   "timekeeping": True, "silence_s": 40, "balance": True, "actions": True, "offtopic": False, "recap_min": 8, "cooldown_s": 25},
    "standup": {"label": "Daily stand-up · ประชุมสั้นประจำวัน", "info": "Short and brisk: makes sure everyone speaks, flags blockers and owners.", "style": "strict",
                "timekeeping": True, "silence_s": 30, "balance": True, "actions": True, "offtopic": False, "recap_min": 0, "cooldown_s": 20},
    "decision": {"label": "Decision meeting · ประชุมตัดสินใจ", "info": "Drives to a decision: options, owner, deadline; recaps often.", "style": "neutral",
                 "timekeeping": True, "silence_s": 60, "balance": True, "actions": True, "offtopic": True, "recap_min": 6, "cooldown_s": 25},
}
DEFAULTS = {"enabled": False, "name": "น้องเลขา", "preset": "facilitator", "goal": "", "agenda": [], "rules": "", "mode": "proactive", "llm": "auto", "guest_questions": False,
            "timekeeping": True, "silence_s": 60, "balance": True, "actions": True, "offtopic": True, "recap_min": 10, "cooldown_s": 25, "closing": True, "speak_name_triggers": True}
STYLE = {  # phrases per style: {n} = bot name
    "neutral": {"time80": "เหลือเวลาประมาณ {m} นาทีสำหรับ “{t}” ครับ", "time100": "หมดเวลาที่วางไว้สำหรับ “{t}” แล้วครับ {nxt}",
                "over": "หัวข้อ “{t}” เกินเวลามา {m} นาทีแล้วครับ {nxt}", "silence": "เงียบไปสักพักแล้วครับ มีใครมีความเห็นเรื่อง “{t}” เพิ่มเติมไหมครับ หรือจะไปหัวข้อถัดไปดี"},
    "strict": {"time80": "เหลือ {m} นาทีสำหรับ “{t}” กรุณาสรุปประเด็นครับ", "time100": "หมดเวลา “{t}” แล้วครับ {nxt}",
               "over": "“{t}” เกินเวลา {m} นาทีแล้ว ขอปิดหัวข้อนี้ครับ {nxt}", "silence": "ยังเงียบอยู่ครับ ถ้าไม่มีประเด็นเพิ่มเติมเรื่อง “{t}” ขอไปต่อครับ"},
    "encourage": {"time80": "ยังมีเวลาอีก {m} นาทีสำหรับ “{t}” ลองแลกเปลี่ยนไอเดียเพิ่มได้เลยครับ", "time100": "ครบเวลาของ “{t}” แล้วครับ ขอบคุณสำหรับไอเดียทุกอย่าง {nxt}",
                  "over": "“{t}” เกินเวลามา {m} นาทีแล้ว ไอเดียสุดท้ายแล้วค่อยไปต่อนะครับ {nxt}", "silence": "ไม่มีอะไรผิดนะครับ ไอเดียเล็กๆ ก็มีค่า ใครอยากเสนอเรื่อง “{t}” ไหมครับ"},
}
COMMIT = re.compile(r"(ผม|ฉัน|ดิฉัน|หนู|เดี๋ยว)\s*(จะ|ขอ)?\s*(รับไป|ไปทำ|จัดการ|ส่ง|ติดตาม|ดำเนินการ|เตรียม|ประสาน|แจ้ง)|มอบหมายให้|รับผิดชอบ|ภายในวัน|ก่อนวัน|ภายในสัปดาห์|สัปดาห์หน้า|พรุ่งนี้")


def numbers_ok(out: str, source: str) -> bool:
    """The 4B model sometimes garbles figures ("500,000" -> "50"). Every multi-digit number it writes must appear in what it was given."""
    have = set(re.findall(r"\d+(?:\.\d+)?", source.replace(",", "")))
    return all(n in have for n in re.findall(r"\d+(?:\.\d+)?", out.replace(",", "")) if len(n.split(".")[0]) >= 2)


WARN_NUM = "\n(ตัวเลขในข้อความนี้ผ่าน AI อาจคลาดเคลื่อน โปรดตรวจสอบกับบันทึก)"


# ------------------------------------------------------------------ configuration
def clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, type(default)(v)))
    except (TypeError, ValueError):
        return default


def resolve(raw: dict | None, base: dict | None = None) -> dict:
    """Validate a bot configuration from the UI (anything unknown is dropped, numbers are clamped)."""
    raw = raw or {}
    cfg = {**DEFAULTS, **(base or {})}
    preset = raw.get("preset", cfg["preset"])
    if preset in PRESETS and ("preset" in raw and raw["preset"] != cfg.get("preset") or not base):
        cfg.update({k: v for k, v in PRESETS[preset].items() if k in DEFAULTS})  # a preset sets the toggles; explicit values below still win
    cfg["preset"] = preset if preset in PRESETS else "facilitator"
    for k in ("enabled", "timekeeping", "balance", "actions", "offtopic", "closing", "guest_questions", "speak_name_triggers"):
        if k in raw:
            cfg[k] = bool(raw[k])
    for k, lo, hi in (("silence_s", 0, 600), ("recap_min", 0, 60), ("cooldown_s", 5, 600)):
        if k in raw:
            cfg[k] = clamp(raw[k], lo, hi, DEFAULTS[k])
    if "name" in raw:
        cfg["name"] = " ".join(str(raw["name"]).split())[:30] or DEFAULTS["name"]
    if "goal" in raw:
        cfg["goal"] = str(raw["goal"]).strip()[:400]
    if "rules" in raw:
        cfg["rules"] = str(raw["rules"]).strip()[:1500]
    if raw.get("mode") in ("proactive", "on_request"):
        cfg["mode"] = raw["mode"]
    if raw.get("llm") in ("auto", "off"):
        cfg["llm"] = raw["llm"]
    if "agenda" in raw:
        items = []
        for it in (raw["agenda"] or [])[:200]:
            if len(items) >= 20:
                break
            if isinstance(it, dict) and str(it.get("title", "")).strip():
                items.append({"title": " ".join(str(it["title"]).split())[:120], "minutes": clamp(it.get("minutes", 0), 0, 240, 0), "note": str(it.get("note", "")).strip()[:300]})
        cfg["agenda"] = items
    return cfg


def load_store() -> dict:
    d = store.read_json(BOT_FILE, {}) or {}
    return {"defaults": resolve(d.get("defaults") or {}), "templates": d.get("templates") or []}


def save_store(defaults: dict | None = None, templates: list | None = None) -> dict:
    cur = load_store()
    if defaults is not None:
        cur["defaults"] = resolve(defaults, cur["defaults"])
        cur["defaults"]["enabled"] = False  # "enabled" is chosen per meeting, never a global default
    if templates is not None:
        out = []
        for t in templates[:30]:
            if isinstance(t, dict) and str(t.get("name", "")).strip():
                out.append({"name": " ".join(str(t["name"]).split())[:60], **{k: v for k, v in resolve(t).items() if k in ("preset", "goal", "agenda", "rules")}})
        cur["templates"] = out
    config.DATA.mkdir(parents=True, exist_ok=True)
    store.write_json(BOT_FILE, cur)
    return cur


# ------------------------------------------------------------------ the language model (one short-lived llama-server per live meeting)
class LlmHandle:
    def __init__(self, port: int = LLM_PORT):
        self.port, self.proc = port, None
        self.state, self.reason = "off", "not started"
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="botllm")  # one request at a time, first come first served
        self.pending = 0
        self.last_used = time.time()
        self._ban: list = []
        self.lock = threading.Lock()
        import atexit

        atexit.register(self.stop)  # whatever ends this process (a test, a crash handler), its model must not outlive it

    def command(self) -> list[str]:
        binary = config.LLAMA_BUILD / "bin/llama-server"
        return ["nice", "-n", "10", "taskset", "-c", ",".join(map(str, config.WORKER_CORES)), str(binary), "-m", str(config.LLM_GGUF), "--host", "127.0.0.1",
                "--port", str(self.port), "--no-webui", "-dev", "none", "-t", "4", "-c", "4096", "-np", "1"]

    def env(self) -> dict:
        return {**os.environ, "LD_LIBRARY_PATH": str(config.LLAMA_BUILD / "lib")}

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                return b"ok" in r.read()
        except (OSError, urllib.error.URLError):
            return False

    def stale_pids(self) -> list[int]:
        """Processes that ARE a llama-server of ours on this port: the program itself is `llama-server` (not a shell or editor that merely
        mentions it), with `--port <our port>` and our model file as separate arguments."""
        me, out = os.getuid(), []
        for name in os.listdir("/proc"):
            if not name.isdigit() or int(name) == os.getpid():
                continue
            try:
                if os.stat(f"/proc/{name}").st_uid != me:
                    continue
                argv = open(f"/proc/{name}/cmdline", "rb").read().decode("utf-8", "replace").split("\0")
            except OSError:
                continue
            if argv and os.path.basename(argv[0]) == "llama-server" and "--port" in argv and argv[argv.index("--port") + 1:argv.index("--port") + 2] == [str(self.port)] \
                    and str(config.LLM_GGUF) in argv and int(name) != (self.proc.pid if self.proc else -1):
                out.append(int(name))
        return out

    def _reap_stale(self):
        """A previous Secrets server that was killed hard may have left ITS bot model on this port: stop that one (only this port, only exact matches)."""
        for pid in self.stale_pids():
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        if self.stale_pids():
            time.sleep(1.0)

    def start(self) -> bool:
        """Blocking (run it in an executor). True when the model answers."""
        with self.lock:
            if self.proc and self.proc.poll() is None and self._healthy():
                self.state = "ready"
                return True
            if not (config.LLAMA_BUILD / "bin/llama-server").exists() or not config.LLM_GGUF.exists():
                self.state, self.reason = "off", "the language model files were not found"
                return False
            self.state, self.reason = "loading", "starting the language model…"
            self._reap_stale()
            logf = open(config.DATA / "bot-llm.log", "ab")
            self.proc = subprocess.Popen(self.command(), stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self.env())
            t = time.time()
            while time.time() - t < 180:
                if self.proc.poll() is not None:
                    self.state, self.reason = "off", f"the language model stopped (exit {self.proc.returncode})"
                    return False
                if self._healthy():
                    self._ban = self._ban_tokens()
                    self.state, self.reason = "ready", ""
                    self.last_used = time.time()
                    return True
                time.sleep(0.5)
            self.stop()
            self.state, self.reason = "off", "the language model did not start in time"
            return False

    def stop(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
        if self.state != "off" or not self.reason:
            self.state, self.reason = "off", "stopped"

    def _ban_tokens(self) -> list:
        """This Typhoon/Qwen3 model sometimes answers a long persona prompt with the special <tool_call> token and stops (or loops on it). Ban those tokens."""
        ids = []
        try:
            for tok in ("<tool_call>", "</tool_call>", "<tool_response>"):
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}/tokenize", json.dumps({"content": tok, "special": True}).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    ids += json.loads(r.read())["tokens"]
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            pass
        return [[i, False] for i in ids]

    def _chat(self, system: str, user: str, max_tokens: int, temperature: float, prefill: str = "") -> str:
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{prefill}"
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature, "top_p": 0.9, "repeat_penalty": 1.1, "stop": ["<|im_end|>", "<|im_start|>"], "stream": False, "cache_prompt": True}
        if self._ban:
            body["logit_bias"] = self._ban
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            text = json.loads(r.read())["content"]
        self.last_used = time.time()
        import summarize

        return summarize.clean(text)

    async def chat(self, system: str, user: str, max_tokens: int = 240, temperature: float = 0.3, prefill: str = "") -> str | None:
        if self.state != "ready" or self.pending >= 3:
            return None
        self.pending += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self.pool, self._chat, system, user, max_tokens, temperature, prefill)
        except Exception as e:  # noqa: BLE001
            log.warning("bot LLM request failed: %s", e)
            return None
        finally:
            self.pending -= 1


# ------------------------------------------------------------------ the bot of one meeting
class MeetingBot:
    def __init__(self, mid: str, cfg: dict, emit, llm: LlmHandle | None = None, now=time.time, participants=lambda: [], enrolled=lambda: [], cap_settings=lambda: {}):
        self.mid, self.cfg, self.emit, self.llm, self.now = mid, cfg, emit, llm, now
        self.participants, self.enrolled, self.cap_settings = participants, enrolled, cap_settings
        self.started = now()
        self.msgs: list[dict] = []
        self.idx = -1
        self.item_start = self.started
        self.item_log: list[dict] = []
        self.caps: deque = deque(maxlen=600)
        self.talk: dict[str, float] = {}
        self.last_caption: float | None = None
        self.last_nudge = 0.0
        self.flags: dict = {}
        self.overruns = 0
        self.last_recap = self.started
        self.last_balance = self.started
        self.balance_n = 0
        self.last_actions_check = self.started
        self.actions: list[dict] = []
        self.candidates: list[dict] = []
        self.last_ask: dict[str, float] = {}
        self.asking = 0
        self.paused = False
        self.closed = False
        self.drift_hits = 0
        self.last_drift = self.started
        self._seq = 0
        self._llm_task: asyncio.Task | None = None
        self.llm_failed_at = -1e9

    # -- output
    async def say(self, kind: str, text: str, sources: list[str] | None = None, llm: bool = False, force: bool = False, by: str = "") -> dict | None:
        if self.paused and not force:
            return None
        if not force and self.cfg["mode"] != "proactive" and kind in ("time", "nudge", "silence", "drift", "recap", "action", "balance"):
            return None
        if not force and kind in ("time", "nudge", "silence", "drift", "balance", "action", "recap") and self.now() - self.last_nudge < self.cfg["cooldown_s"]:
            return None
        self._seq += 1
        msg = {"id": self._seq, "t": round(self.now() - self.started, 1), "kind": kind, "text": text, "sources": sources or [], "llm": llm, "by": by, "bot": self.cfg["name"]}
        self.msgs.append(msg)
        if kind in ("time", "nudge", "silence", "drift", "balance", "action", "recap"):
            self.last_nudge = self.now()
        try:
            with open(store.mdir(self.mid) / "bot.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        except OSError:
            pass
        try:
            await self.emit(msg)
        except Exception:  # noqa: BLE001 - a closed screen must never stop the bot
            pass
        return msg

    def can_nudge(self) -> bool:
        return self.now() - self.last_nudge >= self.cfg["cooldown_s"]

    # -- model access (lazy, only when capacity allows)
    def llm_wanted(self) -> bool:
        return self.cfg["llm"] == "auto" and self.llm is not None

    def llm_fits(self) -> bool:
        """Memory allows the language model, and a recent start has not just failed (no retry storm)."""
        if self.now() - self.llm_failed_at < 120:
            return False
        return capacity.plan(self.cap_settings(), recording=True, bot_llm=True)["bot_llm_ok"]

    async def ensure_llm(self) -> bool:
        if not self.llm_wanted():
            return False
        if self.llm.state == "ready":
            return True
        if self.llm.state == "loading" and self._llm_task:
            await asyncio.shield(self._llm_task)
            return self.llm.state == "ready"
        if not self.llm_fits():
            if self.llm.state != "loading":
                self.llm.state, self.llm.reason = "off", "not enough free memory (or it just failed): the bot works in simple mode"
            return False
        loop = asyncio.get_running_loop()
        self._llm_task = loop.run_in_executor(None, self.llm.start)
        await asyncio.shield(self._llm_task)
        if self.llm.state != "ready":
            self.llm_failed_at = self.now()
        return self.llm.state == "ready"

    # -- helpers
    @property
    def item(self) -> dict | None:
        ag = self.cfg["agenda"]
        return ag[self.idx] if 0 <= self.idx < len(ag) else None

    def recent_text(self, seconds: float = 240, max_chars: int = 1800) -> str:
        cutoff = self.now() - seconds
        lines = [f"{c['who']}: {c['text']}" if c["who"] else c["text"] for c in self.caps if c["at"] >= cutoff]
        return "\n".join(lines)[-max_chars:]

    def _next_hint(self) -> str:
        ag = self.cfg["agenda"]
        return f"ไปหัวข้อถัดไป “{ag[self.idx + 1]['title']}” ดีไหมครับ" if self.idx + 1 < len(ag) else "ใกล้ถึงช่วงสรุปแล้วครับ"

    def system_prompt(self) -> str:
        rules = f"\nกติกาเพิ่มเติมจากผู้จัด: {self.cfg['rules']}" if self.cfg["rules"] else ""
        goal = f"\nเป้าหมายการประชุม: {self.cfg['goal']}" if self.cfg["goal"] else ""
        return (f"คุณคือ “{self.cfg['name']}” ผู้ช่วยดำเนินการประชุมภาษาไทย พูดสุภาพ กระชับ ไม่เกิน 3 ประโยค ตอบเป็นภาษาไทย (ชื่อเฉพาะคงเดิม) "
                "ใช้เฉพาะข้อมูลจากเอกสารอ้างอิงและเนื้อหาการประชุมที่ให้ ห้ามเดาหรือแต่งข้อมูล ถ้าไม่มีข้อมูลให้บอกตรงๆ ว่าไม่พบข้อมูล "
                "คัดลอกตัวเลข วันที่ จำนวนเงินให้ตรงตัว" + goal + rules)

    # -- lifecycle
    async def start(self):
        c = self.cfg
        ag = c["agenda"]
        parts = [f"สวัสดีครับ ผมคือ{c['name']} ผู้ช่วยดำเนินการประชุม"]
        if c["goal"]:
            parts.append(f"เป้าหมายวันนี้: {c['goal']}")
        if ag:
            parts.append("วาระ: " + " / ".join(f"{i + 1}) {a['title']}" + (f" ({a['minutes']} นาที)" if a["minutes"] else "") for i, a in enumerate(ag)))
        await self.say("opening", "\n".join(parts), force=True)
        if ag:
            await self.goto(0, opening=True)

    async def goto(self, i: int, opening: bool = False, by: str = "host"):
        ag = self.cfg["agenda"]
        if not ag or not 0 <= i < len(ag):
            return None
        now = self.now()
        if self.item is not None and self.item_log and self.item_log[-1].get("end") is None:
            self.item_log[-1]["end"] = round(now - self.started, 1)
        self.idx, self.item_start, self.flags, self.overruns, self.drift_hits = i, now, {}, 0, 0
        it = ag[i]
        self.item_log.append({"index": i, "title": it["title"], "planned_min": it["minutes"], "start": round(now - self.started, 1), "end": None})
        self.save_snapshot()
        text = f"หัวข้อที่ {i + 1}/{len(ag)}: {it['title']}" + (f" (เวลา {it['minutes']} นาที)" if it["minutes"] else "") + (f"\n{it['note']}" if it["note"] else "")
        srcs: list[str] = []
        hits = await asyncio.get_running_loop().run_in_executor(None, lambda: kb.search_meeting(self.mid, it["title"] + " " + it["note"], 1))
        if hits:
            text += f"\nข้อมูลประกอบ ({hits[0]['name']}): " + re.sub(r"\s+", " ", hits[0]["text"])[:240]
            srcs = [hits[0]["name"]]
        return await self.say("agenda", text, srcs, force=True)

    async def next(self):
        if self.idx + 1 < len(self.cfg["agenda"]):
            return await self.goto(self.idx + 1)
        return await self.say("info", "นี่คือหัวข้อสุดท้ายแล้วครับ", force=True)

    async def prev(self):
        return await self.goto(max(0, self.idx - 1)) if self.idx > 0 else None

    def save_snapshot(self):
        try:
            store.write_json(store.mdir(self.mid) / "bot_state.json", {"idx": self.idx, "item_start": self.item_start, "started": self.started, "log": self.item_log, "actions": self.actions})
        except OSError:
            pass

    def restore(self) -> bool:
        s = store.read_json(store.mdir(self.mid) / "bot_state.json")
        if not s:
            return False
        self.idx, self.item_start, self.started = s["idx"], s["item_start"], s.get("started", self.started)
        self.item_log, self.actions = s.get("log", []), s.get("actions", [])
        self._seq = len(self.item_log) + 1000
        return True

    def feed(self, rec: dict):
        """A live caption arrived: {t0, t1, text, speaker}."""
        now = self.now()
        who = rec.get("speaker") or ""
        self.caps.append({"at": now, "who": who, "text": rec["text"]})
        self.last_caption = now
        self.flags.pop("silence", None)
        self.talk[who or "?"] = self.talk.get(who or "?", 0.0) + max(0.0, rec.get("t1", 0) - rec.get("t0", 0))
        if self.cfg["actions"] and COMMIT.search(rec["text"]):
            self.candidates.append({"t": round(now - self.started, 1), "who": who, "text": rec["text"][:160]})
        if self.cfg["speak_name_triggers"]:
            m = re.search(r"\s*".join(re.escape(ch) for ch in self.cfg["name"].replace(" ", "")), rec["text"])
            if m:
                q = rec["text"][m.end():].strip(" ,.ครับค่ะนะคะ")
                if len(q) >= 6:
                    asyncio.get_running_loop().create_task(self.ask(q, who or "ในห้องประชุม", quiet=True))

    # -- the rule engine
    async def tick(self):
        if self.closed or self.paused or self.cfg["mode"] != "proactive":
            return  # "on request" bots only answer; they never interrupt
        now, c = self.now(), self.cfg
        st = STYLE[PRESETS.get(c["preset"], PRESETS["facilitator"])["style"]]
        it = self.item
        if c["timekeeping"] and it and it["minutes"] > 0:
            budget, elapsed = it["minutes"] * 60, now - self.item_start
            left = max(0, round((budget - elapsed) / 60))
            if elapsed >= 0.8 * budget and budget >= 180 and "t80" not in self.flags and elapsed < budget:
                self.flags["t80"] = 1
                await self.say("time", st["time80"].format(m=max(1, left), t=it["title"]))
            elif elapsed >= budget and "t100" not in self.flags:
                self.flags["t100"] = 1
                await self.say("time", st["time100"].format(t=it["title"], nxt=self._next_hint()), force=True)
            elif elapsed >= budget + 180 * (self.overruns + 1) and self.overruns < 3 and "t100" in self.flags:
                self.overruns += 1
                await self.say("time", st["over"].format(t=it["title"], m=round((elapsed - budget) / 60), nxt=self._next_hint()))
        if c["silence_s"] and "silence" not in self.flags and it:
            ref = self.last_caption if self.last_caption is not None else self.started
            if now - ref > c["silence_s"] and now - self.item_start > c["silence_s"]:
                self.flags["silence"] = 1
                await self.say("silence", st["silence"].format(t=it["title"]))
        if c["balance"] and now - self.started > 300 and now - self.last_balance > 600 and self.balance_n < 3 and self.can_nudge():
            self.last_balance = now
            quiet = [n for n in self.enrolled() if n.casefold() in {p.casefold() for p in self.participants()} and not any(w.casefold() == n.casefold() for w in self.talk)]
            if quiet:
                self.balance_n += 1
                await self.say("balance", "ยังไม่ได้ฟังความเห็นจาก " + ", ".join(quiet[:4]) + " เลยครับ ขอเชิญแสดงความเห็นได้นะครับ")
        if c["offtopic"] and len(c["agenda"]) >= 2 and now - self.last_drift > 45:
            self.last_drift = now
            await self._drift()
        if self.candidates and c["actions"] and now - self.last_actions_check > 150 and self.can_nudge():
            self.last_actions_check = now
            await self._actions()
        if c["recap_min"] and now - self.last_recap > c["recap_min"] * 60 and self.recent_text(c["recap_min"] * 60) and self.can_nudge():
            self.last_recap = now
            await self._recap()

    async def _drift(self):
        text = self.recent_text(100, 700)
        if len(text) < 150 or not kb.EMB.available():
            return
        ag = self.cfg["agenda"]
        try:
            loop = asyncio.get_running_loop()
            v = await loop.run_in_executor(None, lambda: kb.EMB.embed([a["title"] + " " + a["note"] for a in ag] + [text], "passage"))
        except Exception:  # noqa: BLE001
            return
        sims = v[:-1] @ v[-1]
        best = int(sims.argmax())
        if best != self.idx and sims[best] - sims[self.idx] >= 0.03 and sims[best] >= 0.80:
            self.drift_hits += 1
        else:
            self.drift_hits = 0
        if self.drift_hits >= 2:
            self.drift_hits = 0
            await self.say("drift", f"ดูเหมือนตอนนี้กำลังคุยเรื่อง “{ag[best]['title']}” ซึ่งเป็นหัวข้อที่ {best + 1} ต้องการไปหัวข้อนั้นเลยไหมครับ (หรือกลับมาที่ “{ag[self.idx]['title']}”)")

    async def _actions(self):
        cands, self.candidates = self.candidates[:6], self.candidates[6:]
        if await self.ensure_llm():
            sys_ = self.system_prompt()
            txt = "\n".join(f"- {(c['who'] + ': ') if c['who'] else ''}{c['text']}" for c in cands)
            out = await self.llm.chat(sys_, "จากประโยคเหล่านี้ ระบุงานที่ต้องทำ แต่ละบรรทัดรูปแบบ: งาน | ผู้รับผิดชอบ | กำหนดเวลา (ถ้าไม่ระบุให้เขียน ไม่ระบุ) ถ้าไม่มีงานเลยให้ตอบว่า ไม่มี\n" + txt, 200, 0.1)
            if out and not numbers_ok(out, txt):
                out = None  # a garbled figure in a task is worse than the plain keyword suggestion below
            if out and not out.strip().startswith("ไม่มี"):
                found = []
                for ln in out.splitlines():
                    p = [x.strip(" -•*") for x in ln.split("|")]
                    if len(p) >= 2 and p[0] and len(p[0]) > 3:
                        self.actions.append({"text": p[0], "owner": p[1], "due": p[2] if len(p) > 2 else "ไม่ระบุ", "t": round(self.now() - self.started), "source": "llm"})
                        found.append(f"• {p[0]} (ผู้รับผิดชอบ: {p[1]}, กำหนด: {p[2] if len(p) > 2 else 'ไม่ระบุ'})")
                if found:
                    await self.say("action", "จดงานที่ต้องทำไว้แล้วครับ:\n" + "\n".join(found[:4]), llm=True)
                    return
        shown = [f"• {(c['who'] + ': ') if c['who'] else ''}{c['text']}" for c in cands[:3]]
        for c in cands[:3]:
            self.actions.append({"text": c["text"], "owner": c["who"] or "ไม่ระบุ", "due": "ไม่ระบุ", "t": round(c["t"]), "source": "rule"})
        await self.say("action", "ผมจับได้ว่าอาจมีงานที่ต้องทำครับ:\n" + "\n".join(shown) + "\nกรุณายืนยันผู้รับผิดชอบและกำหนดเวลาด้วยครับ")

    async def _recap(self):
        if not await self.ensure_llm():
            return
        it = self.item
        src = self.recent_text(self.cfg["recap_min"] * 60, 2200)
        out = await self.llm.chat(self.system_prompt(), f"หัวข้อปัจจุบัน: {it['title'] if it else '-'}\nเนื้อหาที่คุยกันล่าสุด:\n{src}\n\nสรุปเป็นข้อสั้นๆ 2-3 ข้อ และบอกว่ามีประเด็นใดที่ยังไม่ได้ข้อสรุป", 200, prefill="สรุป:\n")
        if out:
            await self.say("recap", "สรุปสั้นๆ ที่คุยกันมา:\n" + out + ("" if numbers_ok(out, src) else WARN_NUM), llm=True)

    # -- questions (RAG)
    async def ask(self, question: str, who: str = "", quiet: bool = False) -> dict | None:
        question = " ".join(question.split())[:400]
        now = self.now()
        if self.closed or len(question) < 3:
            return None
        if who != "host" and now - self.last_ask.get(who, 0) < 20:  # the host is never rate-limited; everyone else waits 20 s between questions
            return None if quiet else await self.say("info", "ขอเวลาสักครู่ครับ ขอตอบคำถามก่อนหน้าก่อน", force=True, by=who)
        if self.asking >= 3:
            return await self.say("info", "ตอนนี้มีคำถามรออยู่หลายข้อ ขอตอบทีละข้อนะครับ", force=True, by=who)
        self.last_ask[who] = now
        self.asking += 1
        try:
            loop = asyncio.get_running_loop()
            hits = await loop.run_in_executor(None, lambda: kb.search_meeting(self.mid, question, 3))
            names = list(dict.fromkeys(h["name"] for h in hits))
            use_llm = False
            if self.llm_wanted():
                if self.llm.state not in ("ready", "loading") and self.llm_fits():
                    await self.say("info", "กำลังเตรียมโมเดลภาษา ขอเวลาสักครู่ครับ", force=True)
                use_llm = await self.ensure_llm()
            if use_llm:
                ctx = "\n\n".join(f"[{i + 1}] ({h['name']})\n{h['text'][:700]}" for i, h in enumerate(hits)) or "(ไม่พบเอกสารที่เกี่ยวข้อง)"
                it = self.item
                user = (f"เอกสารอ้างอิง:\n{ctx}\n\nเนื้อหาการประชุมล่าสุด:\n{self.recent_text(240, 900) or '(ยังไม่มี)'}\n\nหัวข้อปัจจุบัน: {it['title'] if it else '-'}\n\nคำถามจาก {who or 'ผู้เข้าร่วม'}: {question}\n\nตอบสั้นๆ และบอกชื่อเอกสารที่ใช้")
                out = await self.llm.chat(self.system_prompt(), user, 160, prefill="ตอบ: ")
                if out and numbers_ok(out, ctx + " " + self.recent_text(240, 900) + " " + question):
                    return await self.say("answer", out, names, llm=True, force=True, by=who)
                # no usable AI answer (empty, or it changed a figure): quote the document instead, which cannot be wrong about numbers
            if hits:
                return await self.say("answer", f"ตามเอกสาร “{hits[0]['name']}”: " + re.sub(r"\s+", " ", hits[0]["text"])[:320], names[:1], force=True, by=who)
            return await self.say("answer", "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้ในเอกสารที่มีครับ", [], force=True, by=who)
        finally:
            self.asking -= 1

    # -- end of the meeting
    async def close(self):
        if self.closed:
            return
        self.closed = True
        now = self.now()
        if self.item_log and self.item_log[-1].get("end") is None:
            self.item_log[-1]["end"] = round(now - self.started, 1)
        spent = "\n".join(f"• {l['title']}: {round(((l['end'] or 0) - l['start']) / 60)} นาที" + (f" (แผน {l['planned_min']})" if l["planned_min"] else "") for l in self.item_log)
        acts = "\n".join(f"• {a['text']} (ผู้รับผิดชอบ: {a['owner']}, กำหนด: {a['due']})" for a in self.actions[:10])
        text = "ขอบคุณทุกท่านครับ ปิดการประชุมแล้ว"
        used = False
        if self.cfg["closing"] and self.caps and self.llm_wanted() and self.llm.state == "ready":  # never start the model just to say goodbye: processing is about to need the memory
            src = self.recent_text(900, 2400)
            out = await self.llm.chat(self.system_prompt(), "เนื้อหาการประชุมช่วงท้าย:\n" + src + "\n\nสรุปมติหรือข้อตกลงสำคัญของการประชุม 2-4 ข้อ", 220, prefill="สรุป:\n")
            if out:
                text += "\nสรุปประเด็นสำคัญ:\n" + out + ("" if numbers_ok(out, src) else WARN_NUM)
                used = True
        if spent:
            text += "\nเวลาที่ใช้ต่อหัวข้อ:\n" + spent
        if acts:
            text += "\nงานที่จดไว้:\n" + acts
        await self.say("closing", text, llm=used, force=True)
        try:
            store.write_json(store.mdir(self.mid) / "bot_notes.json", {"agenda": self.item_log, "actions": self.actions, "closing": text, "goal": self.cfg["goal"]})
        except OSError:
            pass

    def status(self) -> dict:
        it, now = self.item, self.now()
        return {"enabled": True, "paused": self.paused, "closed": self.closed, "index": self.idx, "item": it, "elapsed_s": round(now - self.item_start) if it else 0,
                "budget_s": (it["minutes"] * 60) if it else 0, "agenda": self.cfg["agenda"], "actions": self.actions, "talk": {k: round(v) for k, v in self.talk.items()},
                "llm": {"state": self.llm.state if self.llm else "off", "reason": self.llm.reason if self.llm else "off", "mode": self.cfg["llm"]},
                "messages": len(self.msgs), "asking": self.asking, "name": self.cfg["name"], "log": self.item_log}
