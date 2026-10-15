"""Thai meeting summary with Typhoon2.5-Qwen3-4B (llama.cpp, CPU, low priority). The server is started only while a
summary is made and stopped afterwards, so its ~3 GB are free again for recording / the other apps on the box.
Long transcripts are summarized map-reduce; the partial results are checkpointed, so a crash resumes mid-way."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

import config
import store

# Measured on the NPU (Typhoon 4B, 24 of 36 layers, nothing else running): the transcript is READ at ~78 tok/s but every answer is WRITTEN at only
# 9 tok/s, and the old map step wrote 500 tokens for each 2600-character part (54 s of the 75 s per call). Bigger parts (fewer calls) and a
# 300-token cap on the notes cut the total time of a long meeting by roughly 40 %.
CHUNK_CHARS = 4200
MAP_TOKENS = 300
MAP_VERSION = "c4200-t300-v2"  # stored with the saved notes: notes made with other settings are not mixed in

SYS = ("คุณคือเลขานุการมืออาชีพที่สรุปการประชุมภาษาไทย ตอบเป็นภาษาไทยเท่านั้น (ชื่อเฉพาะและคำศัพท์อังกฤษคงไว้ตามเดิม) "
       "ใช้เฉพาะข้อมูลที่อยู่ในบันทึก ห้ามแต่งเติมหรือเดา ถ้าไม่มีข้อมูลให้เขียนว่า \"ไม่มี\" "
       "บันทึกเสียงอาจมีคำผิดจากการถอดเสียง ให้ตีความจากบริบทอย่างระมัดระวัง "
       "ถ้าบันทึกระบุชื่อผู้พูดไว้หน้าข้อความ ให้ใช้ชื่อนั้นระบุผู้รับผิดชอบ "
       "เขียนตัวเลขเป็นเลขอารบิก (เช่น 15, 500,000) ห้ามสะกดเป็นคำอ่านภาษาไทย และเขียนคำภาษาอังกฤษด้วยตัวอักษรอังกฤษ "
       "คัดลอกตัวเลข จำนวนเงิน วันที่ ตามบันทึกให้ตรงตัว ห้ามแปลงหน่วยหรือคำนวณใหม่ (เช่น \"2 แสน\" คือ 200,000 ไม่ใช่ 20,000)")

MAP_TASK = ("นี่คือบันทึกการประชุมส่วนหนึ่ง สรุปเป็นข้อๆ ไม่เกิน 7 ข้อ: ประเด็นที่คุยกัน, สิ่งที่ตกลงหรือตัดสินใจ, "
            "งานที่ต้องทำ (ใคร ทำอะไร เมื่อไร), และตัวเลข/วันที่/ชื่อสำคัญ")

PERSON_TASK = ("ต่อไปนี้คือสิ่งที่ {name} พูดในการประชุม สรุปเป็นข้อๆ สั้นๆ ไม่เกิน 5 ข้อ: ประเด็น/ความเห็นหลักที่เสนอ, "
               "สิ่งที่รับปากหรือได้รับมอบหมาย (ถ้ามี), คำถามสำคัญที่ถาม ใช้เฉพาะข้อมูลในบันทึกของ {name} เท่านั้น ห้ามเดา ห้ามเขียนข้อซ้ำกัน "
               "ถ้าไม่มีข้อมูลในหัวข้อใดให้ข้ามหัวข้อนั้น ไม่ต้องเขียนว่าไม่มี ห้ามนับจำนวนคนหรือพูดถึงสิ่งที่ไม่ได้อยู่ในบันทึกของ {name}")

FINAL_TASK = ("จากบันทึกการประชุมด้านล่าง เขียนสรุปการประชุมเป็น Markdown ตามหัวข้อนี้เท่านั้น (ห้ามเพิ่มหัวข้ออื่น):\n"
              "## ภาพรวม\n(4-6 ประโยค: การประชุมเรื่องอะไร ใครเข้าร่วม ประเด็นหลักและข้อสรุปโดยรวม)\n"
              "## ประเด็นสำคัญ\n(ข้อๆ ให้ละเอียด ครอบคลุมทุกเรื่องที่คุย พร้อมตัวเลข ชื่อ และเหตุผล)\n"
              "## ประเด็นค้าง / ต้องติดตาม\n(ข้อๆ เรื่องที่ยังไม่ได้ข้อสรุปหรือต้องคุยต่อ)")

DECISION_TASK = ("ต่อไปนี้คือประโยคจากการประชุมที่เกี่ยวกับมติหรือการตัดสินใจ เขียนรายการ \"มติ / ข้อตัดสินใจ\" เป็นข้อๆ ในรูปแบบ: "
                 "เรื่อง – ผลที่ตกลง (เช่น อนุมัติ / เห็นชอบ / รับทราบ / ไม่อนุมัติ) – รายละเอียด ตัวเลข หรือคะแนนเสียง (ถ้ามี) "
                 "ใช้เฉพาะข้อมูลในประโยคเหล่านี้ ห้ามเดาหรือแต่งเพิ่ม รวมข้อที่ซ้ำกันเป็นข้อเดียว เรียงตามเวลา")
ACTION_TASK = ("ต่อไปนี้คือประโยคจากการประชุมที่เกี่ยวกับงานที่ต้องทำหรือการมอบหมายงาน เขียนรายการ \"งานที่ต้องทำ\" เป็นข้อๆ ในรูปแบบ: "
               "ผู้รับผิดชอบ – งาน – กำหนดเวลา ถ้าประโยคไม่ได้ระบุผู้รับผิดชอบหรือกำหนดเวลา ให้เขียนว่า \"ไม่ระบุ\" "
               "ใช้เฉพาะข้อมูลในประโยคเหล่านี้ ห้ามเดาหรือแต่งเพิ่ม รวมข้อที่ซ้ำกันเป็นข้อเดียว")



def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{config.LLM_PORT}/health", timeout=2) as r:
            return b"ok" in r.read()
    except (OSError, urllib.error.URLError):
        return False


def kill_orphan():
    """The worker starts: a summary model left by a previous worker (frozen during a recording, it ignores the parent-death SIGTERM) is killed."""
    f = config.DATA / ".llm.pid"
    try:
        pid = int(f.read_text())
        with open(f"/proc/{pid}/cmdline", "rb") as c:
            cmd = c.read().split(b"\0")
        if any(x.endswith(b"llama-server") for x in cmd) and str(config.LLM_PORT).encode() in cmd:
            os.kill(pid, signal.SIGKILL)
    except (OSError, ValueError):
        pass
    f.unlink(missing_ok=True)


def config_settings() -> dict:
    import store

    return store.settings()


class LlamaServer:
    """llama-server for the summary. First on the Hexagon NPU (llama.cpp 'snapdragon' build, Q4_0 weights: measured in projectAI
    prompt 318 tok/s vs 43 on the CPU, generation about the same - and the summary is mostly prompt), then on the CPU if the NPU
    cannot map memory (another process holds it) or the setting summary_on_npu is off. `backend` says which one is running."""

    backend = "CPU"

    def __init__(self, on_pause=None):
        self.metrics = {"calls": 0, "prompt_n": 0, "prompt_ms": 0.0, "gen_n": 0, "gen_ms": 0.0}  # read by the statistics (stats.py)
        self.on_pause = on_pause  # callback(bool): the worker shows on the meeting that the summary waits for a recording
        self.paused_n = 0         # times the model was frozen: chat() asks again when a freeze made a call time out
        self._guard_stop = threading.Event()
        self._guard_t = None

    def _guard(self):
        """Recording comes first: while a meeting is being recorded the model server is frozen (SIGSTOP, mid-answer too) and woken after
        (SIGCONT), so the recorder and live captions get every core. Its weights are a file mapping: the kernel may reclaim them meanwhile."""
        proc, stopped = self.proc, False
        while not self._guard_stop.is_set() and proc.poll() is None:
            rec = store.recording_active()
            if rec != stopped:
                try:
                    os.kill(proc.pid, signal.SIGSTOP if rec else signal.SIGCONT)
                except OSError:
                    break
                stopped = rec
                self.paused_n += rec
                self._notify(rec)
            self._guard_stop.wait(1.0)
        if stopped:
            self._wake()
            self._notify(False)

    def _notify(self, on: bool):
        try:
            if self.on_pause:
                self.on_pause(on)
        except Exception:  # noqa: BLE001 - only the display of the pause
            pass

    def _wake(self):
        """A frozen process acts on no signal but SIGKILL: wake it before stopping it."""
        try:
            if self.proc.poll() is None:
                os.kill(self.proc.pid, signal.SIGCONT)
        except (OSError, AttributeError):
            pass

    def _stop_guard(self):
        self._guard_stop.set()
        if self._guard_t is not None and self._guard_t is not threading.current_thread():
            self._guard_t.join(5)
        self._wake()

    def summary_metrics(self) -> dict:
        m = self.metrics
        return {"calls": m["calls"], "prompt_tokens": m["prompt_n"], "gen_tokens": m["gen_n"],
                "prompt_tps": round(1000 * m["prompt_n"] / m["prompt_ms"], 1) if m["prompt_ms"] else None,
                "gen_tps": round(1000 * m["gen_n"] / m["gen_ms"], 1) if m["gen_ms"] else None}

    def _start(self, npu: bool, ngl: int = 24, dev: str = "HTP0"):
        """dev: 'HTP0' (first NPU core), 'HTP1' (second core) or 'HTP0,HTP1' (both: the model is split over them)."""
        binary = config.LLAMA_BUILD / "bin/llama-server"
        logf = open(config.DATA / "llm.log", "ab")
        logf.write(f"\n=== {time.strftime('%F %T')} starting on {f'NPU ({dev}, {ngl} layers)' if npu else 'CPU'}\n".encode())
        logf.flush()
        env = {**os.environ, "LD_LIBRARY_PATH": str(config.LLAMA_BUILD / "lib")}
        if npu:  # the FastRPC driver refuses ~1 GB mappings: weights go in 256 MB buffers
            env.update({"ADSP_LIBRARY_PATH": f"{config.LLAMA_BUILD / 'lib'};/usr/lib/dsp/cdsp", "GGML_HEXAGON_MBUF": "256", "GGML_HEXAGON_DEVICES": "2"})  # 2: this board has two NPU cores (cdsp, cdsp1)
            dev = ["-dev", dev, "-ngl", str(ngl), "-t", "4"]  # layers beyond ngl run on the CPU threads
        else:  # jobs only start when nothing is being recorded, and the worker is nice 10: use every core
            dev = ["-dev", "none", "-t", str(config.CORES)]
        cmd = ["nice", "-n", "15", str(binary), "-m", str(self.gguf), "--host", "127.0.0.1", "--port", str(config.LLM_PORT),
               "--no-webui", *dev, "-c", "8192", "-np", "1"]
        self.proc = config.spawn(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        (config.DATA / ".llm.pid").write_text(str(self.proc.pid))  # kill_orphan(): a frozen copy outlives a worker that died during a recording
        logf.close()
        t = time.time()
        while time.time() - t < (120 if npu else 900 if "30b" in self.gguf.name else 240):  # the 18 GB model is read from disk on the first request
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}), see data/llm.log")
            if _healthy():
                return
            time.sleep(1)
        self.proc.terminate()
        raise RuntimeError("llama-server did not become ready")

    def __enter__(self):
        binary = config.LLAMA_BUILD / "bin/llama-server"
        self.gguf = config.LLM_GGUF
        choice = config_settings().get("summary_model")
        other = config.SUMMARY_GGUF.get(choice) if choice != "typhoon-4b" else None
        slow = config.DATA / ".llm_big_slow"
        if choice == "typhoon-30b" and slow.exists() and time.time() - slow.stat().st_mtime < 6 * 3600:
            import logging

            logging.getLogger("summarize").warning("the 30B model stalled recently: using the Typhoon 4B model for now")
            other = None
        if other is not None:
            if other.exists():
                self.gguf = other
            else:
                import logging

                logging.getLogger("summarize").warning("summary_model %s is not downloaded: using the Typhoon 4B model", choice)
        if not binary.exists() or not self.gguf.exists():
            raise RuntimeError("summary model not found (llama-server / Typhoon GGUF)")
        self.family = "gemma" if "gemma" in self.gguf.name.lower() else "qwen"  # Typhoon = Qwen3 (ChatML, empty think block); Gemma = its own turns
        import atexit

        atexit.register(self._kill)  # the worker runs this in a thread: a SIGTERM must not leave the model server behind
        import socket

        waited = False
        for _ in range(40):  # a previous summary model may still be shutting down: it holds the port AND the NPU memory (that made the next start fall back to fewer NPU layers)
            try:
                socket.create_connection(("127.0.0.1", config.LLM_PORT), 0.5).close()
            except OSError:
                break
            waited = True
            time.sleep(1)
        if waited:
            time.sleep(3)
        bad = config.DATA / ".npu_llm_bad"  # the NPU model died while in use: stay on the CPU for 30 minutes
        recently_bad = bad.exists() and time.time() - bad.stat().st_mtime < 1800
        # Gemma 4 E4B stays on the CPU (measured 2026-10-15): on the NPU the same prompt either hung after reading 1280 tokens or came back in a broken
        # format (no headings), even with 256-token batches and the KV checkpoints off ('VTCM-TOO-SMALL'); on the CPU it reads 50 tok/s and writes
        # 11 tok/s, the same writing speed as on the NPU, with a correct answer. Only the Qwen-based Typhoon 4B Q4_0 goes on the NPU.
        want_npu = bool(config_settings().get("summary_on_npu", True)) and "q4_0" in self.gguf.name.lower() and self.family == "qwen" and not recently_bad
        if want_npu:
            # Measured on this board (Typhoon 4B Q4_0, 36 layers, 2.3 GB): one NPU core maps at most ~1.5 GB (24 layers load, 27+ fail with 'fastrpc_mmap
            # failed'), but the board has TWO cores: the whole model split over both reads 400 tok/s (75 on one core, 30-43 on the CPU). When something else
            # already holds core 0 (the speech model, another service) the second core alone, then fewer layers, then the CPU.
            attempts = [("HTP0,HTP1", 99), ("HTP0,HTP1", 99), ("HTP1", 24), ("HTP0", 24), ("HTP0", 20), ("HTP0", 12)]  # (the first start can fail only because the NPU memory of a previous model is still being released)
            for dev_, ngl in attempts:
                try:
                    self._start(True, ngl, dev_)
                    self.backend = "NPU (both cores, all layers)" if dev_ == "HTP0,HTP1" else f"NPU ({dev_}, {ngl} layers)"
                    break
                except RuntimeError as e:
                    self._kill()
                    import logging

                    logging.getLogger("summarize").warning("summary model could not start on the NPU (%s, %d layers): %s", dev_, ngl, e)
                    time.sleep(4)
        if self.backend == "CPU":
            self._start(False)
        self.ban = self._ban_tokens() if self.family == "qwen" else []  # the <tool_call> habit is Qwen's; Gemma has no such tokens
        self._guard_t = threading.Thread(target=self._guard, name="llm-guard", daemon=True)
        self._guard_t.start()
        return self

    ban: list = []

    def _ban_tokens(self) -> list:
        ids = []
        try:
            for tok in ("<tool_call>", "</tool_call>", "<tool_response>"):
                req = urllib.request.Request(f"http://127.0.0.1:{config.LLM_PORT}/tokenize", json.dumps({"content": tok, "special": True}).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    ids += json.loads(r.read())["tokens"]
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            pass
        return [[i, False] for i in ids]

    def _kill(self):  # (failed starts call it too: the guard is not running yet and must stay startable)
        self._wake()
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(10)
        except Exception:  # noqa: BLE001
            pass

    def __exit__(self, *a):
        self._stop_guard()
        if self.backend != "CPU" and self.proc.poll() is not None and a and a[0] is not None:
            (config.DATA / ".npu_llm_bad").touch()  # it crashed under us (not a normal stop)
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    family = "qwen"

    def _prompt(self, system: str, user: str, prefill: str) -> tuple[str, list[str]]:
        """The raw prompt for /completion and the stop strings, per model family."""
        if self.family == "gemma":
            # Gemma 4's own turn format (the GGUF template via /apply-template, thinking off); a hand-written fallback when the server lacks it.
            # /completion adds the BOS token itself: a <bos> written in the text would double it (Gemma is sensitive to that).
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{config.LLM_PORT}/apply-template",
                                             json.dumps({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                         "chat_template_kwargs": {"enable_thinking": False}}).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    text = json.loads(r.read())["prompt"]
            except (OSError, ValueError, KeyError, urllib.error.URLError):  # Gemma 4's turn tokens (not Gemma 3's <start_of_turn>)
                text = f"<|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n"
            if text.startswith("<bos>"):
                text = text[5:]
            return text + prefill, ["<turn|>", "<|turn>", "<end_of_turn>"]
        # ChatML with an empty <think> block: Qwen3-based Typhoon would otherwise spend the budget on reasoning
        return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{prefill}",
                ["<|im_end|>", "<|im_start|>"])

    def chat(self, system: str, user: str, max_tokens: int = 700, prefill: str = "") -> str:
        prompt, stop = self._prompt(system, user, prefill)
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1,
                "stop": stop, "stream": False, "cache_prompt": True}
        if self.ban:
            body["logit_bias"] = self.ban
        req = urllib.request.Request(f"http://127.0.0.1:{config.LLM_PORT}/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        for _ in range(6):
            while store.recording_active():  # the guard keeps the model frozen meanwhile: ask once the recording is over
                time.sleep(2)
            n0 = self.paused_n
            try:
                with urllib.request.urlopen(req, timeout=600) as r:  # normal calls take 40-130 s; one that needs 10 minutes is stuck (a 30B run once crawled at 0.13 tok/s for 30 min)
                    res = json.loads(r.read())
                break
            except OSError as e:  # socket timeout / reset
                if self.paused_n != n0:  # a recording froze the model during this call (a long meeting outlasts the timeout): ask again
                    continue
                if "30b" in self.gguf.name:
                    (config.DATA / ".llm_big_slow").touch()  # the next summary uses the 4B model for a while
                elif self.backend != "CPU":  # the NPU stalled mid-prompt (seen 2026-10-15: one call answered, the next never did): the retry runs on the CPU for 30 min
                    (config.DATA / ".npu_llm_bad").touch()
                raise RuntimeError(f"the summary model did not answer in time ({type(e).__name__}): the job is retried") from e
        else:
            raise RuntimeError("recordings kept interrupting this summary step: the job is retried")
        tm = res.get("timings") or {}
        m = self.metrics
        m["calls"] += 1
        m["prompt_n"] += tm.get("prompt_n", 0)
        m["prompt_ms"] += tm.get("prompt_ms", 0.0)
        m["gen_n"] += tm.get("predicted_n", 0)
        m["gen_ms"] += tm.get("predicted_ms", 0.0)
        text = clean(res["content"])
        if res.get("stop_type") == "limit" and "\n" in text:  # cut off by the token cap in the middle of a line: drop that unfinished last line
            text = text.rsplit("\n", 1)[0].rstrip()
        return text


def clean(text: str) -> str:
    """Strip stray special tokens the 4B model sometimes emits and drop repeated bullet lines."""
    import re

    text = re.sub(r"</?(?:tool_call|tool_response|think|\|im_start\||\|im_end\|)>|<(?:end_of_turn|start_of_turn|bos|eos)>|<\|[a-z_]+\|>|<\|turn>|<turn\|>", "", text).strip()  # Qwen / Gemma control tokens
    seen, out = set(), []
    for ln in text.splitlines():
        key = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(ln.rstrip())
    return "\n".join(out).strip()


def split_lines(lines: list[str], limit: int = CHUNK_CHARS) -> list[str]:
    out, cur, n = [], [], 0
    for ln in lines:
        if n + len(ln) > limit and cur:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def line_of(s: dict, names: dict[str, str]) -> str:
    who = names.get(s.get("spk")) if len(names) > 1 else None
    return f"[{store.fmt_ts(s['t0'])}] " + (f"{who}: " if who else "") + s["text"]


def _checkpoint(mid: str):
    f = store.mdir(mid) / "summary.partial.json"
    part = store.read_json(f, {}) or {}
    if part.get("map_version") != MAP_VERSION:
        part["map"] = {}
        part["map_version"] = MAP_VERSION
    part.setdefault("map", {})
    part.setdefault("notes", {})
    return f, part


def background(glossary: str, context: str, terms: list[str] | None = None, coverage: list[dict] | None = None, P: dict | None = None) -> str:
    out = ""
    words = [w for w in ([glossary.strip()] if glossary.strip() else []) + (terms or [])[:30]]
    if P is not None and P is PROMPTS["en"]:
        if words:
            out += "\nCorrect spellings of names / terms (for spelling only; do not mention them unless the notes do): " + ", ".join(words)
        if coverage:
            no = [c["title"] for c in coverage if not c["covered"]]
            if no:
                out += "\nAgenda items not discussed in the notes (never invent a decision or content for them): " + " | ".join(no)
        if context.strip():
            out += f"\nBackground information (NOT something said in the meeting; never summarize it as meeting content):\n{context.strip()[:1500]}"
        return out
    if words:
        out += f"\nคำศัพท์/ชื่อที่สะกดถูกต้อง (ใช้เพื่อสะกดคำเท่านั้น ห้ามกล่าวถึงคำเหล่านี้ถ้าไม่มีอยู่ในบันทึก): {', '.join(words)}"
    if coverage:
        no = [c["title"] for c in coverage if not c["covered"]]
        if no:
            out += "\nวาระที่ไม่พบการพูดถึงในบันทึก (ห้ามแต่งมติหรือเนื้อหาของวาระเหล่านี้): " + "; ".join(no)
    if context.strip():
        out += f"\nข้อมูลประกอบ (ไม่ใช่สิ่งที่พูดในการประชุม ห้ามนำมาสรุปเป็นเนื้อหาที่ไม่ได้พูด):\n{context.strip()}"
    return out


import re  # noqa: E402


def references(mid: str, text: str, k: int = 2, limit: int = 600) -> str:
    """Relevant passages of the meeting's documents (+ the knowledge base) for this stretch of the meeting."""
    import kb

    try:
        hits = kb.search_meeting(mid, text[:500], k, include_meetings=False)
    except Exception:  # noqa: BLE001 - references are a bonus: the summary works without them
        return ""
    if not hits:
        return ""
    return "\nเอกสารประกอบ (ใช้ตรวจชื่อ คำศัพท์ และบริบทเท่านั้น ไม่ใช่สิ่งที่พูดในที่ประชุม):\n" + "\n".join(f"[{h['name']}] " + " ".join(h["text"].split())[:limit // k] for h in hits)


SYS_EN = ("You are a professional minute-taker summarizing a meeting. Answer in English only. Use only what is in the notes: never invent or guess; "
          "write \"none\" when there is nothing. The notes come from speech recognition and may contain misheard words: read them carefully in context. "
          "When a speaker's name precedes a line, use it to say who is responsible. Write numbers as digits (15, 500,000) and copy figures, amounts and "
          "dates exactly as written: never convert units or recompute.")
MAP_TASK_EN = ("This is one part of a meeting's notes. Summarize it in at most 7 bullet points: topics discussed, what was agreed or decided, "
               "action items (who does what, by when), and the important numbers, dates and names.")
PERSON_TASK_EN = ("Below is what {name} said in the meeting. Summarize it in at most 5 short bullet points: the main points / opinions put forward, "
                  "commitments made or tasks received (if any), important questions asked. Use only {name}'s own lines, never guess, no duplicate points; "
                  "skip a heading that has nothing, do not write \"none\". Do not count people or mention anything outside {name}'s lines.")
FINAL_TASK_EN = ("From the meeting notes below, write the meeting summary in Markdown using exactly these headings (add no other):\n"
                 "## Overview\n(4-6 sentences: what the meeting was about, who took part, the main points and the overall conclusion)\n"
                 "## Key points\n(detailed bullet points covering everything discussed, with numbers, names and reasons)\n"
                 "## Open items / follow-up\n(bullet points: matters left undecided or to be discussed again)")
DECISION_TASK_EN = ("Below are sentences from the meeting that concern decisions. Write the \"Decisions\" list as bullet points in the form: "
                    "subject – outcome (e.g. approved / agreed / noted / rejected) – details, figures or votes (if any). "
                    "Use only these sentences, never guess or add; merge duplicates; keep time order.")
ACTION_TASK_EN = ("Below are sentences from the meeting that concern tasks and assignments. Write the \"Action items\" list as bullet points in the form: "
                  "owner – task – deadline; write \"not stated\" when a sentence gives no owner or deadline. "
                  "Use only these sentences, never guess or add; merge duplicates.")
DECISION_RE_EN = re.compile(r"\b(decid\w*|agree\w*|approv\w*|resol\w*|vot\w*|unanimous\w*|motion|conclu\w*|final(?:ly|ize|ise)\w*|sign(?:ed)? off|go ahead|green.?light)\b", re.I)
ACTION_RE_EN = re.compile(r"\b(action item|to.?do|assign\w*|responsib\w*|owner|deadline|due|by (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next|end of|eod|eow|the \d|\d)|will (?:send|prepare|follow|check|share|draft|update|set up|schedule|report|contact|review|finish|deliver|write)|"
                           r"follow.?up|take care of|take the lead|circulate|get back to|needs? to|has? to|must|should|please (?:send|prepare|check|share|draft|update|review))\b", re.I)
PROMPTS = {
    "th": {"SYS": None, "MAP": None, "PERSON": None, "FINAL": None, "DECISION": None, "ACTION": None, "DEC_RE": None, "ACT_RE": None,  # filled below (the Thai texts above)
           "H": ("ภาพรวม", "ประเด็นสำคัญ", "มติ / ข้อตัดสินใจ", "งานที่ต้องทำ", "ประเด็นค้าง / ต้องติดตาม", "รายละเอียดตามช่วงเวลา"),
           "NONE": ("ไม่พบการระบุมติหรือการตัดสินใจในบันทึก", "ไม่พบการมอบหมายงานในบันทึก", "ไม่พบประเด็นค้างที่ระบุไว้ในบันทึก"),
           "NOTES": "บันทึก", "SHORT": "บันทึกย่อ", "SENT": "ประโยค", "PART": "ส่วนที่", "EMPTY": "## ภาพรวม\nไม่พบเนื้อหาการสนทนาในการบันทึกนี้\n"},
    "en": {"SYS": SYS_EN, "MAP": MAP_TASK_EN, "PERSON": PERSON_TASK_EN, "FINAL": FINAL_TASK_EN, "DECISION": DECISION_TASK_EN, "ACTION": ACTION_TASK_EN,
           "DEC_RE": DECISION_RE_EN, "ACT_RE": ACTION_RE_EN,
           "H": ("Overview", "Key points", "Decisions", "Action items", "Open items / follow-up", "Details by time"),
           "NONE": ("No decision was stated in the notes", "No task or assignment was stated in the notes", "No open item was stated in the notes"),
           "NOTES": "Notes", "SHORT": "Short notes", "SENT": "Sentences", "PART": "part", "EMPTY": "## Overview\nNo conversation was found in this recording.\n"},
}


def prompts(mid: str) -> dict:
    """The prompt set of this meeting's language (state "language": th | th-en | en). Thai + English meetings use the Thai prompts, which already
    keep English names and terms as written, plus a line saying both languages occur."""
    lang = (store.load_state(mid) or {}).get("language") or config_settings().get("language") or "th"
    if lang == "en":
        return PROMPTS["en"]
    th = PROMPTS["th"]
    if th["SYS"] is None:
        th.update(SYS=SYS, MAP=MAP_TASK, PERSON=PERSON_TASK, FINAL=FINAL_TASK, DECISION=DECISION_TASK, ACTION=ACTION_TASK, DEC_RE=DECISION_RE, ACT_RE=ACTION_RE)
    if lang == "th-en":
        return {**th, "SYS": th["SYS"] + " การประชุมนี้พูดทั้งภาษาไทยและอังกฤษ: คงชื่อ คำศัพท์ และประโยคภาษาอังกฤษไว้ตามที่พูด สรุปเป็นภาษาไทย"}
    return th


DECISION_RE = re.compile(r"(มีมติ|ลงมติ|ที่ประชุม\S{0,6}(?:อนุมัติ|เห็นชอบ|รับทราบ|ตกลง|มีมติ)|อนุมัติ|เห็นชอบ|ไม่เห็นด้วย|งดออกเสียง|คะแนนเสียง|เสียงข้างมาก|ผ่านการอนุมัติ|เป็นอันว่า|ตกลงตามนี้|ตกลงกัน|ตัดสินใจ|สรุปว่า)")
ACTION_RE = re.compile(r"(มอบหมาย|ให้(?:ทีม|ฝ่าย|คุณ|ทุกคน)?\S{0,10}(?:ดำเนินการ|จัดทำ|ติดตาม|ส่ง|เตรียม|ประสาน|ตรวจสอบ|รายงาน)|ต้อง(?:ทำ|ส่ง|ดำเนินการ|เตรียม|ติดตาม)|จะ(?:ดำเนินการ|ส่ง|ทำ|ติดตาม|เตรียม|ประสาน|รายงาน)|ภายใน\s*(?:วัน|สัปดาห์|เดือน|ไตรมาส|ปี|\d)|กำหนดส่ง|ผู้รับผิดชอบ|ฝากให้|รบกวน|เดดไลน์|deadline|follow.?up|action item|รับไป|เดี๋ยว(?:ผม|ฉัน|ดิฉัน))", re.I)


def evidence(segs: list[dict], names: dict[str, str], rx, limit: int = 36, max_chars: int = 3600) -> str:
    """The 4B model overlooks decisions and tasks when it only sees a general prompt, so the sentences that carry them are found by rule
    (words like มีมติ / อนุมัติ / มอบหมาย / ภายใน ...) and the model only has to write them up. Each hit comes with the line after it
    (votes and deadlines often follow). Best-scoring hits first, then put back in time order."""
    rows = [(i, s) for i, s in enumerate(segs) if s["text"].strip()]
    hits = []
    for k, (i, s) in enumerate(rows):
        m = rx.findall(s["text"])
        if m:
            hits.append((len(set(m)) + (1 if re.search(r"\d", s["text"]) else 0), k))
    best = sorted(hits, key=lambda h: (-h[0], h[1]))[:limit]
    out, size = [], 0
    for _, k in sorted(best, key=lambda h: h[1]):
        i, s = rows[k]
        txt = line_of(s, names)
        if k + 1 < len(rows) and len(s["text"]) < 90:
            txt += "\n" + line_of(rows[k + 1][1], names)
        if size + len(txt) > max_chars:
            break
        out.append(txt)
        size += len(txt) + 1
    return "\n".join(out)


def _range_of(chunk: str) -> str:
    ts = re.findall(r"^\[(\d\d:\d\d:\d\d)\]", chunk, re.M)
    return f"{ts[0]}–{ts[-1]}" if ts else ""


def summarize(llm: LlamaServer, mid: str, segs: list[dict], names: dict[str, str], glossary: str, progress=lambda f: None, context: str = "",
              terms: list[str] | None = None, coverage: list[dict] | None = None, agenda: list[dict] | None = None, use_refs: bool = True) -> str:
    part_file, part = _checkpoint(mid)
    P = prompts(mid)
    lines = [line_of(s, names) for s in segs if s["text"].strip()]
    if not lines:
        return P["EMPTY"]
    chunks = split_lines(lines)
    gl = background(glossary, context, terms, coverage, P)
    detail: list[tuple[str, str]] = []  # (time range, notes of that part): printed as "details by time" for long meetings
    if len(chunks) == 1:
        material = chunks[0] + (references(mid, chunks[0]) if use_refs else "")
    else:
        notes = []
        for i, c in enumerate(chunks):
            key = str(i)
            if key not in part["map"]:
                part["map"][key] = strip_echo(llm.chat(P["SYS"], f"{P['MAP']}{gl}\n\n{P['NOTES']}:\n{c}" + (references(mid, c) if use_refs else ""), MAP_TOKENS))
                store.write_json(part_file, part)
            notes.append(f"({P['PART']} {i + 1}/{len(chunks)})\n{part['map'][key]}")
            progress((i + 1) / (len(chunks) + 1) * 0.6)
        detail = [(_range_of(c), part["map"][str(i)]) for i, c in enumerate(chunks)]
        # Notes of a long meeting do not fit one final prompt: condense them in groups, level by level, until they do. (The old code condensed
        # only the first 9000 characters, so the end of a long meeting was missing from the summary.)
        level = 0
        while sum(len(n) for n in notes) > 6500 and level < 3:
            groups, cur = [], []
            for n in notes:
                if cur and sum(len(x) for x in cur) + len(n) > 6000:
                    groups.append(cur)
                    cur = []
                cur.append(n)
            if cur:
                groups.append(cur)
            if len(groups) == len(notes):  # nothing to merge
                break
            new = []
            for k, g in enumerate(groups):
                key = f"r{level}-{k}"
                if key not in part["map"]:
                    part["map"][key] = llm.chat(P["SYS"], f"{P['MAP']}{gl}\n\n{P['SHORT']}:\n" + "\n\n".join(g), MAP_TOKENS)
                    store.write_json(part_file, part)
                new.append(part["map"][key])
            notes, level = new, level + 1
        material = "\n\n".join(notes)
        if len(material) > 9000:
            material = material[:9000]
    # decisions and tasks: sentences found by rule, written up by the model (checkpointed); nothing found -> said so, no invention
    found = {}
    for key, rx, task, none in (("decisions", P["DEC_RE"], P["DECISION"], P["NONE"][0]), ("actions", P["ACT_RE"], P["ACTION"], P["NONE"][1])):
        ev = evidence(segs, names, rx)
        if not ev:
            found[key] = none
        else:
            if key not in part["notes"]:
                part["notes"][key] = llm.chat(P["SYS"], f"{task}{gl}\n\n{P['SENT']}:\n{ev}", 450)
                store.write_json(part_file, part)
            found[key] = part["notes"][key]
        progress(0.7 if key == "decisions" else 0.8)
    final = strip_echo(llm.chat(P["SYS"], f"{P['FINAL']}{gl}\n\n{P['NOTES']}:\n{material}", 1000), terms)
    sec = _sections(final)
    H = P["H"]
    out = [f"## {H[0]}\n" + sec.get(H[0], "").strip(), f"## {H[1]}\n" + sec.get(H[1], "").strip(),
           f"## {H[2]}\n" + _drop_template(found["decisions"]), f"## {H[3]}\n" + _drop_template(found["actions"]),
           f"## {H[4]}\n" + (sec.get(H[4], "").strip() or P["NONE"][2])]
    if detail:
        out.append(f"## {H[5]}\n" + "\n\n".join(f"### [{r}]\n{n.strip()}" for r, n in detail))
    return "\n\n".join(out)


def _drop_template(text: str) -> str:
    """The bigger model sometimes repeats the format line it was given ('เรื่อง – ผลที่ตกลง ...'): drop it. Gemma writes the list's title in front of
    the first item ('มติ / ข้อตัดสินใจ: เรื่อง – อนุมัติ …'): the title goes, and an item without a bullet gets one (the UI counts bullets)."""
    out = []
    for ln in text.splitlines():
        if re.match(r"^\s*(เรื่อง\s*[–-]\s*ผลที่ตกลง|ผู้รับผิดชอบ\s*[–-]\s*งาน\s*[–-]\s*กำหนดเวลา)", ln):
            continue
        ln = re.sub(r"^\s*\**(มติ\s*/\s*ข้อตัดสินใจ|งานที่ต้องทำ|รายการ[^:：*]{0,20}|Decisions|Action items)\**\s*[:：][\s*:：]*", "", ln)
        if ln.strip() and not re.match(r"^\s*(?:[-*•]|\d+[.)])\s", ln) and not ln.startswith(" ") and not re.match(r"^\s*(ไม่พบ|ไม่มี|No |None)", ln):  # "nothing found" is a sentence, not an item
            ln = "- " + ln.strip()
        out.append(ln)
    return "\n".join(out).strip()


def _sections(md: str) -> dict[str, str]:
    """Markdown '## heading' sections of the model's answer -> {heading: body}."""
    out, cur = {}, None
    for ln in md.splitlines():
        m = re.match(r"^#{1,3}\s*(.+?)\s*$", ln)
        if m:
            cur = m.group(1).strip("* ")
            out[cur] = ""
        elif cur is not None:
            out[cur] += ln + "\n"
    return out


def strip_echo(md: str, terms: list[str] | None = None) -> str:
    """The 4B model sometimes ends with an empty “หมายเหตุ” or repeats the glossary it was given: drop those lines."""
    out = []
    for ln in md.splitlines():
        if re.match(r"^\s*\*{0,2}(หมายเหตุ|คำศัพท์/ชื่อที่เกี่ยวข้อง|Notes?|Glossary|Terms)[^\n]*?\*{0,2}\s*:?\s*$", ln) and len(ln.strip(" *:")) < 12:
            continue
        if not out and re.match(r"^\s*\*{0,2}#{0,3}\s*(สรุป(การ)?ประชุม|Meeting \d{4}-\d\d-\d\d|บันทึกการประชุม|Meeting (summary|notes)|Summary of)[^\n]{0,60}\*{0,2}\s*$", ln, re.I):
            continue  # Gemma opens its notes with a title line ("**สรุปการประชุม Meeting 2026-10-10 09:13**"): not a note
        if terms and sum(1 for t in terms[:12] if t in ln) >= 5:
            continue
        out.append(ln)
    return "\n".join(out).rstrip()


def by_agenda(llm: LlamaServer, mid: str, segs: list[dict], names: dict[str, str], coverage: list[dict], terms: list[str] | None = None, progress=lambda f: None) -> str:
    """"## สรุปตามวาระ": EVERY agenda item, in order. An item nobody talked about is marked by the program (the model cannot invent a decision for it);
    for the others the model only sees the transcript lines that belong to that item (the matching sentences and their neighbours)."""
    part_file, part = _checkpoint(mid)
    part.setdefault("agenda", {})
    P = prompts(mid)
    en = P is PROMPTS["en"]
    gl = background("", "", terms, None, P)
    out = ["## Summary by agenda item" if en else "## สรุปตามวาระ"]
    for n, c in enumerate(coverage):
        key = str(c["index"])
        if not c["covered"]:
            out.append(f"{c['index'] + 1}) **{c['title']}**: " + ("not discussed in the notes" if en else "ไม่พบการพูดถึงในบันทึก"))
            continue
        if key not in part["agenda"]:
            idx = sorted({j + d for j in c["seg_idx"] for d in (-1, 0, 1) if 0 <= j + d < len(segs)})
            text = "\n".join((f"{names.get(segs[j].get('spk'), '')}: " if names and segs[j].get("spk") else "") + segs[j]["text"] for j in idx)[:2400]
            if en:
                ans = llm.chat(P["SYS"], f"Meeting notes:\n{text}\n\nQuestion: under the agenda item “{c['title']}”, what was discussed, and what was decided or assigned? "
                               f"Answer in 1-3 short sentences in your own words, about this item only, without copying the notes; if nothing was decided, say it was a discussion.{gl}", 170, prefill="Under this item")
                ans = ("Under this item" + ans) if ans else ""
            else:
                ans = llm.chat(P["SYS"], f"บันทึกการประชุม:\n{text}\n\nคำถาม: ในวาระ “{c['title']}” ที่ประชุมพูดคุยอะไร และมีมติหรืองานที่ต้องทำอะไรบ้าง? ตอบเป็นประโยคสั้นๆ 1-3 ประโยคด้วยคำของคุณเอง "
                               f"เฉพาะเรื่องของวาระนี้เท่านั้น ห้ามคัดลอกบันทึก ถ้าไม่มีมติให้บอกว่าเป็นการหารือ{gl}", 170, prefill="ในวาระนี้")
                ans = ("ในวาระนี้" + ans) if ans else ""
            part["agenda"][key] = " ".join((ans or ("summary failed" if en else "สรุปไม่สำเร็จ")).split())
            store.write_json(part_file, part)
        out.append(f"{c['index'] + 1}) **{c['title']}**: {part['agenda'][key]}")
        progress((n + 1) / max(1, len(coverage)))
    return "\n".join(out)


def person_lines(segs: list[dict], names: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in segs:
        if s.get("spk") and s["text"].strip():
            out.setdefault(names.get(s["spk"], s["spk"]), []).append(f"[{store.fmt_ts(s['t0'])}] {s['text']}")
    return out


def narratives(llm: LlamaServer, mid: str, segs: list[dict], names: dict[str, str], rep: dict, glossary: str, progress=lambda f: None, context: str = "") -> dict:
    """Per-person contribution notes (checkpointed). The participation overview is computed from the statistics, not by the LLM."""
    part_file, part = _checkpoint(mid)
    P = prompts(mid)
    gl = background(glossary, context, None, None, P)
    people = {}
    mine = person_lines(segs, names)
    for i, (name, lines) in enumerate(sorted(mine.items(), key=lambda kv: -len(kv[1]))):
        if name not in part["notes"]:
            if sum(len(x) + 1 for x in lines) > 2800:  # sample evenly across the meeting
                step = max(2, int(sum(len(x) for x in lines) / 2800) + 1)
                lines = lines[::step]
            part["notes"][name] = llm.chat(P["SYS"], P["PERSON"].format(name=name) + f"{gl}\n\n{P['NOTES']}:\n" + "\n".join(lines)[:3200], 350)
            store.write_json(part_file, part)
        people[name] = part["notes"][name]
        progress(0.7 + 0.25 * (i + 1) / max(1, len(mine)))
    return {"people": people}
