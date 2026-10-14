"""Live system monitor, capacity model and admission control ("keep the system smooth").

Monitor  : a thread samples /proc every 2 s: CPU (total + per core), RAM, swap, temperature, the biggest memory users, and the NPU.
           The box exposes NO NPU utilisation counter, so the NPU figure is an ESTIMATE: which processes hold the compute-DSP device
           (/dev/fastrpc-cdsp*) open, and how much CPU their NPU-driver threads (thr_h*, qnn_*, iou-sqp) use.
Capacity : plan() turns the current numbers and the user's limits into "how many more guests can join right now", and whether the
           bot's language model fits in memory. The per-guest costs below were measured on this box (see PROGRESS.md).
Admission: admit_guest() / admit_enroll() refuse new people when RAM or CPU would be pushed past the limits, so the recording and
           the live captions keep running smoothly.
"""
from __future__ import annotations

import glob
import os
import re
import threading
import time
from pathlib import Path

import config

HZ = os.sysconf("SC_CLK_TCK")
CORES = os.cpu_count() or 8
NPU_THREADS = ("thr_h", "qnn_", "iou-sqp")
THERMAL = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))

# Measured costs (re-measure with tools/bench_capacity.py after big changes). GB / MB are resident memory, cpu is % of ALL cores.
# Measured 2026-10-06 with tools/bench_capacity.py (120 guests + a live meeting): +0.15 MB RAM and ~0.01-0.03 % CPU per guest, live meeting ~2 % CPU.
# The values below add a safety margin (x2). The bot language model is the file size + context memory (see PROGRESS.md).
# Re-measured 2026-10-07 (captions on the NPU, 500 connected guests, one shared address): +0.36 MB RAM and ~0.0007 % CPU per guest,
# live meeting 1.25 % CPU. Margins ~1.5x.
COST = {"guest_mb": 0.5, "guest_cpu_pct": 0.002, "live_cpu_pct": 2.0, "bot_llm_gb": 4.0, "bot_embed_gb": 0.55, "post_gb": 4.5,
        "enroll_cpu_pct": 12.0}
LABELS = (("uvicorn server:app", "Genie Edge app"), ("llama-server", "llama.cpp server"), ("app/worker.py", "Secrets worker"), ("worker.py", "Secrets worker"),
          ("app/server.py", "Secrets web"), ("chrome", "Chrome"), ("claude", "Claude Code"))


def _read_stat() -> list[tuple[int, int]]:
    out = []
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu"):
                v = [int(x) for x in line.split()[1:9]]
                out.append((sum(v), v[3] + v[4]))  # total, idle+iowait
    return out


def read_mem() -> dict:
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            m[k] = int(v.split()[0]) / 1048576  # GB
    total, avail = m.get("MemTotal", 0), m.get("MemAvailable", 0)
    return {"total_gb": round(total, 2), "avail_gb": round(avail, 2), "used_gb": round(total - avail, 2), "pct": round(100 * (1 - avail / max(total, 1e-9)), 1),
            "swap_used_gb": round(m.get("SwapTotal", 0) - m.get("SwapFree", 0), 2)}


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


BOT_LLM_PORT = config.LLM_PORT + 1  # keep in step with bot.LLM_PORT


def _label(cmd: str) -> str:
    if "llama-server" in cmd and f"--port {BOT_LLM_PORT} " in cmd + " ":
        return "Secrets bot LLM (llama.cpp)"
    if "llama-server" in cmd and f"--port {config.LLM_PORT} " in cmd + " ":
        return "Secrets summary LLM (llama.cpp)"
    for key, label in LABELS:
        if key in cmd:
            return label
    return (cmd.split()[0].rsplit("/", 1)[-1] if cmd else "?")[:24]


def _temp_c() -> float | None:
    best = None
    for p in THERMAL:
        try:
            t = int(p.read_text()) / 1000
            if 0 < t < 150:
                best = t if best is None else max(best, t)
        except (OSError, ValueError):
            pass
    return round(best, 1) if best is not None else None


GPU_DEVFREQ = "/sys/class/devfreq/3d00000.gpu"  # Adreno on this board (read only: Secrets never uses the GPU, Vulkan/Turnip hard-resets the box)


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def gpu_info(holders: list[dict]) -> dict:
    cur, mx = _read(GPU_DEVFREQ + "/cur_freq"), _read(GPU_DEVFREQ + "/max_freq")
    temps = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        if "gpu" in (_read(z + "/type") or "").lower():
            try:
                temps.append(int(_read(z + "/temp") or 0) / 1000)
            except ValueError:
                pass
    return {"present": cur is not None, "freq_mhz": int(cur) // 1_000_000 if cur and cur.isdigit() else None, "max_mhz": int(mx) // 1_000_000 if mx and mx.isdigit() else None,
            "temp_c": round(max(temps), 1) if temps else None, "holders": holders, "state": "in use" if holders else "idle",
            "note": "No load counter is exposed. Secrets never uses the GPU (it can hard-reset this board): this only shows which of your processes have it open, its clock and temperature."}


def disk_info() -> dict:
    import shutil

    try:
        u = shutil.disk_usage(config.DATA)
    except OSError:
        return {}
    return {"total_gb": round(u.total / 1e9, 1), "used_gb": round(u.used / 1e9, 1), "free_gb": round(u.free / 1e9, 1), "pct": round(100 * u.used / max(1, u.total), 1)}


def _tree_bytes(path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


class Monitor(threading.Thread):
    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True, name="monitor")
        self.period = period
        self.latest: dict = {}
        self._ts = time.time()
        self._stat = _read_stat()
        self._npu_ticks: dict[tuple[int, int], int] = {}
        self._procs: dict = {"top": [], "holders": [], "gpu_holders": [], "secrets": {}}
        self._procs_at = 0.0
        self._data = {"meetings_gb": None, "voices_mb": None, "at": 0.0}

    # -- processes (every ~6 s): memory users + who holds the NPU device
    def _scan(self):
        me, uid = os.getpid(), os.getuid()
        top, holders, gpu, secrets = [], [], [], {"web_mb": 0.0, "worker_mb": 0.0, "llm_mb": 0.0, "bot_llm_mb": 0.0}
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            try:
                if os.stat(f"/proc/{pid}").st_uid != uid:
                    continue
            except OSError:
                continue
            rss = _rss_mb(pid)
            if rss < 30:
                continue
            cmd = _cmdline(pid)
            if not cmd:
                continue
            lab = _label(cmd)
            if pid == me:
                lab, secrets["web_mb"] = "Secrets web", rss
            elif lab == "Secrets worker":
                secrets["worker_mb"] += rss
            elif lab == "Secrets summary LLM (llama.cpp)":
                secrets["llm_mb"] += rss
            elif lab == "Secrets bot LLM (llama.cpp)":
                secrets["bot_llm_mb"] += rss
            top.append({"pid": pid, "name": lab, "rss_mb": round(rss)})
            try:
                links = [os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")]
            except OSError:
                links = []
            if any("fastrpc" in x for x in links):
                holders.append({"pid": pid, "name": lab})
            if any(x.startswith(("/dev/dri/renderD", "/dev/kgsl")) for x in links):
                gpu.append({"pid": pid, "name": lab})
        top.sort(key=lambda p: -p["rss_mb"])
        self._procs = {"top": top[:6], "holders": holders, "gpu_holders": gpu, "secrets": {k: round(v) for k, v in secrets.items()}}
        self._procs_at = time.time()
        if time.time() - self._data["at"] > 120:  # what Secrets keeps on disk (walking the folders costs a little: every 2 min)
            self._data = {"meetings_gb": round(_tree_bytes(config.MEETINGS) / 1e9, 2), "voices_mb": round(_tree_bytes(config.VOICES) / 1e6, 1), "at": time.time()}

    def _npu(self, dt: float) -> dict:
        holders = self._procs["holders"]
        ticks: dict[tuple[int, int], int] = {}
        for h in holders:
            pid = h["pid"]
            try:
                for tid in os.listdir(f"/proc/{pid}/task"):
                    with open(f"/proc/{pid}/task/{tid}/stat") as f:
                        s = f.read()
                    comm = s[s.index("(") + 1:s.rindex(")")]
                    if comm.startswith(NPU_THREADS):
                        v = s[s.rindex(")") + 2:].split()
                        ticks[(pid, int(tid))] = int(v[11]) + int(v[12])  # utime + stime
            except (OSError, ValueError):
                pass
        delta = sum(max(0, t - self._npu_ticks.get(k, t)) for k, t in ticks.items())
        self._npu_ticks = ticks
        pct = min(100.0, round(100 * delta / HZ / max(dt, 0.5), 1))
        state = "idle" if not holders else ("active" if pct >= 3 else "loaded")
        return {"state": state, "driver_pct": pct if holders else 0.0, "holders": holders, "estimate": True,
                "note": "No hardware counter is exposed: this is the CPU time of NPU driver threads in the processes that hold the NPU."}

    def run(self):
        while True:
            try:
                time.sleep(self.period)
                now = time.time()
                dt = now - self._ts
                cur = _read_stat()
                per = [round(100 * (1 - (b[1] - a[1]) / max(1, b[0] - a[0])), 1) for a, b in zip(self._stat, cur)]
                self._stat, self._ts = cur, now
                if now - self._procs_at > 6:
                    self._scan()
                mem = read_mem()
                la = os.getloadavg()
                self.latest = {"time": now, "cpu_pct": per[0], "per_core": per[1:], "cores": CORES, "load": [round(x, 2) for x in la], "temp_c": _temp_c(),
                               "mem": mem, "npu": self._npu(dt), "procs": self._procs["top"], "secrets": self._procs["secrets"],
                               "gpu": gpu_info(self._procs["gpu_holders"]), "disk": {**disk_info(), "meetings_gb": self._data["meetings_gb"], "voices_mb": self._data["voices_mb"]}}
            except Exception:  # noqa: BLE001 - a bad sample must never stop the monitor
                time.sleep(3)

    def snapshot(self) -> dict:
        """Latest sample (instant values if the thread has not produced one yet)."""
        s = self.latest
        if s and time.time() - s["time"] < 10:
            return s
        mem = read_mem()
        return {"time": time.time(), "cpu_pct": 0.0, "per_core": [], "cores": CORES, "load": [round(x, 2) for x in os.getloadavg()], "temp_c": _temp_c(), "mem": mem,
                "npu": {"state": "unknown", "driver_pct": 0.0, "holders": [], "estimate": True, "note": ""}, "procs": [], "secrets": {}, "gpu": gpu_info([]), "disk": disk_info()}


MONITOR = Monitor()


# ------------------------------------------------------------------ capacity model
def limits(cfg: dict) -> dict:
    return {"max_guests": int(cfg.get("max_guests", 30)), "reserve_gb": float(cfg.get("reserve_ram_gb", 2.0)), "cpu_limit": float(cfg.get("cpu_limit_pct", 80))}


def plan(cfg: dict, guests: int = 0, recording: bool = False, bot_llm: bool = False, snap: dict | None = None) -> dict:
    """How many MORE guests can join right now, and why. `guests` = people already joined, `bot_llm` = the meeting's bot wants the language model."""
    s = snap or MONITOR.snapshot()
    L = limits(cfg)
    avail, cpu = s["mem"]["avail_gb"], s["cpu_pct"]
    llm_up = s.get("secrets", {}).get("bot_llm_mb", 0) > 500  # the BOT's own model server (the summary worker's one is a different process)
    llm_need = COST["bot_llm_gb"] + COST["bot_embed_gb"] if (bot_llm and not llm_up) else 0.0
    ram_room = avail - L["reserve_gb"] - llm_need
    by_ram = max(0, int(ram_room * 1024 / COST["guest_mb"]))
    cpu_room = L["cpu_limit"] - cpu - (0.0 if recording else COST["live_cpu_pct"])
    by_cpu = max(0, int(cpu_room / COST["guest_cpu_pct"]))
    by_cap = max(0, L["max_guests"] - guests)
    more = min(by_cap, by_ram, by_cpu)
    why = {"your limit": by_cap, "free RAM": by_ram, "free CPU": by_cpu}
    limiter = min(why, key=why.get)
    warnings = []
    if avail < L["reserve_gb"]:
        warnings.append(f"Only {avail:.1f} GB of RAM is free (reserve {L['reserve_gb']:.1f} GB): new people are refused to protect the recording.")
    if cpu > L["cpu_limit"]:
        warnings.append(f"CPU is at {cpu:.0f}% (limit {L['cpu_limit']:.0f}%).")
    if s.get("temp_c") and s["temp_c"] > 85:
        warnings.append(f"The board is hot ({s['temp_c']:.0f} °C): it may slow down.")
    if bot_llm and not llm_up and avail < COST["bot_llm_gb"] + COST["bot_embed_gb"] + L["reserve_gb"]:
        warnings.append(f"The bot's language model needs about {COST['bot_llm_gb'] + COST['bot_embed_gb']:.1f} GB plus the reserve: it will run in simple (rule-based) mode.")
    post_ok = avail >= COST["post_gb"] or llm_up
    return {"more_guests": more, "total_guests": guests + more, "limiter": limiter, "by": why, "avail_gb": avail, "cpu_pct": cpu,
            "bot_llm_ok": llm_up or avail >= COST["bot_llm_gb"] + COST["bot_embed_gb"] + L["reserve_gb"], "bot_llm_running": llm_up,
            "post_processing_ok": post_ok, "limits": L, "costs": COST, "warnings": warnings,
            "text": (f"{more} more people can join right now (limited by {limiter}). Free RAM {avail:.1f} GB, CPU {cpu:.0f}%." if more
                     else f"No more people can join right now (limited by {limiter}). Free RAM {avail:.1f} GB, CPU {cpu:.0f}%.")}


def admit_guest(cfg: dict, guests: int, recording: bool = True, bot_llm: bool = False) -> tuple[bool, str]:
    p = plan(cfg, guests, recording, bot_llm)
    if p["more_guests"] > 0:
        return True, ""
    return False, {"your limit": f"The host limited this meeting to {p['limits']['max_guests']} people.",
                   "free RAM": "The device is low on memory right now, so no more people can join. Please try again in a minute.",
                   "free CPU": "The device is very busy right now, so no more people can join. Please try again in a minute."}[p["limiter"]]


def admit_enroll(cfg: dict, active: int) -> tuple[bool, str]:
    s = MONITOR.snapshot()
    L = limits(cfg)
    if active >= 3:
        return False, "Other people are recording their voice right now. Please try again in a moment."
    if s["mem"]["avail_gb"] < L["reserve_gb"]:
        return False, "The device is low on memory right now. Please try again in a minute."
    if s["cpu_pct"] + COST["enroll_cpu_pct"] > 100 or s["cpu_pct"] > L["cpu_limit"] + 10:
        return False, "The device is very busy right now. Please try again in a minute."
    return True, ""
