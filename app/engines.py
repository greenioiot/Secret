"""Which model does what for a meeting, in words for the Record tab: the live captions, the transcript made after the meeting,
the summary. 'plan' = what the settings ask for and what will really be used (a missing model falls back, as the worker does)."""
from __future__ import annotations

import config

ACCURACY = {  # accuracy mode -> name shown to the user
    "fast": "Typhoon realtime CTC (no second pass)",
    "accurate": "Thonburian Whisper medium + Typhoon cross-check",
    "npu": "Whisper-medium CTC on the NPU",
    "best": "Typhoon ASR Qwen 1.7B (with your names / agenda)",
    "pathumma-rnnt": "Pathumma FastConformer (NECTEC)",
    "vote3": "Best: 3-model vote (Pathumma Whisper Large + medium + FastConformer)",
    **{k: v[4] for k, v in config.WHISPER_CT2.items()},
}
SUMMARY = {
    "typhoon-4b": "Typhoon 2.5 Qwen3-4B", "typhoon-30b": "Typhoon 2.5 Qwen3-30B-A3B", "gemma4-e4b": "Gemma 4 E4B (Google)",
}
DEFAULT_WHISPER = "Thonburian Whisper medium"
# measured on this box (PERFORMANCE.md): character error rate on 80 FLEURS Thai clips, memory while it runs, where it runs, what else to know.
# Settings shows them side by side so the user can trade accuracy for time.
MEASURED = {
    "vote3": {"cer": 5.25, "ram_gb": 9.0, "where": "CPU", "note": "Lowest errors. Lines where the three models disagree show the other versions, one click to pick."},
    "pathumma-large": {"cer": 6.10, "ram_gb": 4.0, "where": "CPU", "note": "Best single model; slow."},
    "pathumma-rnnt": {"cer": 6.65, "ram_gb": 2.1, "where": "CPU", "note": "Fast and close to the best: the same model as the live captions."},
    "pathumma-medium": {"cer": 6.92, "ram_gb": 2.0, "where": "CPU", "note": ""},
    "accurate": {"cer": 7.17, "ram_gb": 1.5, "where": "CPU", "note": "Thonburian Whisper, checked against Typhoon."},
    "best": {"cer": 7.3, "ram_gb": 8.0, "where": "CPU", "note": "Spells your listed names and agenda terms (measured on 24 clips only)."},
    "large": {"cer": 8.65, "ram_gb": 3.0, "where": "CPU", "note": ""},
    "npu": {"cer": 10.2, "ram_gb": 0.5, "where": "NPU", "note": "Very fast; more errors."},
    "fast": {"cer": 12.7, "ram_gb": 0.3, "where": "NPU", "note": "Fastest; most errors."},
}
SUMMARY_MEASURED = {
    "typhoon-4b": {"ram_gb": 2.4, "where": "NPU", "note": "Faster and leaves the CPU free; can get some numbers wrong, so check figures."},
    "typhoon-30b": {"ram_gb": 18.6, "where": "CPU", "note": "Best in the side-by-side test: correct numbers, names and places; nothing invented."},
    "gemma4-e4b": {"ram_gb": 7.8, "where": "CPU", "note": "Google's Gemma 4 (2026): correct numbers and names in the side-by-side test, nothing invented; reads 50 and writes 11 tokens/s on the CPU (the NPU hangs with it)."},
}
LIVE = {"typhoon": "Typhoon realtime CTC", "wctc": "Whisper-medium CTC", "pathumma-rnnt": "Pathumma FastConformer (NECTEC)", "whisper-en": "Whisper small.en (English)"}
LANGUAGE = {"th": "Thai · ไทย", "th-en": "Thai + English · ไทยปนอังกฤษ", "en": "English only"}
ENGLISH_AFTER = "Whisper Large v3 Turbo (English)"
THAI_ONLY = ("fast", "npu", "pathumma-rnnt", "vote3", "best")  # accuracy modes whose models have no English (FastConformer, the CTC heads, the vote, the Qwen ASR)


def language_of(cfg: dict) -> str:
    lang = cfg.get("language") or "th"
    return lang if lang in config.LANGUAGES else "th"


def live_name(engine) -> str:
    """The loaded live-caption model (server.ctc()) in words, with where it runs."""
    if engine is None:
        return "not loaded"
    kind = getattr(engine, "kind", "typhoon")
    name = LIVE.get(kind, "Typhoon realtime CTC")
    return f"{name} · {'NPU' if getattr(engine, 'backend', 'cpu') == 'npu' else 'CPU'}"


def after_plan(cfg: dict) -> dict:
    """The model the worker will use for the transcript after the meeting: {"name", "ok", "note"} (ok False = it falls back)."""
    mode = cfg.get("accuracy", "accurate")
    name = ACCURACY.get(mode, mode)
    ok, note = True, ""
    default_ok = (config.WHISPER_TH / "model.bin").exists()
    lang = language_of(cfg)
    if lang == "en":  # the worker's pick_engine does the same
        ok = (config.WHISPER_EN / "model.bin").exists()
        return {"mode": "english", "name": ENGLISH_AFTER, "uses": ENGLISH_AFTER if ok else (DEFAULT_WHISPER if default_ok else "Typhoon realtime CTC only"), "ok": ok,
                "note": "" if ok else "the English Whisper model is not installed (models/whisper-en): the Thai models would be used"}
    if lang == "th-en" and (mode in THAI_ONLY or mode not in ("accurate", *config.WHISPER_CT2)):
        lv3 = (config.WHISPER_LV3 / "model.bin").exists()
        mode = "large" if lv3 else "accurate"
        note = f"Thai + English: {name} has no English, so {ACCURACY[mode]} is used"
        name = ACCURACY[mode]
    if mode in config.WHISPER_CT2:
        ok = (config.WHISPER_CT2[mode][1] / "model.bin").exists()
    elif mode == "pathumma-rnnt":
        import rnnt

        ok = rnnt.available()
    elif mode == "npu":
        import wctc

        ok = wctc.available()
    elif mode == "best":
        import qwen_asr

        ok = qwen_asr.available()
    elif mode == "vote3":
        import vote_asr

        ok = vote_asr.available()
    elif mode == "accurate":
        ok = default_ok
        if not ok:
            note = "the Thonburian Whisper model is missing: only the fast Typhoon pass will be made"
    uses = name
    if not ok and mode != "accurate":
        uses = DEFAULT_WHISPER if default_ok else "Typhoon realtime CTC only"
        note = f"{name} (chosen in Settings) is not installed"
    if cfg.get("live_accurate") and mode == "accurate":
        note = (note + "; " if note else "") + "the accurate pass also runs during the meeting (live accurate)"
    return {"mode": mode, "name": name, "uses": uses, "ok": ok, "note": note}


def live_plan(cfg: dict, live_engine=None) -> dict:
    """The live-caption model for this language: English-only meetings caption with Whisper small.en on the CPU (sentence by sentence)."""
    if language_of(cfg) == "en":
        ok = (config.WHISPER_EN_LIVE / "model.bin").exists()
        live = {"name": LIVE["whisper-en"] + " · CPU", "wanted": LIVE["whisper-en"]}
        if not ok:
            live["note"] = "the English live model is not installed (models/whisper-en/small.en-ct2): no live captions for English"
        elif live_engine is not None and getattr(live_engine, "kind", "") != "whisper-en":
            live["note"] = "captions come from the Thai model until the English one has loaded"
        else:
            live["note"] = "each sentence is captioned after it ends (about a second later)"
        return live
    want = cfg.get("live_engine", "typhoon") if cfg.get("live_engine") in LIVE else "typhoon"
    want_live = LIVE[want]
    live = {"name": live_name(live_engine) if live_engine is not None else want_live, "wanted": want_live}
    if live_engine is not None:
        if getattr(live_engine, "kind", "typhoon") != want:
            live["note"] = f"{want_live} could not load: the Typhoon realtime model is used"
        elif want == "typhoon" and cfg.get("asr_npu", True) and getattr(live_engine, "backend", "cpu") != "npu":
            live["note"] = "the NPU is not available: captions run on the CPU (slower)" + (f" ({live_engine.npu_error[:80]})" if getattr(live_engine, "npu_error", "") else "")
    return live


def summary_plan(cfg: dict) -> dict:
    choice = cfg.get("summary_model", "typhoon-4b")
    choice = choice if choice in config.SUMMARY_GGUF else "typhoon-4b"  # a removed model (Chinda, OpenThaiGPT, Pathumma) -> the summary falls back to the 4B
    name = SUMMARY.get(choice, choice)
    path = config.SUMMARY_GGUF.get(choice, config.LLM_GGUF)
    server = (config.LLAMA_BUILD / "bin/llama-server").exists()
    if not server:
        return {"name": name, "uses": "none", "ok": False, "note": "llama-server is missing: no summary can be written", "auto": bool(cfg.get("auto_summary"))}
    if path.exists():
        return {"name": name, "uses": name, "ok": True, "note": "" if cfg.get("auto_summary") else "made only when you ask (automatic summary is off)", "auto": bool(cfg.get("auto_summary"))}
    fb = config.LLM_GGUF.exists()
    return {"name": name, "uses": SUMMARY["typhoon-4b"] if fb else "none", "ok": False, "auto": bool(cfg.get("auto_summary")),
            "note": f"{name} (chosen in Settings) is not downloaded" if fb else "no summary model file is installed"}


def plan(cfg: dict, live_engine=None) -> dict:
    return {"live": live_plan(cfg, live_engine), "after": after_plan(cfg), "summary": summary_plan(cfg), "language": language_of(cfg),
            "speakers": bool(cfg.get("live_speakers")), "identify_speakers": bool(cfg.get("identify_speakers"))}
