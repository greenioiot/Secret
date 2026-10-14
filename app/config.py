"""Paths and defaults. Models are reused read-only from projectAI (Genie Edge) so nothing big is copied."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SECRET_DATA", ROOT / "data"))
MEETINGS = DATA / "meetings"
PROJECT_AI = Path(os.environ.get("PROJECT_AI", "/home/dev/projectAI"))

TYPHOON_ASR = PROJECT_AI / "models/typhoon-asr"                          # NeMo FastConformer CTC, ONNX (live + fast)
TYPHOON_NPU = ROOT / "models/typhoon-npu"                                # same Typhoon ASR: mel front end (ONNX, CPU) + encoder/CTC head (TFLite, Hexagon NPU, 16 s window)
QNN_DELEGATE = os.environ.get("QNN_TFLITE_DELEGATE", "/usr/lib/libQnnTFLiteDelegate.so")
WHISPER_LV3 = ROOT / "models/typhoon-whisper-lv3-ct2"  # Typhoon Whisper Large v3 (typhoon-ai, MIT), CTranslate2 int8 (converted here): optional "large" accuracy mode
PATHUMMA = ROOT / "models/pathumma"  # NECTEC Pathumma models (huggingface.co/nectec), converted / downloaded here
# Extra Whisper models for the accurate pass (CTranslate2 int8): accuracy mode -> (worker kind, model dir, engine label, RAM to wait for, name)
WHISPER_CT2 = {
    "large": ("lv3", WHISPER_LV3, "typhoon-whisper-large-v3", 4.0, "Typhoon Whisper Large v3"),
    "pathumma-medium": ("pwm", PATHUMMA / "whisper-th-medium-ct2", "pathumma-whisper-th-medium", 3.0, "Pathumma Whisper medium"),
    "pathumma-large": ("pwl", PATHUMMA / "whisper-th-large-v3-ct2", "pathumma-whisper-th-large-v3", 4.0, "Pathumma Whisper Large v3"),
}
PATHUMMA_RNNT = PATHUMMA / "stt-fastconformer-rnnt/pathumma-stt-th-fastconformer-rnnt-large.nemo"  # NeMo FastConformer-RNNT (CC-BY-4.0): accuracy mode 'pathumma-rnnt'
NEMO_PY = Path(os.environ.get("SECRET_NEMO_PY", "/mnt/genie/projectAI/export/venv-nemo/bin/python"))  # a venv with nemo_toolkit[asr] + torch (CPU)
WHISPER_TH = PROJECT_AI / "models/whisper-th/distill-whisper-th-medium"  # Thonburian, CTranslate2 (accurate pass)
LLAMA_BUILD = PROJECT_AI / "vlm/llama.cpp-snapdragon"                    # llama-server + libs (run on CPU only)
LLM_GGUF = PROJECT_AI / "models/typhoon2.5-qwen3-4b/typhoon2.5-qwen3-4b-q4_0.gguf"
LLM_GGUF_30B = ROOT / "models/typhoon-30b/typhoon2.5-qwen3-30b-a3b-q4_k_m.gguf"  # optional bigger summary model (Typhoon 2.5 on Qwen3-30B-A3B, 3B active; CPU only)
SUMMARY_GGUF = {}  # summary_model -> GGUF (filled below, after the paths); "typhoon-4b" = LLM_GGUF, the fallback
SUMMARY_GGUF.update({"typhoon-4b": LLM_GGUF, "typhoon-30b": LLM_GGUF_30B})  # Chinda 4B, OpenThaiGPT 8B and Pathumma 8B/4B were removed 2026-10-11 (tested worse, see PERFORMANCE.md)
SPEAKER_EMB = ROOT / "models/speaker/campplus.onnx"        # 3D-Speaker CAM++ (zh+en) speaker embeddings, ONNX
VOICES = DATA / "voices"                                    # enrolled voice profiles (biometric data: local only)
PENDING = DATA / "voices_pending"                           # voices sent by participants, waiting for the host's approval
ENROLL_FILE = DATA / "enroll.json"                          # the currently open enrolment link (token + expiry)
ENROLL_MIN_SPEECH_S = 8.0                                   # speech needed from a participant to make a usable voiceprint
ENROLL_MAX_PENDING = 30

PORT = int(os.environ.get("SECRET_PORT", "8444"))
HOST = os.environ.get("SECRET_HOST", "0.0.0.0")  # all network interfaces (phones on the LAN join / enrol); 127.0.0.1 = this device only
LLM_PORT = 8591 + 10 * (PORT - 8444)  # summary model; bot uses +1. Live (8444) = 8591; a second instance (test copy) gets its own ports


_SPAWNER = None


def spawn(cmd, **kw):
    """subprocess.Popen with die_with_parent, started from ONE long-lived thread. PR_SET_PDEATHSIG fires when the THREAD that created the
    child ends, not the process: a model server started from a short-lived thread (the server's warm-up thread) was killed seconds later."""
    global _SPAWNER
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    if _SPAWNER is None:
        _SPAWNER = ThreadPoolExecutor(1, thread_name_prefix="spawner")
    return _SPAWNER.submit(lambda: subprocess.Popen(cmd, preexec_fn=die_with_parent, **kw)).result()


def die_with_parent():
    """preexec_fn for model servers: they exit when the process that started them dies (no orphan holding the port / NPU memory)."""
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - best effort
        pass
HEARTBEAT = DATA / ".recording"        # touched by the server while a recording is live; the worker pauses on it
SETTINGS_FILE = DATA / "settings.json"
TOKEN_FILE = DATA / "access.token"

SR = 16000
CORES = os.cpu_count() or 8
WORKER_CORES = list(range(CORES // 2, CORES))  # cores 4-7: cores 0-3 stay free for the web server / live captions

# Read aloud when enrolling a voice. About 30 s for a person (local Thai TTS: 30.7 s; the old 20-s text took it 18.4 s), with whole
# numbers, decimals, a percentage and a large number so the voiceprint also covers how the person says numbers.
ENROLL_TEXT = "วันนี้อากาศดี อุณหภูมิประมาณ 28.5 องศา ฉันออกจากบ้านตั้งแต่เช้าเพื่อไปประชุมที่สำนักงานชั้น 12 ระหว่างทางแวะซื้อกาแฟ 2 แก้วกับขนมปัง 3 ชิ้น ในที่ประชุมเราคุยกันเรื่องแผนงานของไตรมาสหน้า ยอดขายเดือนที่แล้วเพิ่มขึ้น 4.75 เปอร์เซ็นต์ เป็น 1,280,000 บาท งบประมาณรวม 3.2 ล้านบาท แบ่งให้ 6 ทีม ทีมละประมาณ 0.53 ล้านบาท กำหนดส่งงานรอบแรกคือวันที่ 15 ขอบคุณทุกคนที่สละเวลามาร่วมประชุมในวันนี้"
ENROLL_TEXT_MAX = 1500

DEFAULT_SETTINGS = {
    "mode": "live",              # live = live captions while recording, after = record only, transcribe afterwards
    "accuracy": "pathumma-rnnt",      # accurate = Thonburian Whisper + Typhoon cross-check, best = Typhoon ASR Qwen 1.7B with contextual biasing (slow), large = Typhoon Whisper Large v3 (slow), npu = Whisper-medium CTC on the NPU (~30x faster than accurate), fast = Typhoon only
    "silence_ms": 1000,          # Thai end-of-speech silence (projectAI lesson: >= 1 s)
    "auto_process": True,        # transcribe automatically after the recording ends
    "auto_summary": True,        # summarize automatically after transcription
    "batch_night": True,         # the work after a meeting waits for the processing window (schedule.py): out of working hours, set by an administrator
    "batch_start": "18:00",      # the window opens (local time) ...
    "batch_end": "06:00",        # ... and closes the next morning (start = end: all day)
    "batch_weekends": True,      # Saturday and Sunday are open all day
    "glossary": "",              # names / jargon, one per line or comma separated: biases Whisper and the summary
    "corrections": "",           # "wrong=>right" per line, applied to every transcript
    "mic_echo_cancel": False,    # browser DSP: off by default, it smears speech in a meeting room
    "mic_noise_suppress": False,
    "mic_auto_gain": True,
    "cross_meeting_match": False, # admin "Who is this?" page: also compare unnamed participants' voices across meetings (biometric: off by default)
    "identify_speakers": True,   # tell speakers apart (needs a few seconds of extra processing per meeting)
    "expected_speakers": 0,      # 0 = find out automatically
    "separate_overlaps": True,   # two people talking at once: split the voices and give each its own line (separation.py)
    "guest_mic": True,  # participants who joined by link may send their own phone's microphone (mics.py): their words, cleanly, even when people talk at once
    "speaker_threshold": 0.5,    # cosine similarity needed to treat two utterances as the same voice
    "voice_match_threshold": 0.5,  # similarity needed to auto-name a speaker from an enrolled voice
    "live_speakers": True,       # provisional speaker labels on live captions
    "allow_join": False,         # OPTIONAL: let people on the local network follow a meeting from their own device (off by default)
    "join_add_attendee": True,   # guests who join are added to the meeting's participants
    "summary_model": "typhoon-4b",  # typhoon-4b = Typhoon2.5-Qwen3-4B (NPU, fast) | typhoon-30b = Typhoon2.5-Qwen3-30B-A3B (CPU, better Thai, slower)
    "summary_on_npu": True,      # the summary model (Typhoon 4B Q4_0) runs on the Hexagon NPU: prompt reading ~7x faster; falls back to the CPU
    "live_accurate": True,       # run the accurate (Whisper) pass on finished speech DURING the meeting (cores 4-7, low priority) so little is left after it
    "speakers_after": True,      # transcribe first (text as soon as possible); who-said-what is added afterwards as a post task
    "numbers_arabic": True,      # write numbers as Arabic digits (15, 500,000) instead of Thai number words
    "english_words": True,       # write English words in English letters (campaign, IT, OKR) instead of Thai transliteration
    "live_partial": True,        # interim captions while a sentence is being spoken (only when the ASR runs on the NPU)
    "live_engine": "pathumma-rnnt",    # live captions: typhoon = Typhoon realtime CTC (fast, 12.7 % on public clips) | wctc = Whisper-medium CTC on the NPU (10.2 %, interim text every 2 s)
    "asr_npu": True,             # run the Typhoon ASR encoder on the Hexagon NPU (CPU fallback if it cannot load); False = CPU only
    "allow_enroll": False,       # OPTIONAL: participants may enrol their own voice from their device (host opens a link, then approves)
    "max_guests": 30,            # most people who may join a meeting from the network (the real limit can be lower: see capacity.py)
    "reserve_ram_gb": 2.0,       # RAM always kept free for the recording and live captions; new guests are refused below it
    "cpu_limit_pct": 80,         # new guests are refused when the CPU would go above this
    "guide_name": "น้องแนะนำ",    # the enrolment guide's name
    "guide_lang": "auto",        # auto = Thai + English together, or "th" / "en"
    "guide_wifi": "",            # e.g. the Wi-Fi name and password to show participants
    "meeting_bot_ui": False,     # show the (advanced) meeting-conducting bot in the interface; the enrolment guide is always available
    "use_materials": True,       # use a meeting's agenda + uploaded documents: speech-recognition prompt, term correction, summary context
    "apply_term_corrections": True,  # repair near-miss spellings of known terms in the transcript (originals are kept)
    "handsfree_enroll": False,   # EXPERIMENTAL: spoken, hands-free voice enrolment in the Voices tab (wake phrase + spoken guidance; needs Start in the browser)
    "handsfree_wake": "Secrets", # the name in the wake phrase ("Hey Secrets, enroll my voice" / "เฮ้ ซีเคร็ท ลงทะเบียนเสียง")
    "handsfree_lang": "th",      # th | en: language of the spoken guidance
    "tts_engine": "piper",       # spoken guidance voice: piper = Piper th_TH tsync2 (fast) | pathumma-f5 = NECTEC Pathumma F5-TTS (natural, slow on the CPU)
    "enroll_text": "",           # the paragraph people read aloud when enrolling a voice; "" = config.ENROLL_TEXT
    "res_cpu": True,             # header bar: which gauges to show (CPU / RAM / NPU / GPU / disk); the details panel always has everything
    "res_ram": True,
    "res_npu": True,
    "res_gpu": False,            # read-only GPU clock / temperature / who has it open (Secrets never uses the GPU)
    "res_disk": True,
    "summary_use_passages": False,   # also quote relevant passages of the documents to the summary model (measured: no gain, small risk of mixing them into what was said)
}
JOIN_MAX = 30                    # guests per meeting
JOIN_PER_IP = int(os.environ.get("SECRET_JOIN_PER_IP", "6"))                  # guest connections per device address
