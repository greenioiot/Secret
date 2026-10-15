# Zecrets – on-device Thai meeting recorder

Browser mic → crash-safe disk → live Thai captions → accurate transcript → Thai summary. Everything stays on this box.

Run: `./run.sh` (web app + worker) → open `https://<box-ip>:8444` (self-signed cert), access code in `data/access.token`.
Logs: `data/server.log`, `data/worker.log`, `data/llm.log`. Resume notes for development: `PROGRESS.md`.

## How it works
- `app/server.py` FastAPI + WebSocket. The browser sends 16 kHz PCM16 (AudioWorklet); it is appended to `audio.pcm` and fsynced every second.
  Live captions: energy VAD (Thai end-of-speech 1 s) + Typhoon ASR CTC (ONNX, CPU).
- `app/worker.py` separate low-priority process (nice 10, cores 4-7, oom_score_adj 800). Stages are checkpointed:
  identify speakers → transcribe (speaker-turn chunks → Thonburian Whisper + Typhoon cross-check → `transcript.json/.md`) then summarize
  (`app/summarize.py`, Typhoon2.5-Qwen3-4B Q4_0 on CPU, map-reduce, server stopped afterwards → `summary.md`).
- `data/meetings/<id>/`: `state.json` (source of truth), `audio.wav`, `live.jsonl`, `transcript.*`, `summary.md`.

## When something goes wrong
| Failure | What happens |
|---|---|
| Network drops during recording | Browser keeps recording (5 min buffer), reconnects and takes over the same meeting |
| Browser/tab closed, server killed, power loss | Meeting becomes *interrupted*; audio up to the last second is on disk. Banner on the Record tab: **Continue** or **Finish & process** |
| Worker crashes / out of memory | Server restarts it; it resumes from `transcript.partial.jsonl` / `summary.partial.json` (max 3 attempts, then status *error* + Retry button) |
| Disk < 1.5 GB | Recording refuses to start; < 0.5 GB stops it cleanly |

## Models (reused read-only from projectAI, nothing copied)
`models/typhoon-asr` (live + fast), `models/whisper-th/distill-whisper-th-medium` (accurate), `models/typhoon2.5-qwen3-4b/*q4_0.gguf` + `vlm/llama.cpp-snapdragon` (summary).
Override the location with `PROJECT_AI=/path`. Never use the GPU/Vulkan on this box (it hard-resets it).

## Speakers, voices and report
Settings → Speakers. Utterances are clustered by voice (CAM++ embeddings, `app/speakers.py`); name them in the meeting's **Speakers** tab and tick *Remember this voice*
(or enrol people in the **Voices** tab) so later meetings name them automatically. **Report** tab: talk time, turns, questions, balance, per-person AI notes.
Voice profiles are biometric data: stored only in `data/voices/`, deletable in the UI, enrol people with their consent.
Old meetings: Speakers tab → *Identify speakers now* (does not repeat the transcription).
Access code: Settings → Access code (or delete `data/access.token` and restart for a new random one).

## Meetings table, search, delete, planned meetings with documents
* **Meetings tab** is a table: search keywords (title, people, remarks, summary, transcript; Thai and English, case-insensitive), filters (status, date/time range, person, duration, has summary/report/agenda/documents…), sortable columns, paging (10-200 per page), CSV export, select rows (or *Select all N matching*) and **Delete** with a confirmation (a meeting that is recording or being processed cannot be deleted).
* **Plan a meeting** (button above the table): title, planned date/time, participants, agenda, **documents** (.txt .md .csv .pdf .docx). Start the recording from the plan (Record tab → *Start from a planned meeting*, or *Start* in the table).
  The system reads the documents on this device to find names and terms (editable *Key terms*), repairs their spelling in the transcript (the original text is kept, shown with ✎), tells the summary the agenda and which items were never discussed, and adds a per-agenda-item section.
  Whisper itself gets no prompt: any prompt makes this Thai model much worse (measured).

## Word collection and participants
* **Word collection** (Settings): save "wrong word → right word"; applied to live captions (also what participants see) and to every transcript. Correct a word in any transcript line and Zecrets offers to remember it (and fix the same word in the rest of that meeting). *Apply saved words* fixes an older transcript.
* **Participants** who join over the network see only the live transcript (with speaker names and times). No bot messages, questions, summary or reports.
* **How many participants?** Measured on this board: ~0.4 MB RAM and almost no CPU per participant; hundreds fit. The practical limit is the Wi-Fi access point; set it in Settings → Capacity & limits.

## Writing style: digits and English words
Numbers are written as digits (`ห้าแสนบาท` → `500,000 บาท`, `วันที่สิบห้าพฤศจิกายน` → `วันที่ 15 พฤศจิกายน`) and English words in English letters (`แคมเปญ` → `campaign`, `โอเคอาร์` → `OKR`), in live captions, transcripts and summaries (Settings → Word collection; *Apply saved words & writing style* fixes an old transcript). The English list is `app/loanwords.tsv`; anything else can be added as a word-collection rule.

## Typhoon ASR on the NPU
Live captions show the sentence **while it is spoken** (grey italic, refreshed about once a second, also for participants) and replace it with the final caption when the speaker pauses; Settings → *Speech recognition on the NPU* can switch this off. The compiled NPU graph is cached in `models/typhoon-npu/cache/` so loading takes 0.3 s instead of ~65 s.
The live captions and the quick transcript run the Typhoon ASR realtime encoder on the Hexagon NPU (Settings → *Speech recognition on the NPU*, on by default): ~55 ms per utterance instead of 100-470 ms on the CPU, and the CPU is free for the web app. The mel front end stays on the CPU, the encoder is a fixed 16 s TFLite graph (`models/typhoon-npu/`, compiled with Qualcomm AI Hub from the public CC-BY model; export/compile scripts are in that folder). If the NPU cannot be used the CPU model takes over automatically. Measured on one synthetic Thai meeting: CER 4.8 % on the NPU vs 4.2 % on the CPU (fp16). The accurate Whisper pass stays on the CPU.

## Enrolment guide (explains, guides, tells the address)
Voices tab → open the enrolment link → **Show the guide on a screen** (or open `/guide`). The room display shows a big QR code, the link, the system address (e.g. `https://192.168.1.161:8444/`),
the 6 steps in Thai/English and rotating tips, plus who is waiting for approval. On the participant's phone the guide explains each step, coaches while they read
(too quiet / too loud / keep going), turns errors into plain next steps and answers questions about enrolment only (it politely refuses anything else). It is rule-based, so it is instant and never invents facts.
Settings → *Enrolment guide*: its name, language and a Wi-Fi note. The meeting-conducting bot below is hidden unless you switch it on in Settings → *Meeting bot (advanced)*.

## Meeting bot, knowledge base, limits (Round 7)
* **Bot tab**: defaults, agenda templates and the **knowledge base** (upload/paste documents, test a question). **Record tab → Meeting bot**: goal, agenda (minutes per item), rules, style; while recording a panel shows the current item and timer
  (Next / Previous / Pause), what the bot said and an *Ask the bot* box. The bot works without the language model (timers, nudges, quoting your documents) and uses the model for short answers, recaps and a closing recap when memory allows.
  Guests can ask it too if you allow that (rate limited). Everything it says is saved in the meeting (Bot tab of the meeting).
* **End meeting for everyone**: red banner on any other device while a meeting records, button in the Meetings list and on the meeting page.
* **Header bar CPU / RAM / NPU** (click for details). The NPU figure is an estimate: the board exposes no NPU counter.
* **Capacity**: Settings → *Capacity & limits*. People are refused when free RAM or CPU would drop below your limits; *Calculate capacity now* explains why. Measured: a guest costs ~0.15 MB and almost no CPU; the bot's language model needs ~4.5 GB. Re-measure on a new board: `.venv/bin/python tools/bench_capacity.py --wav speech.wav`.

## Participants enrol their own voice (optional)
Settings → *Participants enrol their own voice* (off by default). In the **Voices** tab press *Open enrolment link*: a QR code and link appear (open 15–60 min, new link each time).
Participants on the same network open it on their phone, type their name, tick the consent box and read a Thai text aloud (~20 s). Their recording is not saved, only a voiceprint,
and it waits in a list until **you approve it** (the list warns about duplicate names or voices that sound like someone already enrolled). The app's network address is shown in Settings
(for example `https://192.168.1.161:8444/`); the main app still needs the access code. Voiceprints are biometric data: collect consent and delete voices in the Voices tab on request.

## E-mail (optional)
Settings → *Email* (off by default): your SMTP server, account and default recipients; *Send test e-mail* checks it. Then use *Email…* on a meeting, or tick *send automatically* to mail the summary
and participation report once each meeting has been summarized. Gmail/Outlook/Yahoo need an **app password**. The password is kept in `data/smtp.json` (private file) and is never shown again;
the audio is never attached. The meeting text leaves the device for your mail provider when you use this.

## Join from the local network (optional)
Settings → *Allow participants to join* (off by default). Then, on the Record tab, tick *Let participants join from this network*: a QR code and link appear.
People on the same Wi-Fi/LAN open it, type their name and follow the live captions on their own phone (microphones are not used, nothing is recorded from their devices).
The link only works while the meeting is being recorded, a new link is made each time, and *Stop joining* closes it. Guests cannot reach recordings, other meetings or settings.
Their device will warn about the self-signed certificate once. Names are self-declared. Install: `pip install -r requirements.txt` (includes `segno` for the QR code).

## Meeting details
Each meeting has a name, a remark and a participants list (Record tab before/during recording, or the meeting page later). Attendees who never speak are shown in the report as *did not speak*;
name speakers with the same spelling as the participant list so they match. The remark and names are background for the summary and spelling hints for the transcription.

## Not done yet
Start-at-boot service, benchmark on real meeting audio, tuning voice-matching thresholds on real Thai voices.
