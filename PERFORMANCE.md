# Secrets: what each step uses, and how fast and accurate it is (2026-10-11)

All numbers were measured on this box (8-core Arm CPU with 4 fast + 4 slow cores, 34 GB RAM, a Hexagon NPU with 2 cores) unless marked otherwise.

**How to read the numbers:**
- **Error rate (CER):** the share of characters wrong, after normalising both texts. Lower is better.
- **Speed (× real time):** processing time ÷ audio length. 0.1 means a 1-hour meeting takes about 6 minutes.

★ marks the current default.

## 1. Live captions (during the meeting)

| Model | Runs on | Time per sentence | Error rate | Notes |
|---|---|---|---|---|
| ★ Pathumma FastConformer (NECTEC) | CPU, 4 threads | 0.94 s per 8-s sentence | 6.65% | Interim text every 3 s; 2.1 GB RAM, always loaded (starts in 15 s) |
| Typhoon realtime CTC | NPU (CPU fallback) | about 0.1 s on the NPU; 0.2 s per 4-s sentence on the CPU | 12.7% | Lightest; the fallback when NeMo is missing |
| Whisper-medium CTC | NPU | 0.71 s median, 0.97 s 95th percentile | 10.2% | Fixed 30-s window |

Error rates are on the FLEURS Thai test set (80 clips, 15.9 min).

## 2. Transcript after the meeting

FLEURS Thai, 80 clips, 15.9 min, 8 threads:

| Mode | Model | Error rate | Speed | 1-hour meeting takes | RAM |
|---|---|---|---|---|---|
| ★ `vote3` (since Round 43) | Pathumma Whisper Large + Pathumma Whisper medium + FastConformer: the text the other two agree with most | **5.25%** (lowest) | 3.9–4.6× | about 4–5 h | about 9 GB |
| `pathumma-rnnt` | Pathumma FastConformer-RNNT | 6.65% | 0.10× | about 6 min | 2.1 GB |
| `pathumma-large` | Pathumma Whisper Large v3 (CT2 int8) | 6.10% (best single model) | 3.06× | about 3 h | 4 GB |
| `pathumma-medium` | Pathumma Whisper medium (CT2 int8) | 6.92% | 1.63× | about 1 h 40 min | about 2 GB |
| `accurate` | Thonburian Whisper medium + Typhoon cross-check | 7.17% | 0.54× | about 32 min | about 1.5 GB |
| `best` | Typhoon ASR Qwen 1.7B with names/agenda as context | 7.3%* | 1.7× | about 1 h 40 min | 8 GB |
| `npu` | Whisper-medium CTC on the NPU | 10.2% | about 0.04× | about 2.5 min | NPU |
| `fast` | Typhoon realtime only | 12.7% | 0.01× | under 1 min | NPU |

\* The `best` mode was measured on the first 24 clips only. Its real strength is spelling names and terms exactly as listed, which this test set does not measure.

Real meetings with ★: a 1-minute meeting was transcribed in 21 s, and a 5.4-minute one in 41 s.

## 3. Who is speaking (speaker identification)

| Step | Method | Speed | Result |
|---|---|---|---|
| Speaker identification | Silero speech detector, then CAM++ voiceprints, then grouping | about 0.05× | 10.6 min, 4 people: 27 s. 125 min: 123 s |
| **New:** overlap detection | pyannote segmentation-3.0 (ONNX, CPU) | **0.006×** (125 min in 36 s) | It found the core of every overlap in all 12 test conversations and never flagged a stretch where only one voice was talking |
| **New:** overlapped stretches left out when learning voices | (same detector) | free | Before: in 2 of 12 test conversations, the two people were merged into one. After: 0 of 12 |
| **New:** voice separation | ★ Conv-TasNet (Libri2Mix noisy, 16 kHz), as ONNX since Round 42 | **0.14×** (24 s in 3.5 s; it was 0.3× with torch) | Error rate on the overlapped part: **47.7% → 20.6%** (Round 41), 19.1% / 19.4% for A / B (Round 42) |
| (rejected) voice separation | SepFormer WHAMR! 16 kHz | 5× (20 s in 99 s) | Same accuracy (28.3% vs 28.7% in the same test), but 14× slower |
| **Round 42:** short remarks said over someone | The same separation, on the overlap ± 0.5 s; an echo filter (voice and words) | Seconds per meeting | Interjections of 0.6–1.6 s: B's error rate 102% → 71%. Still missed: those at −12 dB, which the overlap detector does not hear |
| (rejected) scanning every line for a quiet second voice | Conv-TasNet on all speech | +1.2× real time | Found 1 more interjection out of 6, but added 3 false lines |
| **Round 42:** participants' own phone microphones | Phone audio aligned to the room recording (energy cross-correlation, clock drift), owner's voice by level and voiceprint | 3–12 s per 1-minute conversation | Overlapped part: person with a phone **36.7% → 9.6%**; the other person **38.0% → 15.8%** (their room lines are no longer mixed with the phone owner's words) |
| **Round 42:** ⇄ mark on live captions | pyannote overlap model on each finished sentence | About 0.04 s per sentence | The overlapped sentence is marked; the others are not |

The separation test used 12 synthetic Thai conversations: A alone, then B alone, then both talking at once with 3.5–16 s of overlap, levels −3/0/+3 dB.

How the error rate fell on the overlapped parts:

| Stage | Error rate |
|---|---|
| Old method | 47.7% |
| Overlaps left out of voice learning | 40.2% |
| Plus each voice separated and transcribed alone | 20.6% |

For comparison, one person speaking alone gets 0–13%.

Real meetings:
- A 10.6-min meeting had 5 overlaps (9 s). 3 stretches were separated (4 lines became 5), in 30 s.
- A 1-min meeting had one line holding two people. It became one line for each person, in 11 s.
- The 125-min meeting has 16 overlaps (16 s). Separating them would add about 1–2 min.

## 4. Summary and report (LLM)

The same 5.4-minute Thai shareholder meeting was run through each model. Installed now: the two Typhoon models and Gemma 4 E4B (added 2026-10-15); the others were removed after this test:

| Model | Runs on | Time | Quality (read side by side) |
|---|---|---|---|
| Typhoon 2.5 Qwen3-4B | NPU, both cores (prompt read at 400 tokens/s, written at 7–9 tokens/s); CPU 57 / 11.5 tokens/s | 200 s (NPU), 189 s (CPU, 2026-10-15) | Some wrong numbers: in the 2026-10-15 run the paid-up capital became 20,000,000 instead of 2,016,929,700, the month of the meeting was wrong, share counts were written as baht |
| **Gemma 4 E4B (Google, 2026-10-15)** | CPU (prompt read at 50, written at 11 tokens/s, 7.8 GB RAM); the NPU is not used (see below) | **138 s** | **Correct**: paid-up capital 2,016,929,700, 29 in person, 1,310 by proxy, the board decision of 24 Feb; nothing invented. Fewer invented details than the 4B, less rich than the 30B |
| Typhoon 2.5 Qwen3-30B-A3B (**your current setting**) | CPU (32 / 7.3 tokens/s) | 381 s | **Best**: correct numbers, names and room; nothing invented |
| OpenThaiGPT ThaiLLM 8B | CPU | 360 s | Medium |
| OpenThai Chinda 4B | NPU | about the same as Typhoon 4B | Fluent, but invents more |
| Pathumma ThaiLLM 8B | CPU | 359 s | Invented a chairman and percentages: not recommended |
| Pathumma LLM 4B | CPU (fails on the NPU) | 138 s | Empty key points and an invented date: not recommended |

Gemma 4 E4B on the 10.6-minute insurance meeting (9 model calls): 463 s against 822 s for the 30B model on the same transcript; the text was faithful (SLA, 96.5 % / 95 % data delivery, the June target, Mr.Eddy's points) but it missed two items the 30B kept (the 99 % year-end target, the hologram system).

Gemma 4 E4B on the NPU (llama.cpp Hexagon backend, Q4_0, both cores): reads 120 and writes 10 tokens/s, i.e. no faster at writing than the CPU, and it is not reliable: one run answered without the required headings, the next hung after reading 1280 tokens of the prompt. The same day Typhoon 4B stalled on the NPU twice (once after its first call, once mid-sentence) with nothing else holding the NPU. Gemma therefore always runs on the CPU, and a summary call that times out on the NPU sends the retry to the CPU for 30 minutes.

What this means for you: Gemma 4 E4B is the best value now (correct figures, faster than both Typhoon models, no NPU dependence); the 30B is still the richest. With the 30B model, your real meetings took 650–660 s for the summary even when the meeting itself lasted 0.5–1 minute. The summary is by far the slowest step after a meeting.

## 5. Other activities

| Activity | What is used | Measured |
|---|---|---|
| Pre-meeting check | The live model is really loaded and answers 1 s of audio | About 0.3 s; check plus start in 2 s |
| Guidance voice | ★ Piper (th_TH) | About 0.3 s per sentence |
| | Pathumma F5-TTS | 48–57 s per sentence, so it is made in the background and served from a cache in 0.01 s |
| Meeting search | SQLite FTS5 trigram | 400 meetings: index 0.4 s, search about 40 ms |
| Documents for the bot | multilingual-e5-small int8 plus trigram score | 9/9 relevant answers found, 0/6 unrelated answers given |
| Bot answers | Typhoon 4B | 8–14 s per answer |
| Remote participants | WebSocket | +0.15 MB RAM and 0.01–0.03% CPU per guest |

## 6. Where the time goes after a 1-hour meeting (current settings)

Since Round 43 nothing is shown live by default (Record only): the meeting is transcribed carefully afterwards and the summary follows.

| Step | About |
|---|---|
| Transcript (Best, 3-model vote; current setting since Round 43) | 4–5 h (FastConformer alone: 6 min) |
| Who is speaking | 3 min |
| Overlap detection | 20 s |
| Voice separation (typical meeting: under 1 min of overlap) | under 1 min |
| Participants' phones (only when used) | about 10–20 s per phone per 10 minutes |
| Summary + report | the largest part: 30B about 7 min for a 5-min meeting, Gemma 4 E4B about 2.5 min, much longer for an hour |

The biggest gain left: the 4B model on both NPU cores, with a quality cost. Or keep the 30B model only for important meetings.

Since Round 44 this work runs in the processing window (default 18:00–06:00 and all weekend: 12 hours a weeknight). One night fits about
2 one-hour meetings with the Best vote and the 30B summary (each about 5 h); more spill over to the next night or the weekend. The Meetings
list shows each meeting's expected ready time (app/schedule.py: the table above, corrected by this box's own past meetings).

## Licences to note

- Conv-TasNet weights: CC BY-SA 3.0.
- Conv-TasNet was trained with WHAM! noise, which is CC BY-NC 4.0 (non-commercial research). Check this before any commercial use. (SepFormer, also trained with WHAM!, was removed.)
- pyannote segmentation-3.0: MIT.
- Pathumma models: CC-BY-4.0 / Apache-2.0 / MIT, as listed in PROGRESS.md (Round 38).
