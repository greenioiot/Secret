# Zecrets (Zecret) — project rules

Thai meeting recorder: FastAPI server `app/server.py` (HTTPS port **8444**), worker `app/worker.py`, `nemo_server`,
single-file UI `app/static/index.html`. Records of work: `PROGRESS.md`, `features.txt`, `PERFORMANCE.md`. Git: github.com/greenioiot/Secret
(private, branch `main`; still the old repo name, rename it on GitHub and `git remote set-url origin git@github.com:greenioiot/Zecret.git`); commit and push after every round. Tests live in `tools/tests/` (they run against a scratch instance on 8445).

## Scope: this project only

This box also runs a second, unrelated project: **Genie Edge** (home assistant) in `/mnt/genie/projectAI`
(also `/home/dev/projectAI`), systemd service `genie-edge.service`, ports 8443, 8581, 8584. It has its own Claude session
and its own CLAUDE.md.

- Never edit, start, stop or kill anything of Genie Edge, and never touch its data (`/mnt/genie/genie-data`).
- Exception, read-only use: Zecrets runs models and Python environments that live in Genie's folder (`app/config.py`
  `PROJECT_AI`, `NEMO_PY`; `app/tts.py`): `models/typhoon-asr`, `models/whisper-th`, `models/typhoon2.5-qwen3-4b`,
  `models/piper`, `vlm/llama.cpp-snapdragon`, `.venv`, `export/venv-nemo`, `app/thai_speech.py`. Use them, never change them.
  Anything new goes under this project (`models/`, `.venv`), not Genie's folder.
- Don't copy code or settings between the two projects. If a task seems to need the other project, stop and ask the user.
- Never kill processes by broad name patterns (`pkill python`, `pkill uvicorn`, `pkill llama`); act only on PIDs found to be
  this project's (port 8444/8445, or `/proc/<pid>/environ`).
- Both projects share the CPU, RAM and Hexagon NPU.

## Standing rules

- Deploy live only when nothing is recording, transcribing, summarizing or analysing (check `data/meetings/*/state.json`
  and `data/.recording`). Never delete the user's real meetings.
- Test on a scratch copy: `ZECRET_DATA=<scratch> ZECRET_PORT=8445 .venv/bin/python app/server.py`; delete it afterwards.
- Runs as a systemd unit (`tools/zecrets.service`, Restart=on-failure, starts at boot). The folder was renamed `/mnt/project2/Zecret`
  (`/home/dev/Zecret`) on 2026-10-15; `/mnt/project2/Secret` and `/home/dev/Secret` are compatibility symlinks. The unit installed on the box
  is still named `secrets.service` and points at the old path; `tools/install-service.sh` with sudo installs it as `zecrets.service` with the
  new path, after which the two symlinks can be removed. Restart:
  `sudo systemctl restart secrets` (the user runs sudo), or kill the server PID and let systemd start it again; SIGKILL a
  leftover worker / nemo_server; verify exactly one server, worker and nemo_server and that /login answers 200.
- `index.html` / `join.html` are served from disk (UI changes are live); `server.py` changes need a restart.
