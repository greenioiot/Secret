#!/usr/bin/env bash
# Install Zecrets as a systemd service (starts at boot, restarts after a crash), then hand over from a copy started by hand.
# Run in your own terminal (asks for the sudo password):  bash /mnt/project2/Zecret/tools/install-service.sh
set -euo pipefail
ROOT=/mnt/project2/Zecret
UNIT=zecrets.service

echo "== installing $UNIT"
sudo cp "$ROOT/tools/$UNIT" /etc/systemd/system/$UNIT
sudo systemctl daemon-reload
sudo systemctl enable $UNIT

if systemctl is-active --quiet $UNIT; then
  echo "== $UNIT is already running under systemd"; exit 0
fi

# a copy started by hand (setsid nohup ./run.sh) holds port 8444: stop it first, but never during a recording
manual=$(pgrep -f "$ROOT/run.sh|\./run\.sh|app/server\.py" || true)
if [ -n "$manual" ]; then
  rec=$(grep -l '"status": *"recording"' "$ROOT"/data/meetings/*/state.json 2>/dev/null || true)
  if [ -n "$rec" ]; then
    echo "!! a meeting is being recorded right now:"; echo "$rec"
    echo "   Installed and enabled; it takes over at the next boot. Run this script again after the meeting to switch now."
    exit 0
  fi
  read -r -p "Stop the copy started by hand (pids: $(echo $manual)) and start the service now? [y/N] " a
  [ "${a,,}" = y ] || { echo "Left as is; the service takes over at the next boot."; exit 0; }
  pkill -f "$ROOT/run.sh|\./run\.sh" || true          # the loop first, so it does not restart the server
  pkill -INT -f "app/server\.py" || true
  for _ in $(seq 1 30); do pgrep -f "app/server\.py|app/worker\.py" >/dev/null || break; sleep 1; done
  pkill -KILL -f "app/server\.py|app/worker\.py" 2>/dev/null || true
fi

echo "== starting $UNIT"
sudo systemctl start $UNIT
for _ in $(seq 1 60); do
  code=$(curl -sk -o /dev/null -w '%{http_code}' https://127.0.0.1:8444/login || true)
  [ "$code" = 200 ] && break; sleep 1
done
systemctl --no-pager --lines=0 status $UNIT | head -5
[ "${code:-}" = 200 ] && echo "== OK: https://<box-ip>:8444 answers" || { echo "!! no answer on 8444 yet: journalctl -u $UNIT -n 50; tail $ROOT/data/server.log"; exit 1; }
