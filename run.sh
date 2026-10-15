#!/usr/bin/env bash
# Start Zecrets (web app + worker). Logs: data/server.log, data/worker.log. Access code: data/access.token
cd "$(dirname "$0")" && mkdir -p data
while true; do
  # a log that grew past 100 MB is set aside (one old copy kept): a runaway error once wrote 5.7 GB in under an hour
  if [ -f data/server.log ] && [ "$(stat -c %s data/server.log)" -gt 104857600 ]; then mv -f data/server.log data/server.log.old; fi
  nice -n 5 .venv/bin/python app/server.py >> data/server.log 2>&1
  rc=$?
  # exit code 75 = the server's own watchdog asked for a restart (stuck event loop, nothing was being recorded); anything else (kill, stop) ends here
  [ "$rc" -eq 75 ] || exit "$rc"
  echo "$(date '+%F %T') server restarted by its watchdog" >> data/server.log
  sleep 2
done
