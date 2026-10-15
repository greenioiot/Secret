#!/bin/bash
# stop every process started for the scratch instance (env ZECRET_PORT=8445); nothing else is touched
hit() { for p in /proc/[0-9]*; do pid=${p#/proc/}; [ "$pid" = "$$" ] && continue; [ -r $p/environ ] || continue; tr '\0' '\n' < $p/environ 2>/dev/null | grep -qx 'ZECRET_PORT=8445' && echo $pid; done; }
for pid in $(hit); do echo "kill $pid $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-70)"; kill $pid 2>/dev/null; done
sleep 3; for pid in $(hit); do echo "kill -9 $pid"; kill -9 $pid 2>/dev/null; done; exit 0
