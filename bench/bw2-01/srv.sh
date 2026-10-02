#!/bin/bash
# Start or stop this checkout's Strata server for one config (strata-<name>.json), port 8091, logs to ~/scratch/strata/.
#   srv.sh start q2_0 <label>      srv.sh stop
# Only the server this script started (its PID file) is stopped: another checkout's server is left alone.
cd /home/zephyr-zdz/Projects/Strata
KEY=$(cat ~/.config/strata/bw2-api-key)
PIDF=~/scratch/strata/srv.pid
mkdir -p ~/scratch/strata
case "$1" in
start)
  nohup .venv/bin/python serve/server.py --engine strata --config strata-$2.json --port 8091 \
    > ~/scratch/strata/server-$2-$3.out 2>&1 < /dev/null &
  echo $! > $PIDF
  sleep 2
  until curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:8091/health | grep -q '"loaded": true'; do
    sleep 3
    kill -0 "$(cat $PIDF)" 2>/dev/null || { echo "server died"; tail -20 ~/scratch/strata/server-$2-$3.out; exit 1; }
  done
  echo "ready $(date +%T)"; nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader; free -g | sed -n 2p ;;
stop)
  P=$(cat $PIDF 2>/dev/null)
  if [ -n "$P" ] && [ "$(readlink /proc/$P/cwd)" = /home/zephyr-zdz/Projects/Strata ]; then
    E=$(pgrep -P "$P")
    kill -TERM "$P"
    for i in $(seq 60); do         # until the server and its engine have both exited (the engine frees GPU 1)
      alive=0
      for p in $P $E; do kill -0 "$p" 2>/dev/null && alive=1; done
      [ $alive = 0 ] && break
      sleep 1
    done
  fi
  rm -f $PIDF
  nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader ;;
esac
