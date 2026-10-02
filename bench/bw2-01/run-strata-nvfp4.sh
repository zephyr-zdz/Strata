#!/bin/bash
# The strata-nvfp4 OpenAI/Anthropic-compatible server on bw2-01: GPU 1 only, 127.0.0.1:8097, API key required.
#
#   bench/bw2-01/run-strata-nvfp4.sh start [config.json] [label]   start in the background, wait until loaded
#   bench/bw2-01/run-strata-nvfp4.sh stop                          stop it and check that GPU 1 is free again
#   bench/bw2-01/run-strata-nvfp4.sh status
#
# The key lives outside the repository in ~/.config/strata/strata-nvfp4.key (0600, made on first start) and reaches
# the server as $STRATA_API_KEY.  Clients: Authorization: Bearer $(cat ~/.config/strata/strata-nvfp4.key).
# The config's "gpu": 1 makes the server start the engine with CUDA_VISIBLE_DEVICES=1 (PCI bus order, as nvidia-smi).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CFG=${2:-$ROOT/bench/bw2-01/strata-nvfp4-bw2.json}
LABEL=${3:-default}
PORT=8097
LOGS=/scratch/users/zephyr-zdz/strata-nvfp4
KEYFILE=$HOME/.config/strata/strata-nvfp4.key
mkdir -p "$LOGS" "$(dirname "$KEYFILE")"
if [ ! -s "$KEYFILE" ]; then
  (umask 077; "$ROOT/.venv/bin/python" -c "import secrets; print(secrets.token_hex(24))" > "$KEYFILE")
fi
chmod 600 "$KEYFILE"
KEY=$(cat "$KEYFILE")
health() { curl -sf -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/health"; }

case "${1:-}" in
start)
  if health >/dev/null 2>&1; then echo "already running on :$PORT"; exit 1; fi
  cd "$ROOT"
  t0=$(date +%s.%N)
  STRATA_API_KEY=$KEY STRATA_REQUEST_LINES=1 nohup "$ROOT/.venv/bin/python" -m serve.server --engine strata \
    --config "$CFG" --host 127.0.0.1 --port $PORT > "$LOGS/server-$LABEL.out" 2>&1 < /dev/null &
  echo $! > "$LOGS/server.pid"
  until health 2>/dev/null | grep -q '"loaded": true'; do
    sleep 2
    kill -0 "$(cat "$LOGS/server.pid")" 2>/dev/null || { echo "server died"; tail -30 "$LOGS/server-$LABEL.out"; exit 1; }
  done
  awk -v a="$t0" -v b="$(date +%s.%N)" -v t="$(date +%T)" 'BEGIN {printf "ready in %.1f s (%s)\n", b - a, t}'
  nvidia-smi -i 1 --query-gpu=memory.used,memory.total --format=csv,noheader ;;
stop)
  [ -f "$LOGS/server.pid" ] && kill -TERM "$(cat "$LOGS/server.pid")" 2>/dev/null || true
  for _ in $(seq 90); do pgrep -f "Strata-nvfp4/build/strata --serve" >/dev/null || break; sleep 1; done
  pgrep -af "Strata-nvfp4/build/strata --serve|serve.server --engine strata --config $ROOT" | grep -v pgrep || echo "stopped"
  nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader
  nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader ;;
status)
  health || echo "not running" ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
