#!/bin/bash
# One engine run outside the server, on GPU 1, with the config's arguments plus the given ones (A/B runs, cold start).
#
#   bench/bw2-01/oneshot.sh LABEL TOKENS_FILE MAX_NEW [extra engine args ...]
#
# Env: CFG (default bench/bw2-01/strata-nvfp4-bw2.json), DROP="ARG ..." removes those flags (and a value that
# follows) from the config's arguments first.  Writes $LOGS/oneshot-LABEL.{out,err,time,disk}.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CFG=${CFG:-$ROOT/bench/bw2-01/strata-nvfp4-bw2.json}
LOGS=${LOGS:-/scratch/users/zephyr-zdz/strata-nvfp4}
label=$1 tokens=$2 max_new=$3; shift 3
mapfile -t args < <("$ROOT/.venv/bin/python" - "$CFG" "${DROP:-}" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))["args"]
for d in sys.argv[2].split():
    if d in a:
        i = a.index(d)
        del a[i:i + (2 if i + 1 < len(a) and not a[i + 1].startswith("--") else 1)]
print("\n".join(a))
EOF
)
disk() { awk '$3=="sda"||$3=="nvme0n1" {printf "%s %d MB read\n", $3, $6*512/1e6}' /proc/diskstats; }
disk > "$LOGS/oneshot-$label.disk"
cd "$ROOT"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1} LD_LIBRARY_PATH=${CUDA_LIB:-/usr/local/cuda-13.1/lib64} \
  /usr/bin/time -v -o "$LOGS/oneshot-$label.time" \
  "$ROOT/build/strata" "${args[@]}" --tokens-file "$tokens" --max-new "$max_new" --stop-eos "$@" \
  > "$LOGS/oneshot-$label.out" 2> "$LOGS/oneshot-$label.err" || echo "exit $?"
echo "after:" >> "$LOGS/oneshot-$label.disk"; disk >> "$LOGS/oneshot-$label.disk"
grep -E "Elapsed|Maximum resident" "$LOGS/oneshot-$label.time"
cat "$LOGS/oneshot-$label.disk"
grep -E "speculation|tok/s|expert cache auto|elastic|decode|prompt" "$LOGS/oneshot-$label.out" | head -20
