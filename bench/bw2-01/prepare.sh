#!/bin/bash
# Prepare RadixArk/Qwen3.8-Flash-Next-NVFP4 for the strata-nvfp4 engine on bw2-01 (README "Prepare the model", on Linux).
#
#   bench/bw2-01/prepare.sh [step ...]      steps: embd convert verify pack mtp vocab (default: all, in that order)
#
# Inputs:  the HF snapshot in the shared cache (read only).  Outputs: $OUT on the HDD (/data).  Logs, the converter's
# temporary file and the RSS trace: $LOGS on scratch.  The n-gram table is not built: the engine reads ISTA-DASLab's
# IQ4_NL table (README: KL 0.0026 against the FP8 one), the SSD copy under ~/models-ssd.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${MODEL:-/data/shared/models/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594}
OUT=${OUT:-/data/users/zephyr-zdz/models/strata-nvfp4}
LOGS=${LOGS:-/scratch/users/zephyr-zdz/strata-nvfp4}
PLE=${PLE:-/data/users/zephyr-zdz/models/strata/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf}
PY="nice -n 10 $ROOT/.venv/bin/python"
export PYTHONPATH=$ROOT/third_party/llama.cpp/gguf-py OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR=$LOGS/tmp          # nvfp4_convert --use-temp-file spools the ~74 GB GGUF body here (scratch HDD)
mkdir -p "$OUT/models" "$OUT/packs" "$OUT/mtp" "$LOGS" "$TMPDIR"
cd "$ROOT"

# the step's peak RSS from /usr/bin/time, and a trace of every tools/ process's RSS every 15 s
rss_trace() { while sleep 15; do ps -eo rss=,args= | awk -v t="$(date +%T)" '/tools\/[a-z_0-9]+\.py/ && !/awk/ {printf "%s %.2f GiB %s\n", t, $1/1048576, $3" "$4}'; done >> "$LOGS/rss-trace.log"; }
rss_trace & TRACE=$!
trap 'kill $TRACE 2>/dev/null' EXIT
run() { local name=$1; shift; echo "=== $name: $* ($(date +%T))"; /usr/bin/time -v -o "$LOGS/time-$name.txt" "$@" 2>&1 | tee "$LOGS/prepare-$name.log"; grep -E "Maximum resident|Elapsed" "$LOGS/time-$name.txt"; }

steps=${*:-embd convert verify pack mtp vocab}
for s in $steps; do case $s in
embd)    # 1.3 GB: the token embedding in BF16 as shipped
  run embd $PY tools/embd_bf16_pack.py --model "$MODEL" --out "$OUT/models/token-embd-bf16.gguf" ;;
convert) # ~74 GB: NVFP4 experts (lossless repack) + Q8_0/BF16 dense; the PLE table and the MTP head are left out
  run convert $PY tools/nvfp4_convert.py --model "$MODEL" --outfile "$OUT/models/radixark-nvfp4.gguf" --use-temp-file ;;
verify)  # the repack against ModelOpt's own definition (every scale byte, a few experts bit for bit), then the GGUF,
         # and the IQ4_NL table's rows against this checkpoint's FP8 rows (the same table, or the engine reads garbage)
  run verify $PY tools/nvfp4_verify.py --model "$MODEL" --experts 4 --scan-all
  run verify-gguf $PY tools/nvfp4_verify_gguf.py --model "$MODEL" --gguf "$OUT/models/radixark-nvfp4.gguf" --ple-gguf "$PLE" ;;
pack)    # experts.bin (63 GiB, the NVFP4 scale tails only live here), index, dense, tokenizer
  run pack $PY tools/iq_pack.py --gguf "$OUT/models/radixark-nvfp4.gguf" --out "$OUT/packs/radixark-nvfp4" ;;
mtp)     # the checkpoint's own MTP head (BF16, unchanged from Qwen's): extract, Q2_0 experts, runtime files
  run mtp-extract $PY tools/mtp_extract.py --model "$MODEL" --out "$OUT/mtp/extract"
  run mtp-pack $PY tools/mtp_pack.py --src "$OUT/mtp/extract" --experts q2_0 --out "$OUT/mtp/mtp-q2_0.gguf"
  run mtp-rt $PY tools/mtp_rt.py --gguf "$OUT/mtp/mtp-q2_0.gguf" --out "$OUT/mtp/rt" ;;
vocab)   # draft subsets: the fork's default (en/code + Cyrillic) and en/code + CJK; rt/draft_vocab.bin is the one used
  G="$OUT/models/radixark-nvfp4.gguf"
  run vocab-cjk $PY tools/draft_vocab.py --gguf "$G" --base data/draft_vocab_en.bin --add cjk --out "$OUT/mtp/draft_vocab_en_cjk.bin" --stats
  cp data/draft_vocab.bin "$OUT/mtp/draft_vocab_fork_default.bin"
  cp "$OUT/mtp/draft_vocab_en_cjk.bin" "$OUT/mtp/rt/draft_vocab.bin" ;;
*) echo "unknown step $s"; exit 2 ;;
esac; done
df -h "$OUT"; du -sh "$OUT"/*
