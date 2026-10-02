#!/bin/bash
# The short NVMe-copy session (PLE shard 2 on ~/models-ssd, everything else on the HDD): Q2_0 4K, IQ3_S 4K/32K and
# random-words 1K, --ple-io direct.  Records disk reads (sda = HDD, nvme0n1 = SSD) and IO pressure per block.
cd /home/zephyr-zdz/Projects/Strata
KEY=$(cat ~/.config/strata/bw2-api-key)
R=bench/bw2-01/results
disk() { echo "$1 $(date +%T) $(awk '$3=="sda"||$3=="nvme0n1"{printf "%s rd_sectors=%s ", $3, $6}' /proc/diskstats) io_pressure: $(head -1 /proc/pressure/io)"; }
bench() { .venv/bin/python bench/bw2-01/benchmark.py --api-key "$KEY" "$@" 2>&1 | .venv/bin/python bench/bw2-01/show.py | grep -v summary; }
start() { timeout 600 bench/bw2-01/srv.sh start "$1" ssd || { echo "START FAILED $1"; tail -5 strata-$1.log 2>/dev/null; tail -15 ~/scratch/strata/server-$1-ssd.out; bench/bw2-01/srv.sh stop; exit 1; }; }
nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader
start q2_0-ssd
grep -E "PLE on|expert cache auto|VRAM free|resident RAM mode" strata-q2_0-ssd.log | tail -4
disk before-q2_0
bench --out $R/q2_0-ssd-direct-code-none --targets 4096 --runs 3
disk after-q2_0
bench/bw2-01/srv.sh stop
start iq3_s-ssd
grep -E "PLE on|expert cache auto|VRAM free" strata-iq3_s-ssd.log | tail -3
disk before-iq3_s
bench --pack /data/users/zephyr-zdz/models/strata/packs/iq3_s --out $R/iq3_s-ssd-direct-code-none --targets 4096,32768 --runs 3
disk mid-iq3_s
bench --pack /data/users/zephyr-zdz/models/strata/packs/iq3_s --style random --out $R/iq3_s-ssd-direct-random-none --targets 1024 --runs 3
disk after-iq3_s
bench/bw2-01/srv.sh stop
nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader
