#!/bin/bash
# cpu.sh <cmd...>: run cmd, print the Strata engine's average CPU cores used over it (utime+stime / wall)
P=$(pgrep -f "engine/strata --serve" | head -1)
t() { awk '{print $14+$15}' /proc/$P/stat; }
a=$(t); s=$(date +%s.%N)
"$@"
b=$(t); e=$(date +%s.%N)
echo "engine pid $P: $(echo "($b-$a)/$(getconf CLK_TCK)/($e-$s)" | bc -l | cut -c1-5) cores on average over $(echo "$e-$s" | bc | cut -c1-6) s"
