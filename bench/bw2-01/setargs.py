#!/usr/bin/env python3
"""Edit a Strata config's engine args: setargs.py strata-q2_0.json --flag value --flag2 value2 --boolflag= -flag-to-remove"""
import json, sys
path, rest = sys.argv[1], sys.argv[2:]
cfg = json.load(open(path))
a = cfg["args"]
i = 0
while i < len(rest):
    f = rest[i]
    if f.startswith("-") and not f.startswith("--"):          # -flag: remove --flag (and its value)
        name = "-" + f
        if name in a:
            j = a.index(name)
            n = 2 if j + 1 < len(a) and not a[j + 1].startswith("--") else 1
            del a[j:j + n]
        i += 1
        continue
    if f.endswith("="):                                       # --boolflag=: present without a value
        if f[:-1] not in a:
            a.append(f[:-1])
        i += 1
        continue
    v = rest[i + 1]
    if f in a:
        a[a.index(f) + 1] = v
    else:
        a += [f, v]
    i += 2
json.dump(cfg, open(path, "w"), indent=1)
print(" ".join(a))
