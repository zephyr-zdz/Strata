#!/usr/bin/env python3
"""Drop these files' pages from the OS page cache (posix_fadvise DONTNEED), for a cold-start measurement without
root (no drop_caches).  Only clean pages no process has mapped go; run it while the engine is stopped.

    python bench/bw2-01/dropcache.py FILE ...      (then fincore.py FILE ... shows what is left)
"""
import os
import sys

for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    os.fdatasync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    print("dropped", path)
