#!/usr/bin/env python3
"""How much of each file is in the OS page cache (mincore), like util-linux's fincore."""
import ctypes
import mmap
import os
import sys

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PAGE = os.sysconf("SC_PAGE_SIZE")
CHUNK = 1 << 30

for path in sys.argv[1:]:
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    cached = 0
    for off in range(0, size, CHUNK):
        n = min(CHUNK, size - off)
        addr = libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, off)
        pages = (n + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(pages)
        libc.mincore(ctypes.c_void_p(addr), n, vec)
        cached += sum(b & 1 for b in vec.raw)
        libc.munmap(ctypes.c_void_p(addr), n)
    os.close(fd)
    print(f"{cached * PAGE / 2**30:8.2f} / {size / 2**30:8.2f} GiB cached  {path}")
