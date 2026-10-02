# Strata on bw2-01: one RTX PRO 6000 Blackwell Max-Q (96 GB), batch 1

Measured on 2026-10-03, 01:30-02:10 JST, by an AI agent for the lab. Fork `zephyr-zdz/Strata` at `d9ab843`
(engine 0.1.35). Nothing is committed or pushed. The working tree only adds `BW2-01_RESULTS.md` and
`bench/bw2-01/`. No source file needed changing.

**Verdict:** the CPU half does disappear on a 96 GB card. All 24,576 experts sit in VRAM, the decode hit rate is
100%, and the engine uses under 1 CPU core. But decode does **not** reach 250+ tok/s. The GPU path itself is the
limit: 6.8 ms per plain forward pass, and about 14 ms per MTP verify round that yields about 2.5 tokens.

Against the SGLang NVFP4 reference (TPOT 6.0 ms, 148-167 tok/s, random prompts), single-stream decode is **not
faster**:

- **Random-words 1K prompt** (my closest stand-in for SGLang's random dataset): Strata Q2_0 is 7.4 ms (135 tok/s),
  about 20% slower than the reference.
- **Code prompts:** Strata matches or beats the reference when MTP acceptance is high (0.72, against 0.52 on random
  words). Q2_0 is 5.4 ms at 1K, 5.65 ms at 4K and 6.2 ms at 32K (162-186 tok/s). The reference has no code-prompt
  number to compare against.
- **IQ3_S**, the size whose quality matches the full model: no valid speed number. Every IQ3_S run was taken under
  about 70% disk-IO pressure from other jobs, so those numbers are discarded (see the IQ3_S section).

Q2_0 is a 2-bit quantization, against 4-bit NVFP4, so the speed is not a quality match either. SGLang also serves 16
concurrent requests, and Strata serves one at a time.

## Hardware and conditions

- **GPU:** GPU 1 = RTX PRO 6000 Blackwell **Max-Q** Workstation Edition, 96 GB, 300 W cap, sm_120, PCIe 5.0 x16,
  driver 595.91.07. GPU 0 was busy with the user's own training job (ts-ml) the whole time (96%, 300 W) and was not
  touched.
- **CPU and RAM:** Ryzen 9 9950X (AVX-512), 245 GiB RAM shared, with 183-218 GiB available during the runs.
- **Storage:** every weight file is on the HDD. `df` gives `/dev/sda1` for `models/Q2_0`, `models/IQ3_S`,
  `packs/q2_0`, `packs/iq3_s` and `mtp`, all under `/data/users/zephyr-zdz/models/strata`. Nothing is in `~`,
  `~/.cache`, the repo or `/tmp`. The one exception the user allows is a copy of the n-gram shard 2. The lead made it,
  not me, at `~/models-ssd/strata/Q2_0/` (NVMe, 28,800,138,432 bytes, same size as the original). No number here
  used it: the session meant to measure it was cancelled (see "Not measured"). The HDD placement matters, see the PLE row under the build problems.
- **Background load:**
  - During the Q2_0 runs: load average about 2.7. From about 01:41, the lead's RadixArk `hf download` wrote about
    100 MB/s to the same HDD. The PLE shard was in the page cache in full (`fincore.py`: 26.82 / 26.82 GiB), so the
    mmap runs should not have needed the disk, but I did not check that with `/proc/diskstats`.
  - During the IQ3_S runs: another job wrote heavily to disk. `/proc/pressure/io` showed some=70%, load average
    about 9, CPU pressure about 0.
- **Toolchain:** CUDA 13.1 (`/usr/local/cuda-13.1`), GCC 15.2, CMake/Ninja from the venv, Python 3.12 (uv).

## Build and setup: problems and fixes

There were no CUDA 13 / sm_120 source problems. The engine compiled cleanly on the first try with CUDA 13.1 and
GCC 15.2 for `CMAKE_CUDA_ARCHITECTURES=120`, with warnings only. Every problem I hit was in the environment:

| Problem | What I did (no code change) |
| --- | --- |
| System Python 3.14 has no `ensurepip`, so `setup.sh` would run `sudo apt install python3-venv` | Made the venv myself first: `uv venv --seed --python 3.12 .venv`. `setup.sh` then skips the Python step. |
| No ready-made Linux engine: `releases/download/v0.1.35/strata-linux-x64.zip` and `latest` both return 404 | Setup compiled it by itself, which took about 8 minutes on 16 jobs. `engine/BUILD.json` records `source: local`, archs `[120]`. |
| Setup's single-stream downloader got 10-50 MB/s while the 1 GbE link was shared | Downloaded shard 2 (Q2_0) and IQ3_S shard 1 with `hf download` (xet) at the pinned revision `ed59f92`, then moved them into `models/<size>/`. Setup accepted them by size (`already downloaded`), and the IQ3_S shard 2 was hard-linked from Q2_0's. |
| **The PLE / n-gram shard (26.8 GiB) on the HDD.** Setup's default `--ple-io direct` reads 16 rows per token with O_DIRECT, which bypasses the page cache. | Added `--ple-io mmap` to the engine args and warmed the shard into the page cache once (`cat shard2 > /dev/null`, 2 min 38 s). This took 1K decode from **23.4 to 184 tok/s** and 1K prefill from 169 to 2,262 tok/s. `--ple-io ram` (mlock) cannot work here: the hard `ulimit -l` is 8 MB. |
| `run-q2_0.sh` passes `--open` (it would try to open a browser on the desktop session) | Started `serve/server.py` directly, without `--open`. |

Files go where the server rules say:

- **`/data/users/zephyr-zdz/models/strata`:** models 113 GB (Q2_0 66 GB + IQ3_S shard 1 51 GB, shard 2 shared),
  packs 39 GB (the Q2_0 AVX-512 copy, 31.6 GiB `experts.bin`; the IQ3_S pack is small), mtp 6.5 GB.
- **`~/Projects/Strata`:** code, the venv, the 45 MB engine and the 151 MB build tree.
- **`~/.config/strata/settings.json`:** `data_dir` points at the folder above.

## Commands

```bash
cd ~/Projects/Strata
UV_CACHE_DIR=~/scratch/uv-cache uv venv --seed --python 3.12 .venv
export PIP_CACHE_DIR=~/scratch/pip-cache
KEY=$(cat ~/.config/strata/bw2-api-key)            # 48 hex chars, private file (0600)
./setup.sh --yes --family qwen --model Q2_0 --context 65536 --vision no --gpu 1 --port 8091 \
  --host 127.0.0.1 --api-key "$KEY" --data-dir /data/users/zephyr-zdz/models/strata --no-start
./setup.sh --setup --yes --family qwen --model IQ3_S --context 65536 --vision no --gpu 1 --port 8091 \
  --host 127.0.0.1 --api-key "$KEY" --kv-streaming off --data-dir /data/users/zephyr-zdz/models/strata --no-start
# engine-arg edits in strata-<model>.json (bench/bw2-01/setargs.py); setup's original: strata-q2_0.setup-default.json
.venv/bin/python bench/bw2-01/setargs.py strata-q2_0.json --ple-io mmap -kv-resident --resident-experts=
.venv/bin/python bench/bw2-01/setargs.py strata-iq3_s.json --ple-io mmap
cat /data/users/zephyr-zdz/models/strata/models/Q2_0/*00002-of-00002.gguf > /dev/null   # warm the PLE table
bench/bw2-01/srv.sh start q2_0 <label>      # server.py --config strata-q2_0.json --port 8091 (CUDA_VISIBLE_DEVICES=1 via config "gpu": 1)
.venv/bin/python bench/bw2-01/benchmark.py --api-key "$KEY" --out bench/bw2-01/results/<run> \
  --targets 1024,4096,32768 --runs 3 [--style random] [--reasoning high] [--pack .../packs/iq3_s]
.venv/bin/python tools/calibrate.py strata-q2_0.json          # with the server stopped
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 \
  engine/strata <config args> --tokens 9707,11,1052,525,498,30 --gpu-only-full
bench/bw2-01/srv.sh stop
```

`bench/bw2-01/benchmark.py` is the community RTX 5090 script
(`bench/results/2026-09-30-community-rtx-5090/benchmark.py`) with five additions:

- an API key
- the targets 1K, 4K and 32K
- a reasoning level
- a "random words" prompt style
- VRAM and MemAvailable recorded per row

Every request is fresh: a nonce comes before the filler, so no prompt-cache reuse (`cached_tokens` is 0 in every
row). The prompt is cut to exactly 1,024, 4,096 or 32,768 tokens with the pack's tokenizer. Generation is greedy with
`max_tokens` 256, and every run reached 256 (`finish: length`). The numbers come from the engine's own `/metrics`
row (`prompt_ms`, `decode_ms`, drafts). TPOT = `decode_ms / 256`. Raw rows are in `bench/bw2-01/results/*/`, and the
logs (setup, server, engine, calibrate, GPU floor) are in `bench/bw2-01/logs/`.

## Results

Unless a row says otherwise: context 65536, int8 KV, MTP `--spec 4 --spec-min-p 0.5`, `--ple-io mmap` with a warm
page cache, thinking off, code-explanation prompt, median of 3 runs (min-max in brackets). "tok/round" = generated /
(generated - accepted drafts).

### Q2_0

| Config | Prompt | Prefill tok/s | Decode tok/s | TPOT ms | tok/round | Client TTFT s | Client e2e s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Setup default (`--ple-io direct` on HDD, KV streaming on) | 1,024 | 169 | 23.4 (1 run) | 42.8 | 2.53 | 6.08 | 17.0 |
| mmap, KV streaming on | 1,024 | 2,262 | 184.3 [170-186] | 5.42 | 2.64 | 0.47 | 1.85 |
| mmap, KV streaming on | 4,096 | 4,014 | 174.9 [174-176] | 5.72 | 2.53 | 1.04 | 2.49 |
| mmap, KV streaming on | 32,768 | 5,315 | 159.2 [154-166] | 6.28 | 2.42 | 6.20 | 7.80 |
| mmap, KV streaming **off** | 1,024 | 2,395 | 185.3 [176-193] | 5.40 | 2.61 | 0.44 | 1.81 |
| mmap, KV streaming off | 4,096 | 4,422 | 177.0 [176-177] | 5.65 | 2.53 | 0.94 | 2.38 |
| mmap, KV streaming off | 32,768 | 5,899 | 162.1 [157-167] | 6.17 | 2.42 | 5.59 | 7.16 |
| **+ `--resident-experts` (low-RAM, recommended here)** | 1,024 | 2,518 | 185.9 [177-193] | 5.38 | 2.61 | 0.42 | 1.79 |
| + `--resident-experts` | 4,096 | 4,479 | 177.0 [176-177] | 5.65 | 2.53 | 0.93 | 2.37 |
| + `--resident-experts` | 32,768 | 5,928 | 162.1 [157-167] | 6.17 | 2.42 | 5.57 | 7.14 |
| `--spec 6` (KV streaming off) | 1,024 | 2,440 | 182.9 [173-190] | 5.47 | 2.94 | 0.43 | 1.82 |
| `--spec 6` | 4,096 | 4,408 | 173.0 [170-176] | 5.78 | 2.84 | 0.94 | 2.42 |
| **Random-words prompt** (KV streaming on) | 1,024 | 2,400 | 135.5 [133-137] | 7.38 | 1.75 | 0.44 | 2.36 |
| Thinking **on** (`reasoning_effort: high`, KV streaming on) | 1,024 | 2,267 | 178.9 [175-185] | 5.59 | 2.44 | 0.47 | 1.89 |
| Thinking on | 4,096 | 3,970 | 192.1 [191-196] | 5.21 | 2.78 | 1.05 | 2.38 |

In the thinking-on rows, all 256 measured tokens are `reasoning_content` (thinking text), not the answer. The text
is more predictable (acceptance 0.82-0.83), so the higher tok/s there is not a general speedup.

Same request through the OpenAI API, non-streaming (`curl`-style POST of the 1K code prompt): **1.83 s
end-to-end** for 1,025 prompt + 256 output tokens (prefill 421 ms, decode 1,407 ms = 182 tok/s). The same request
again took 1.43 s, because the conversation cache reused 1,018 tokens. The streaming client numbers in the table go
through the same `/v1/chat/completions`.

- **GPU-only floor** (`--gpu-only-full`, one token, no speculation): **6.83 ms per token = 146 tok/s** (6.53 ms for
  48 layers, 0.136 ms per layer, 0.30 ms LM head). MTP turns that into about 14 ms per verify round of about 3 rows
  for 2.5-2.6 tokens, so 180-190 tok/s on code.
- **Rough roofline:** a token reads about 0.66 GB of experts (480 × 1.38 MB) plus about 1.5-2.5 GB of dense weights
  and the head, so 2-3 GB, which is about 1.5 ms at 1.8 TB/s. The engine runs at about 20-25% of memory bandwidth at
  batch 1: it is latency- and launch-bound, not bandwidth-bound.
- **`--calibrate`** (`tools/calibrate.py`, 113 s): **+0%**, nothing kept. The PCIe share 0-0.75 measured 169.5-169.9
  tok/s, the same because nothing misses. The draft floor 0.3 / 0.5 / 0.7 measured 164 / 170 / 161 tok/s, so the
  default 0.5 is best. The pool workers 15 / 10 / 8 measured 170.0 / 170.2 / 170.2 tok/s.
- **`--spec 6`:** more drafts accepted per round (2.9 tok/round) but slower rounds, so no gain.

### IQ3_S (3.5-bit, `--ple-io mmap`, KV streaming off, default RAM mode): DISCARDED

These numbers were measured while other jobs wrote to the same HDD (`/proc/pressure/io` some = 61-74%). At the
lead's request they are **not valid results**. They are listed only to show that the runs happened. The planned
clean rerun from the NVMe copy was cancelled (see "Not measured").

| Run (under IO contention) | Prompt | Prefill tok/s | Decode tok/s | TPOT ms | tok/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| First runs after loading, under IO contention | 1,024 | 2,331 | 95.3 [28-97] | 10.5 | 2.37 |
| First runs after loading, under IO contention | 4,096 | 4,223 | 113.0 [63-113] | 8.9 | 2.56 |
| First runs after loading, under IO contention | 32,768 | 5,694 | 65.8 [54-103] | 15.2 | 2.39 |
| Rerun, warm (median of 5) | 1,024 | 2,353 | 156.6 [150-162] | 6.39 | 2.37 |
| Rerun, warm (median of 5) | 4,096 | 4,205 | 162.4 [156-170] | 6.16 | 2.56 |
| Rerun, warm (median of 3) | 32,768 | 5,633 | 154.1 [152-167] | 6.49 | 2.39 |
| Random-words prompt | 1,024 | 2,498 | 130.8 [116-135] | 7.64 | 1.84 |

The first IQ3_S runs had a 100% hit rate and the same number of expert lookups, but took 2-5x longer to decode. In
my rerun the engine had only 1-3 major page faults per request, so I attribute the slow first runs to cold mapped
pages of the IQ3_S GGUF (the native pack reads some tensors from it in place) under heavy IO contention from another
job. That is a likely cause, not a proven one. The "warm" reruns were steadier, but they ran under the same
contention, so they are discarded too. `--gpu-only-full` does not run on the native IQ packs ("not captured with
post graphs"), so there is no GPU floor for IQ3_S.

### Footprint

| | Q2_0 default RAM mode | Q2_0 `--resident-experts` | IQ3_S default RAM mode |
| --- | ---: | ---: | ---: |
| Experts in VRAM | 24,576 / 24,576 (31.64 GiB) | 24,576 (31.64 GiB) | 24,576 (46.84 GiB) |
| Decode hit rate | 100.0% | 100.0% | 100.0% |
| GPU 1 VRAM used (nvidia-smi) | 39.1-39.5 GB | 39.5 GB | 55.5 GB |
| VRAM still free (engine log) | 57.8-58.3 GB | 57.9 GB | 41.8 GB |
| RAM pinned for experts | 31.64 GiB | **3.58 GiB** | 46.84 GiB |
| `free -g` "used", before → after load | 25 → 60 GB | 26 → 31 GB | 27 → 80 GB |
| MemAvailable during runs | 183-185 GiB | 212-213 GiB | 164-168 GiB |
| Engine CPU, average over a benchmark | 0.84-0.90 cores (of 32) | not sampled | not sampled |
| Load time (model files in page cache / not) | about 45-55 s | about 50 s | 5 min 49 s (47 GB read from the HDD at 0.15 GiB/s) |

In the resident mode the engine keeps in RAM only the experts the prompt path borrows from the VRAM cache (2,783
slots). Speed was the same as the default, prefill was 1-5% faster, and there were 0 reads from the file
(`resident RAM: ... 0 blob reads from the file`). On this shared box it saves about 28 GB of pinned RAM, so it is
the mode to use with Q2_0. I did not try it with IQ3_S, whose native pack has no `experts.bin`.

## Quality sanity

Run through `/v1/chat/completions`, greedy, script `bench/bw2-01/quality.py`. Outputs are in
`bench/bw2-01/results/quality-*.json`.

| Model | Thinking | GSM8K (8 test questions) | Chinese |
| --- | --- | --- | --- |
| Q2_0 | off | 8/8 | MoE explanation: correct, fluent, kept the 150-character limit, with a hospital/triage analogy. Apple word problem: correct (6), clean steps. |
| Q2_0 | high | 8/8 (the thinking is streamed as `reasoning_content`) | Both correct. |
| IQ3_S | off | 8/8 | Both correct. |

Nothing broken was seen: no looping and no wrong language. Chinese decode on Q2_0 ran at about 130 tok/s end-to-end
for short answers (CJK draft vocabulary, the default `cjk`). Eight easy questions are a smoke test, not a benchmark.

## Against the SGLang reference

The reference: TPOT about 6.0 ms at 1 request (148-167 tok/s), NVFP4, MTP, 1024-in/256-out random prompts, same card
type.

| | TPOT ms at 1K | TPOT ms at 32K | Notes |
| --- | ---: | ---: | --- |
| SGLang NVFP4 + MTP (reference, random prompts) | about 6.0 | - | 16 concurrent requests possible |
| Strata Q2_0, code prompt | 5.4 | 6.2 | 2-bit |
| Strata Q2_0, random-words prompt | 7.4 | - | lower MTP acceptance (0.52 vs 0.72) |
| Strata IQ3_S | - | - | discarded (IO contention) |

The prompts are not identical. SGLang's random dataset uses random token ids, and mine uses random English words.
MTP acceptance depends strongly on the text, so the fair comparison is SGLang on the same requests. The exact
request bodies are saved as `bench/bw2-01/results/<run>/tokens-<n>-run-<k>-request.json`. Strata's prefill is
2.4K / 4.5K / 5.9K tok/s at 1K / 4K / 32K, with a TTFT of 0.42 s at 1K and 5.6 s at 32K. The task gave no SGLang
prefill reference to compare with.

## What I could not measure, or did not

- SGLang itself on the same prompts: GPU 0 is the lead's.
- `--ple-io ram`: the hard memlock limit is 8 MB, and mlock of 27 GB needs an admin.
- Keeping the PLE shard on the NVMe: forbidden by the server rules unless the user agrees. On the HDD it only works
  while the page cache keeps the shard (26.8 GiB of reclaimable cache). If other jobs evict it, decode falls toward
  the 23 tok/s of the direct mode.
- **Cancelled by the lead (not pending):** the NVMe-copy session. It would have measured Q2_0 4K, IQ3_S 4K/32K and
  random-words 1K, all with `--ple-io direct`, plus the disk-read check during decode. Loading the models would have
  re-read about 70 GB from `/dev/sda` while another job (the nvfp4 port's conversion) needed that disk. The nvfp4 port
  measures `--ple-io direct` on the NVMe copy with its own engine, which answers the same question. The HDD
  cold-vs-warm test was dropped too. What stands: HDD + direct (it reads the disk on every token, whatever the cache
  holds) = 23.4 tok/s; HDD + mmap with the shard in the page cache = 184 tok/s.

  The session is still in the repo, never run: `bench/bw2-01/run-ssd.sh`, with the configs `strata-q2_0-ssd.json`
  and `strata-iq3_s-ssd.json` (gitignored). One finding from preparing it, read from the code but never run: moving
  only `--ple-gguf` to another path makes the engine refuse to start. With `--native`, the engine adds the PLE file
  as a third dense shard, and the copy has the same split number as the HDD shard 2 ("native dense: duplicate split
  shard number", `src/core/native_dense.cpp:103`). The configs avoid that by listing the HDD shards with
  `--native-dense-gguf <shard1> --native-dense-gguf <shard2>`, so only the n-gram reads go to the NVMe.
- The GPU floor for IQ3_S: not supported for native packs.
- IQ3_S with `--resident-experts`, KV streaming on vs off for IQ3_S, 64K+ prompts, thinking on at 32K, and images.
- Engine CPU share for the IQ3_S runs: not sampled. RAM and VRAM are from `free` and `nvidia-smi` snapshots, not
  continuous traces.

## Left in place

- **Server:** stopped at 02:07. GPU 1 went back to 25 MiB, and the lead's SGLang server has used it since.
- **Model files, packs and the MTP layer:** kept in `/data/users/zephyr-zdz/models/strata`.
- **`strata-q2_0.json`:** holds the tuned args (mmap PLE, no KV streaming, `--resident-experts`). Setup's original
  config is in `strata-q2_0.setup-default.json`. Both are gitignored and hold the API key, so keep them private.
- **Restart:** warm the PLE shard if the page cache dropped it (`python3 bench/bw2-01/fincore.py <shard2>` shows
  how much is cached), then run `bench/bw2-01/srv.sh start q2_0 run`.
- **Deleted temporary files:** `~/scratch/pip-cache`, `~/scratch/hf-home` and `~/scratch/strata`. The logs worth
  keeping are copied to `bench/bw2-01/logs/`. `~/scratch/uv-cache` predates this run and is shared with other work,
  so I left it.
