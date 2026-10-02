# strata-nvfp4 on bw2-01: RadixArk Qwen3.8-Flash-Next NVFP4 on one RTX PRO 6000 (96 GB)

A record run, measured on 2026-10-03 between 04:18 and 04:28 JST. Branch `bw2-01-nvfp4` of `zephyr-zdz/Strata`,
on top of `sergqwer/strata-nvfp4` v0.1.35-nvfp4.1 (`0dc4093`). Not pushed. The scope was cut to a single record
run: no memory-mode A/B, and no GPU-only floor or per-round profile.

**Verdict.** The 125B NVFP4 model fits on one 96 GB card with every expert in VRAM. It decodes at **153-171
tok/s** (TPOT 5.8-6.6 ms) after 1K and 32K code prompts, 181 tok/s on a random-words 1K prompt and 161 tok/s in
Chinese. A 32K prompt is read at about 5,830 tok/s (TTFT 5.7 s). It answered 8/8 GSM8K questions, both Chinese
prompts and the coding prompt correctly. During requests it reads nothing from the HDD and uses about 1 CPU core.

- **Against upstream Strata Q2_0 on the same card** (strata-bench, `~/Projects/Strata/BW2-01_RESULTS.md`): the
  speed is about the same (Q2_0 was 186 / 162 tok/s at 1K / 32K and 135 on random words), with 4.5-bit experts
  instead of 2-bit ones.
- **Against the SGLang NVFP4 + MTP cookbook number** (TPOT about 6.0 ms, 148-167 tok/s at batch 1): about the same
  for one stream. SGLang also serves 16 streams at once, and Strata serves one.
- **Start-up costs:** a cold start reads 75.6 GB from the HDD (6 min), and 13.85 GiB of host RAM stay page-locked.

## Machine and conditions

| | |
| --- | --- |
| GPU | GPU 1 only (`CUDA_VISIBLE_DEVICES=1`, PCI bus order): RTX PRO 6000 Blackwell **Max-Q** Workstation, 96 GB (97,887 MiB), 300 W cap, sm_120, PCIe 5 x16, driver 595.91.07. GPU 0 ran the user's training job throughout and was not touched. |
| CPU / RAM | Ryzen 9 9950X (16C/32T, AVX-512), 245 GiB shared. MemAvailable was 224 GiB before the start. |
| Storage | All model files are on the HDD `/dev/sda1` (`/data/users/zephyr-zdz/models/strata-nvfp4`). The one exception is the n-gram (PLE) table, read from the NVMe copy `~/models-ssd/strata/Q2_0/…-00002-of-00002.gguf` on `/dev/nvme0n1p2`. |
| Background | Nobody else used `/dev/sda` during the window (main held the other agents off it). `/proc/pressure/io` some avg10 was 0.61 at the first request (avg300 21%, from this run's own cold load) and 0.00 after the last one. |
| OS / tools | Linux 7.0.0-34, CUDA 13.1 (nvcc V13.1.115, `/usr/local/cuda-13.1`), GCC 15.2.0, CMake 4.2.3, Ninja 1.13.2, Python 3.12.14 (uv venv), torch 2.14.1+cpu, numpy 2.5.3, safetensors 0.8.0, transformers 5.18.0, gguf-py 0.19.0 (from the pinned llama.cpp) |

## What exactly was used (reproducibility)

| | |
| --- | --- |
| Engine source | this branch. Base `0dc409390a615445c097e7ece494e11ee93176f9` (= tag v0.1.35-nvfp4.1 = upstream 0.1.35 `d9ab843` plus the fork's commits), plus the commits listed under "Changes on this branch" |
| llama.cpp (ggml, gguf-py, converter) | `3cf03257f219afbe7334045ff7c6a06ac68c627d`, the pinned commit (`third_party/ggml/VERSION.txt`), copied into `third_party/llama.cpp` |
| Checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4`, snapshot `7b719225242aacd3dbd3f9407468c2ee9a9d2594` (135.25 GB, 419 files, in the shared HF cache, read only). Its `model.safetensors.index.json` has the same 296,475 tensor names and the same file for each as the OrcaRouter checkpoint the fork targets, and the same `total_size`. |
| n-gram table | ISTA-DASLab IQ4_NL shard (`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`, 28,800,138,432 B). The NVMe copy matched the HDD original in 64 sampled 1 MiB blocks. |
| Expert profile | `data/expert-profile.bin` as shipped (24,576 ranked pairs) |
| sha256 of the produced files | `bench/bw2-01/sha256.txt`, computed after the run at idle priority |

## Changes on this branch

- **The build needed no source change.** It built on the first try for `CMAKE_CUDA_ARCHITECTURES=120` with CUDA
  13.1 and GCC 15.2. There were 30 warnings, all `%llu`/`%lld` used with `uint64_t`/`int64_t`; on LP64 Linux these
  are the same width, so they are harmless.
- **`b9cad8a` `nvfp4_convert: --use-temp-file`.** llama.cpp's `_generate_nvfp4_tensors` repacks every NVFP4
  expert up front, and `GGUFWriter.add_tensor` keeps each array until the file is written: about 63 GiB of RSS. The
  flag passes llama.cpp's own temp-file mode through, so each tensor is written to `$TMPDIR` (scratch HDD) as it is
  made. Peak RSS was 5.2 GiB, and the converter's other bookkeeping is unchanged.
- **`36b663e` and the final commit: `bench/bw2-01/`.** The prepare, run and benchmark scripts, and this file.

## Build

```bash
git -C ~/Projects/Strata remote add nvfp4 https://github.com/sergqwer/strata-nvfp4 && git -C ~/Projects/Strata fetch nvfp4
git -C ~/Projects/Strata worktree add ~/Projects/Strata-nvfp4 -b bw2-01-nvfp4 nvfp4/main
cd ~/Projects/Strata-nvfp4
cp -r <a checkout of llama.cpp at 3cf0325> third_party/llama.cpp
UV_CACHE_DIR=~/scratch/uv-cache uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt safetensors sentencepiece transformers
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu   # the converters need no GPU
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build --target strata -j 12          # about 2 min; build/strata 52.6 MB
```

## Prepare (`bench/bw2-01/prepare.sh`)

`MODEL`, `OUT`, `LOGS` and `PLE` are environment variables with bw2-01 defaults. All steps ran in a
`systemd-run --user --scope -p MemoryMax=64G` at nice 15, with `OMP_NUM_THREADS=8`. The wall times include main's
SIGSTOP of the converter from 02:23 to 02:35, which kept the HDD free for SGLang's load.

| step | command (tools/…) | wall | peak RSS | output |
| --- | --- | ---: | ---: | --- |
| embedding | `embd_bf16_pack.py --model $MODEL --out $OUT/models/token-embd-bf16.gguf` | 9 s | 0.2 GiB | 1.27 GB, BF16 as shipped |
| GGUF | `nvfp4_convert.py --model $MODEL --outfile $OUT/models/radixark-nvfp4.gguf --use-temp-file` | 34.5 min (12 min of it paused) | 5.2 GiB | 73.94 GB: NVFP4 experts x144, Q8_0 x302, BF16 x484, F32 x580, F16 x1; 128 PLE shards left out |
| pack | `iq_pack.py --gguf …/radixark-nvfp4.gguf --out $OUT/packs/radixark-nvfp4` | 11.9 min | 45 GiB, of which 2.9 GiB anonymous (the rest is the mapped GGUF, page cache) | `experts.bin` 63.28 GiB (48 x 512 blobs of 2,764,816 B with the scale tail), `dense.bin` 1.54 GB, index, tokenizer |
| MTP | `mtp_extract.py` → `mtp_pack.py --experts q2_0` → `mtp_rt.py` | 2 min | 8.5 GiB | `mtp/rt` 0.82 GB |
| draft vocab | `draft_vocab.py --gguf … --base data/draft_vocab_en.bin --add cjk --out …/rt/draft_vocab.bin` | 2 s | 0.2 GiB | 106,299 ids (65,774 CJK added) |
| verify | `nvfp4_verify.py --model $MODEL --experts 4 --scan-all` | 1.4 min | 0.4 GiB | **lossless**: 7.03 GiB of scale bytes, sign bit set 0, NaN 0; sampled experts bitwise equal to ModelOpt's decoding |
| verify | `nvfp4_verify_gguf.py --model $MODEL --gguf … --ple-gguf <IQ4_NL>` | 24 s | 1.1 GiB | **OK**: expert order, scale order and dense types; the IQ4_NL rows match RadixArk's FP8 rows at cosine 0.9967 (worst 0.9956 of 64), so it is the same table |

Cross-checks:

- `mtp/rt/{experts.bin, dense.bin, dense.txt}` are byte-identical to upstream Q2_0's `mtp/rt`, which was built
  from Qwen's original head. This agrees with RadixArk's `audit_unchanged_report.json` (31 MTP tensors unchanged).
- The en+CJK draft vocabulary is byte-identical to upstream's default `data/draft_vocab.bin`.

The n-gram table was not rebuilt as FP8 (51 GB). The fork still reads IQ4_NL (`src/kernels/ngram.cpp`), and its
README puts the cost at a first-token KL of 0.0026.

Disk use:

| location | contents | size |
| --- | --- | ---: |
| `/data/users/zephyr-zdz/models/strata-nvfp4` (HDD, sda1) | GGUF + embedding 75.21 GB, pack 69.50 GB, MTP runtime 0.82 GB | 145.5 GB (151.6 GB before the cleanup) |
| `~/models-ssd/strata/Q2_0/` (NVMe) | the IQ4_NL table (main's copy, shared with upstream Strata) | 28.8 GB |
| `~/Projects/Strata-nvfp4` (NVMe) | code, `build/` 170 MB, `.venv` 1.1 GB, llama.cpp 167 MB | 1.5 GB |
| `~/scratch/strata-nvfp4` (HDD, sdb1) | logs only; the 74 GB converter spool is gone | 68 MB |

Deleted after the run, since the engine reads only `mtp/rt`: `mtp/extract` (5.21 GB) and `mtp/mtp-q2_0.gguf`
(0.89 GB). The GGUF stays: the engine serves the dense tensors from it (`--native`, `--native-dense-gguf`).
`experts.bin` holds the NVFP4 scale tails, and the GGUF does not.

## Run (`bench/bw2-01/run-strata-nvfp4.sh`, config `bench/bw2-01/strata-nvfp4-bw2.json`)

```bash
bench/bw2-01/run-strata-nvfp4.sh start      # serve.server on 127.0.0.1:8097, GPU 1 (config "gpu": 1), logs in ~/scratch/strata-nvfp4
bench/bw2-01/run-strata-nvfp4.sh stop       # stops it and prints GPU 1's memory and processes
```

The engine arguments, all from the config:

```
--pack $OUT/packs/radixark-nvfp4 --native $OUT/models/radixark-nvfp4.gguf --native-dense-gguf $OUT/models/radixark-nvfp4.gguf
--ple-gguf ~/models-ssd/strata/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --ple-io direct
--embd-gguf $OUT/models/token-embd-bf16.gguf --mtp $OUT/mtp/rt --spec 4 --spec-min-p 0.5 --prefill auto
--expert-profile data/expert-profile.bin --expert-cache auto --max-context 262144 --kv int8 --resident-experts
```

- **`--resident-experts`** is upstream's low-RAM resident mode. The resident arena is off, and every expert is
  mapped from `experts.bin`. Only the experts in the 5,375 VRAM slots that the prompt path borrows for a 32K chunk
  stay page-locked in RAM (13.85 GiB), so they refill without file reads. Without it, a 245 GB box pins all
  63 GiB (upstream's default above 96 GB), and `--low-ram` would make the budget the available RAM less 6 GiB.
- **`--ple-io direct`** reads the n-gram table from the NVMe with O_DIRECT: about 40-160 MB per request (16 rows of
  90 B per token, in 4 KiB pages), and none of the table in the page cache.
- **The API key** is in `~/.config/strata/strata-nvfp4.key` (0600, created on the first start) and reaches the
  server as `$STRATA_API_KEY`. It is not in the repo or the config.
- **Thinking off:** requests use `"reasoning_effort": "none"`. The chat template then renders an empty
  `<think>\n\n</think>` block, which is the same rendering as `enable_thinking=false`.

### VRAM and RAM (engine log, `bench/bw2-01/results/engine-record.log`)

| | |
| --- | --- |
| dense weights | 2,948 MiB of native Q8_0 projections (attention, GDN out, shared experts, LM head) + 1,467 MiB of pack weights (routers, hyper-connections, gates; BF16/F32) |
| MTP draft layer | 1,098 MiB (experts 675, dense 111) + a 276 MiB draft head over 106,299 tokens |
| expert cache | **24,573 of 24,576 experts, 63.28 GiB** (`expert cache auto` asked for 24,576 and the sized slots came to 24,573. The 3 missing pairs would run on the CPU when routed; the reported decode hit rate was 1.0) |
| K/V | int8, with the fork's defaults: Hadamard rotation and elastic growth up to 262,144 cells. The serve log does not print its size; at full length it is about 3.35 GiB including the draft layer's (README). |
| verify buffers | 77 MiB (windows of up to 6 tokens) |
| free with everything loaded | 21,060 MiB |
| nvidia-smi, GPU 1 | 76,190 MiB after loading, 76,310 MiB during requests |
| host | page-locked 13.85 GiB (`FileExpertSource` cache complement). Engine RSS 18.1 GiB = anonymous 1.85 + file 0.28 + shmem 15.96 (the page-locked copy plus CUDA host buffers). VmHWM 78.8 GiB, reached while the start read `experts.bin` through the mapping. Those pages are reclaimable page cache, not held memory. |
| CPU while decoding | 1.02 cores (one host thread). The CPU expert pool has nothing to do at a 100% hit rate. |
| cold start | **367 s** to "ready", with 75.6 GB read from sda (about 206 MB/s): 63.28 GiB of experts + the dense GGUF parts, with the page cache mostly cold. A warm start was not measured. |

## Results

Server API (`/v1/chat/completions`, streaming), greedy, thinking off (`reasoning_effort: none`), MTP `--spec 4
--spec-min-p 0.5`, 262K context. A fresh nonce goes before every prompt (no prompt-cache reuse). Every answer ran
to its own end of turn (`finish: stop` in every row; `max_tokens` 3000 was only a cap). The numbers are the
engine's own (`/metrics`):

- decode tok/s = generated / decode_ms
- tok/round = generated / (generated - accepted drafts)

Raw rows: `bench/bw2-01/results/record-*/results.json`.

| prompt | runs | prompt tokens | generated | prefill tok/s | client TTFT s | decode tok/s | TPOT ms | tok/round | draft accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code, 1K | 3 | 1,024 | 1,185-1,504 | 1,895 / 2,308 / 2,321 | 0.55 / 0.46 / 0.45 | **169.0 / 163.1 / 171.3** | 5.92 / 6.13 / 5.84 | 2.46 / 2.30 / 2.49 | 0.65-0.67 |
| code, 32K | 3 | 32,768 | 1,040-1,261 | **5,868 / 5,829 / 5,793** | 5.63 / 5.66 / 5.71 | **161.1 / 152.7 / 157.8** | 6.21 / 6.55 / 6.34 | 2.46 / 2.30 / 2.39 | 0.63-0.68 |
| random words, 1K | 1 | 1,024 | 2,188 | 2,230 | 0.48 | **180.8** | 5.53 | 2.64 | 0.72 |
| Chinese (CJK drafts) | 1 | 70 | 465 | (short) | 0.26 | **160.6** | 6.23 | 2.25 | 0.63 |
| short chat, English (CPU sample) | 1 | 32 | 663 | - | - | 170.7 | 5.86 | 2.46 | 0.66 |

- **Expert cache hit rate:** 1.0 in every row.
- **Disk reads during the requests:** sda 0 MB in every row. nvme0n1 35-161 MB (the n-gram rows, O_DIRECT).
- **The prompts:**
  - The code prompt is the upstream Q2_0 run's: a synthetic Python module plus "explain it in at least 600 words".
  - The random-words prompt is that run's stand-in for SGLang's random dataset (the same seed and word list). The
    model wrote a coherent 2,188-token story from it and stopped by itself.
  - The Chinese prompt asks for a 600-character explanation of hash tables.

### Bytes per token (an estimate from the file sizes, not a profile)

| | bytes | at 1.79 TB/s |
| --- | ---: | ---: |
| dense, per main-model pass (VRAM-resident weights above, LM head included) | ~4.63 GB | 2.6 ms |
| experts per token (10 x 48 x 2,764,816 B) | 1.33 GB | 0.74 ms |
| one plain token (dense + experts) | ~5.96 GB | **3.3 ms** |
| a verify round: dense once, experts as the union over the window's rows, plus up to 4 draft steps (~0.4 GB each: draft dense, its 10 Q2_0 experts, the 276 MiB head) | ~8-10 GB | ~4.5-5.6 ms |
| measured round: TPOT x tok/round = 5.9 ms x 2.4 | | **~14 ms** |

At batch 1 the engine runs at about a third of HBM bandwidth. This agrees with strata-bench's measured Q2_0 GPU
floor on the same card (6.83 ms per plain token against a ~1.5 ms byte floor): the round is bound by kernel
latency and launches, not by bandwidth. Not measured here: the GPU-only floor (`--gpu-only-full`) and the
per-round GPU / draft split.

## Quality sanity (`bench/bw2-01/quality.py`, `results/quality-record.json`)

Greedy, thinking off, the same questions as the upstream Q2_0 run, plus one coding prompt:

- **GSM8K:** 8/8 (18, 3, 70000, 540, 20, 64, 260, 160), with the step-by-step reasoning in the answer text.
- **Chinese, MoE in 150 characters:** correct and fluent, within the limit, with a hospital-triage analogy.
- **Chinese, apples word problem:** correct steps, 答案：6.
- **Code:** `merge_intervals` passed all 5 asserts, including the touching and unsorted cases.

Eight easy questions are a smoke test, not a benchmark.

## Against the other two

| | TPOT at 1K | TPOT at 32K | random 1K | prefill 32K | experts |
| --- | ---: | ---: | ---: | ---: | --- |
| **strata-nvfp4, RadixArk NVFP4** (this run) | 5.84-6.13 ms (163-171 tok/s) | 6.21-6.55 ms (153-161) | 5.53 ms (181) | ~5,830 tok/s | 4.5-bit NVFP4, 63.3 GiB, all in VRAM |
| upstream Strata Q2_0 (strata-bench, same card, `--resident-experts`) | 5.38 ms (186) | 6.17 ms (162) | 7.38 ms (135) | 5,928 tok/s | 2-bit, 31.6 GiB, all in VRAM |
| SGLang NVFP4 + MTP (cookbook reference, random prompts) | ~6.0 ms (148-167) | - | - | - | NVFP4, batch up to 16 |

Two caveats on the Q2_0 row:

- It stopped at a fixed 256 tokens, and these rows ran to each answer's end (1,000-1,500 tokens). The random-words
  row differs most: there the NVFP4 model accepted more drafts (0.72 against 0.52).
- The SGLang number is not on the same requests.

## Answers to the lead's questions

- **Draft-weight fix of 2026-10-02:** I found no such commit. Nothing in the fork's history up to
  v0.1.35-nvfp4.1, or in upstream `Niko1221/Strata` main up to `36fa455` (0.1.36, 2026-10-02), fixes the draft
  weights. Upstream 0.1.36's draft-related change is only a "draft head does not fit" hint (#474). The head used
  here is the checkpoint's own, which RadixArk reports as unchanged from Qwen's, and it packs byte-identically to
  upstream 0.1.35's `mtp/rt`.
- **Draft vocabulary:** en/code + CJK: `data/draft_vocab_en.bin` (40,525 ids) plus all Han, kana, Hangul and CJK
  punctuation, 106,299 ids in all, identical to upstream's default. The fork's default (en + Cyrillic, 58,963 ids)
  is kept beside it as `mtp/draft_vocab_fork_default.bin`. Not A/B-tested (scope cut).
- **Thinking:** `reasoning_effort: "none"`, which renders the same as `enable_thinking=false`.

## Not measured (scope cut)

- `--mmap-experts` (nothing pinned) against `--resident-experts`.
- `--ple-io mmap` against `direct`.
- The draft vocabulary A/B.
- 4K prompts, and thinking on.
- A warm start.
- The GPU-only floor and the per-round timing breakdown.
- Prompts above 32K.

## Left in place

- **Server:** stopped at 04:27:41. GPU 1 is back at 23 MiB with no processes (`nvidia-smi`).
- **Model files:** kept in `/data/users/zephyr-zdz/models/strata-nvfp4`. The n-gram table is the NVMe copy
  (main's).
- **Logs:**
  - `bench/bw2-01/logs/`: prepare logs, `/usr/bin/time -v` outputs, the RSS trace, the build log and the start's
    disk counters.
  - `bench/bw2-01/results/`: the request rows, the engine and server logs, and `/proc/pressure/io` before and after.
