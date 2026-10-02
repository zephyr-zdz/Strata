#!/usr/bin/env python3
"""Serial, fresh-prompt benchmark of a running Strata server on bw2-01 (the NVFP4 fork).

Adapted from the upstream Q2_0 run's bench/bw2-01/benchmark.py (itself from bench/results/2026-09-30-community-rtx-5090)
so the two are comparable: the same code-module filler and nonce per prompt, greedy, thinking off.  Changes:

  * every answer runs to its own end (max_tokens is only a safety cap, and each row records finish_reason), not to a
    fixed 256 tokens - the fork's README warns that tokens after the end-of-turn are a loop, not an answer;
  * a short chat case ("chat") and a Chinese one ("zh") beside the 1K / 4K / 32K prompts;
  * per request: the engine's RSS breakdown and locked/pinned pages (/proc/<pid>/status), GPU memory, MemAvailable,
    and the bytes read from each disk (/proc/diskstats) while the request ran;
  * tokens per verify round, estimated as generated / (generated - accepted drafts): each round commits its
    accepted drafts plus one token.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DISKS = ("sda", "nvme0n1")
CHAT = ("Explain how a hash table works: hashing, collisions (chaining and open addressing), load factor and "
        "resizing, and when a balanced tree is the better choice. About 500 words.")
ZH = ("请用中文详细介绍一下哈希表的工作原理：哈希函数、冲突处理（链地址法和开放寻址法）、负载因子与扩容，"
      "以及什么时候平衡树是更好的选择。大约600字。")


def disk_read_bytes():
    out = {}
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] in DISKS:
            out[f[2]] = int(f[5]) * 512
    return out


def proc_status(pid):
    keys = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmPin", "VmHWM")
    out = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k = line.split(":")[0]
            if k in keys:
                out[k + "_gib"] = round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return out


def engine_pid():
    try:
        return int(subprocess.check_output(["pgrep", "-n", "-f", "build/strata .*--serve"], text=True).split()[0])
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--pack", type=Path, default=Path("/data/users/zephyr-zdz/models/strata-nvfp4/packs/radixark-nvfp4"))
    p.add_argument("--url", default="http://127.0.0.1:8097")
    p.add_argument("--api-key-file", type=Path, default=Path.home() / ".config/strata/strata-nvfp4.key")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", default="chat,zh,1024,4096,32768", help="chat, zh, and prompt token targets")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=3000, help="a cap only: answers end at their end of turn")
    p.add_argument("--reasoning", default="none", help="none | low | medium | high")
    p.add_argument("--gpu", default="1", help="nvidia-smi index to sample VRAM from")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(a.root), str(a.root / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    d = a.pack / "tokenizer"
    vocab = json.loads((d / "vocab.json").read_text())
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tokenizer = Tokenizer(toks, (d / "merges.txt").read_text().splitlines(), json.loads((d / "token_type.json").read_text()))
    template = ChatTemplate(d / "chat_template.jinja")
    headers = {"Content-Type": "application/json"}
    if a.api_key_file.exists():
        headers["Authorization"] = "Bearer " + a.api_key_file.read_text().strip()

    def get(path):
        with urllib.request.urlopen(urllib.request.Request(a.url + path, headers=headers), timeout=30) as r:
            return json.load(r)

    def vram_mib():
        try:
            return int(subprocess.check_output(["nvidia-smi", "-i", a.gpu, "--query-gpu=memory.used",
                                                "--format=csv,noheader,nounits"], text=True).strip())
        except Exception:
            return None

    def mem_available_gib():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 2**20, 1)
        return None

    (a.out / "initial-status.json").write_text(json.dumps(get("/v1/status"), indent=2) + "\n")
    rows = []

    def request(content, maximum=a.max_tokens):
        return {"model": "strata", "messages": [{"role": "user", "content": content}],
                "temperature": 0, "reasoning_effort": a.reasoning, "max_tokens": maximum,
                "stream": True, "stream_options": {"include_usage": True}}

    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))

    def perform(label, req, expected=None):
        body = json.dumps(req).encode()
        print("start", label, "tokens", expected, flush=True)
        chunks, texts, first, finish = [], [], None, None
        pid = engine_pid()
        disk0 = disk_read_bytes()
        started = time.perf_counter()
        wire = urllib.request.Request(a.url + "/v1/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(wire, timeout=3600) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                chunks.append(chunk)
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if text:
                        first = first if first is not None else time.perf_counter() - started
                        texts.append(text)
                    finish = choice.get("finish_reason") or finish
        elapsed = time.perf_counter() - started
        disk1 = disk_read_bytes()
        e = get("/metrics")["requests"][0]
        usage = next((c["usage"] for c in reversed(chunks) if "usage" in c), {})
        gen, acc = e.get("engine_generated") or 0, e.get("drafts_accepted") or 0
        row = {"label": label, "expected_prompt_tokens": expected,
               "request_sha256": hashlib.sha256(body).hexdigest(),
               "client_ttft_s": first, "client_elapsed_s": elapsed, "finish_reason": finish,
               "usage": usage, "engine": e,
               "prefill_tok_s": (e["prompt_tokens"] - (e.get("reused") or 0)) / (e["prompt_ms"] / 1000)
               if e.get("prompt_ms") else None,
               "decode_tok_s": gen / (e["decode_ms"] / 1000) if gen and e.get("decode_ms") else None,
               "tokens_per_round": gen / (gen - acc) if gen > acc else None,
               "draft_accept": acc / e["drafts_offered"] if e.get("drafts_offered") else None,
               "disk_read_mb": {k: round((disk1[k] - disk0.get(k, 0)) / 1e6, 1) for k in disk1},
               "vram_mib": vram_mib(), "mem_available_gib": mem_available_gib(),
               "engine_mem": proc_status(pid) if pid else {}, "text": "".join(texts)}
        if expected is not None and e["prompt_tokens"] != expected:
            raise RuntimeError(f"token count mismatch: {expected} vs {e['prompt_tokens']}")
        if not texts:
            raise RuntimeError(f"no text received for {label}")
        rows.append(row)
        (a.out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        show = {k: row[k] for k in ("client_ttft_s", "finish_reason", "prefill_tok_s", "decode_tok_s",
                                    "tokens_per_round", "disk_read_mb", "vram_mib", "engine_mem")}
        show["generated"] = gen
        print("done", label, json.dumps(show), flush=True)
        return row

    perform("warmup", request("Reply with exactly the word READY.", 16))
    filler = "\n".join(f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
                       for i in range(12000))
    ending = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, "
              "modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    for case in cases:
        for run in range(1, a.runs + 1):
            nonce = f"Benchmark nonce: series-nvfp4-{a.reasoning}-{case}-trial-{run}.\n"
            if case == "chat":
                perform(f"chat-run-{run}", request(nonce + CHAT))
                continue
            if case == "zh":
                perform(f"zh-run-{run}", request(nonce + ZH))
                continue
            target = int(case)
            prefix = nonce + "Review this synthetic Python module:\n"
            lo, hi = 0, len(filler)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if count(request(prefix + filler[:mid] + ending)) <= target:
                    lo = mid
                else:
                    hi = mid - 1
            req = request(prefix + filler[:lo] + ending)
            actual = count(req)
            if actual < target - 20 or actual > target:
                raise RuntimeError(f"could not construct target {target}: {actual}")
            perform(f"tokens-{target}-run-{run}", req, actual)
    measured = [r for r in rows if r["label"] != "warmup"]
    groups = {}
    for case in cases:
        sub = [r for r in measured if r["label"].startswith(("chat-" if case == "chat" else "zh-" if case == "zh"
                                                             else f"tokens-{case}-"))]
        vals = {
            "prompt_tokens": [r["engine"]["prompt_tokens"] for r in sub],
            "output_tokens": [r["engine"]["engine_generated"] for r in sub],
            "finish_stop": [1 if r["finish_reason"] == "stop" else 0 for r in sub],
            "client_ttft_s": [r["client_ttft_s"] for r in sub],
            "prefill_tok_s": [r["prefill_tok_s"] for r in sub if r["prefill_tok_s"]],
            "decode_tok_s": [r["decode_tok_s"] for r in sub if r["decode_tok_s"]],
            "tpot_ms": [1000 / r["decode_tok_s"] for r in sub if r["decode_tok_s"]],
            "tokens_per_round": [r["tokens_per_round"] for r in sub if r["tokens_per_round"]],
            "draft_accept": [r["draft_accept"] for r in sub if r["draft_accept"] is not None],
            "hit_rate": [r["engine"]["hit_rate"] for r in sub if r["engine"].get("hit_rate") is not None],
            "sda_read_mb": [r["disk_read_mb"].get("sda", 0) for r in sub],
            "nvme_read_mb": [r["disk_read_mb"].get("nvme0n1", 0) for r in sub],
            "vram_mib": [r["vram_mib"] for r in sub if r["vram_mib"] is not None],
            "engine_rss_gib": [r["engine_mem"].get("VmRSS_gib") for r in sub if r["engine_mem"].get("VmRSS_gib")],
        }
        groups[case] = {k: {"median": statistics.median(v), "min": min(v), "max": max(v)} for k, v in vals.items() if v}
    (a.out / "summary.json").write_text(json.dumps(groups, indent=2) + "\n")
    print("summary", json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
