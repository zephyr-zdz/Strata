#!/usr/bin/env python3
"""Serial, fresh-prompt Strata benchmark for bw2-01 (adapted from bench/results/2026-09-30-community-rtx-5090).

Changes against the community script: an API key, a reasoning level, a "random words" prompt style (closer to
SGLang's random dataset than the code prompt), and each row also records nvidia-smi VRAM of the engine's GPU and
MemAvailable, so the footprint is next to the speed it was measured with.
"""
import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pack", type=Path, default=Path("/data/users/zephyr-zdz/models/strata/packs/q2_0"))
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", default="1024,4096,32768")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning", default="none", help="none | low | medium | high")
    parser.add_argument("--style", default="code", choices=["code", "random"])
    parser.add_argument("--gpu", default="1", help="nvidia-smi index to sample VRAM from")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.root), str(args.root / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    directory = args.pack / "tokenizer"
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    tokenizer = Tokenizer(tokens, (directory / "merges.txt").read_text().splitlines(),
                          json.loads((directory / "token_type.json").read_text()))
    template = ChatTemplate(directory / "chat_template.jinja")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key

    def get(path):
        with urllib.request.urlopen(urllib.request.Request(args.url + path, headers=headers), timeout=30) as r:
            return json.load(r)

    def vram_mib():
        try:
            return int(subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=memory.used",
                                                "--format=csv,noheader,nounits"], text=True).strip())
        except Exception:
            return None

    def mem_available_gib():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 2**20, 1)
        return None

    initial = get("/v1/status")
    (args.out / "initial-status.json").write_text(json.dumps(initial, indent=2) + "\n")
    rows = []

    def request(content, maximum=args.max_tokens):
        return {"model": "strata", "messages": [{"role": "user", "content": content}],
                "temperature": 0, "reasoning_effort": args.reasoning, "max_tokens": maximum,
                "stream": True, "stream_options": {"include_usage": True}}

    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))

    def perform(label, req, expected=None):
        body = json.dumps(req).encode()
        (args.out / (label + "-request.json")).write_bytes(body + b"\n")
        print("start", label, "tokens", expected, flush=True)
        chunks, texts, first, finish = [], [], None, None
        started = time.perf_counter()
        wire = urllib.request.Request(args.url + "/v1/chat/completions", data=body, headers=headers)
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
        metrics = get("/metrics")
        engine = metrics["requests"][0]
        usage = next((chunk["usage"] for chunk in reversed(chunks) if "usage" in chunk), {})
        row = {"label": label, "expected_prompt_tokens": expected,
               "request_sha256": hashlib.sha256(body).hexdigest(),
               "client_ttft_s": first, "client_elapsed_s": elapsed, "finish_reason": finish,
               "usage": usage, "engine": engine, "vram_mib": vram_mib(), "mem_available_gib": mem_available_gib(),
               "text": "".join(texts)}
        if expected is not None and engine["prompt_tokens"] != expected:
            raise RuntimeError(f"token count mismatch: {expected} vs {engine['prompt_tokens']}")
        if not texts:
            raise RuntimeError(f"no text received for {label}")
        (args.out / (label + "-raw.json")).write_text(json.dumps({"chunks": chunks, "row": row}, indent=2) + "\n")
        rows.append(row)
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        e = engine
        tpot = e["decode_ms"] / e["engine_generated"] if e.get("engine_generated") else None
        print("done", label, json.dumps({"ttft_s": first, "elapsed_s": elapsed, "tpot_ms": tpot,
                                         "vram_mib": row["vram_mib"], "engine": engine}), flush=True)
        return row

    perform("warmup", request("Reply with exactly the word READY.", 16))
    if args.style == "code":
        # Each nonce occurs before the filler so previous long prompts cannot be reused.
        filler = "\n".join(f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
                           for i in range(12000))
        ending = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, "
                  "modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")
    else:
        # random English words, a stand-in for SGLang's random dataset (random token ids): a prompt with no structure
        rng = random.Random(1234)
        words = [w for w in (t.lstrip("Ġ") for t in tokens[1000:60000] if t) if w.isascii() and w.isalpha()
                 and 3 <= len(w) <= 10]
        filler = " ".join(rng.choice(words) for _ in range(60000))
        ending = "\n\nContinue this text for at least 600 more words."
    for target in map(int, args.targets.split(",")):
        for run in range(1, args.runs + 1):
            prefix = f"Benchmark nonce: series-{args.style}-{args.reasoning}-{target}-trial-{run}.\n" + \
                     ("Review this synthetic Python module:\n" if args.style == "code" else "")
            lo, hi = 0, len(filler)
            while lo < hi:
                middle = (lo + hi + 1) // 2
                if count(request(prefix + filler[:middle] + ending)) <= target:
                    lo = middle
                else:
                    hi = middle - 1
            req = request(prefix + filler[:lo] + ending)
            actual = count(req)
            if actual < target - 20 or actual > target:
                raise RuntimeError(f"could not construct target {target}: {actual}")
            perform(f"tokens-{target}-run-{run}", req, actual)
    measured = [row for row in rows if row["label"] != "warmup"]
    groups = {}
    for target in map(int, args.targets.split(",")):
        subset = [r for r in measured if r["label"].startswith(f"tokens-{target}-")]
        values = {
            "prompt_tokens": [r["engine"]["prompt_tokens"] for r in subset],
            "cached_tokens": [r["engine"].get("reused") or 0 for r in subset],
            "output_tokens": [r["engine"]["engine_generated"] for r in subset],
            "client_ttft_s": [r["client_ttft_s"] for r in subset],
            "client_elapsed_s": [r["client_elapsed_s"] for r in subset],
            "prefill_tok_s": [(r["engine"]["prompt_tokens"] - (r["engine"].get("reused") or 0)) / (r["engine"]["prompt_ms"] / 1000) for r in subset],
            "decode_tok_s": [r["engine"]["engine_generated"] / (r["engine"]["decode_ms"] / 1000) for r in subset],
            "tpot_ms": [r["engine"]["decode_ms"] / r["engine"]["engine_generated"] for r in subset],
            "client_decode_tok_s": [(r["engine"]["engine_generated"] - 1) / (r["client_elapsed_s"] - r["client_ttft_s"])
                                    for r in subset if r["client_ttft_s"] is not None],
            "draft_accept": [r["engine"]["drafts_accepted"] / r["engine"]["drafts_offered"]
                             for r in subset if r["engine"].get("drafts_offered")],
            "hit_rate": [r["engine"]["hit_rate"] for r in subset if r["engine"].get("hit_rate") is not None],
            "vram_mib": [r["vram_mib"] for r in subset if r["vram_mib"] is not None],
            "mem_available_gib": [r["mem_available_gib"] for r in subset],
        }
        groups[str(target)] = {key: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                               for key, v in values.items() if v}
    (args.out / "summary.json").write_text(json.dumps(groups, indent=2) + "\n")
    print("summary", json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
