import sys, json
for line in sys.stdin:
    if line.startswith("done"):
        lab, js = line.split(" ", 2)[1], json.loads(line.split(" ", 2)[2])
        e = js["engine"]
        print(lab, "pp", e["prompt_tokens"], "pf_ms", e["prompt_ms"], "pf tok/s", round(e["prompt_tokens"] / e["prompt_ms"] * 1000) if e["prompt_ms"] else None,
              "gen", e["engine_generated"], "dec tok/s", e["decode_tok_s"], "tpot", round(js["tpot_ms"] or 0, 2), "acc", e["drafts_accepted"], "/", e["drafts_offered"],
              "hit", e["hit_rate"], "ttft", round(js["ttft_s"], 2), "e2e", round(js["elapsed_s"], 2), "vram", js["vram_mib"], flush=True)
    elif not line.startswith("start"): print(line.strip()[:300], flush=True)
