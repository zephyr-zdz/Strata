#!/usr/bin/env python3
"""Quick quality sanity for bw2-01: 8 GSM8K test questions, two Chinese prompts and one coding prompt (its function
run against a few asserts) through the OpenAI API.  The same questions as the upstream Q2_0 run's quality.py."""
import argparse
import subprocess
import sys
import json
import re
import time
import urllib.request
from pathlib import Path

GSM8K = [
    ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her "
     "friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
     "How much in dollars does she make every day at the farmers' market?", 18),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", 3),
    ("Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This "
     "increased the value of the house by 150%. How much profit did he make?", 70000),
    ("James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he "
     "run a week?", 540),
    ("Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and "
     "vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the "
     "morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 "
     "cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the "
     "size of Wendi's flock is 20 chickens?", 20),
    ("Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass "
     "costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?", 64),
    ("Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many "
     "sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", 260),
    ("Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the "
     "download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the "
     "download from the beginning. How long does it take to download the file?", 160),
]
SUFFIX = "\nSolve it step by step, then give the final answer on the last line as 'Answer: <number>'."
CHINESE = [
    "请用中文简要解释什么是混合专家模型（MoE），并用一个生活中的类比说明。不超过150字。",
    "小明有12个苹果，给了小红三分之一，又吃掉了2个，还剩几个？请用中文写出计算过程，最后一行写“答案：<数字>”。",
]


CODE = ("Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges "
        "overlapping closed intervals and returns them sorted by start. Touching intervals such as [1,2] and [2,3] "
        "merge. Reply with one ```python code block containing only the function.")
CODE_TESTS = """
assert merge_intervals([]) == []
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([[5,7],[1,2],[2,3]]) == [[1,3],[5,7]]
assert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]
print("tests passed")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--api-key-file", type=Path, default=Path.home() / ".config/strata/strata-nvfp4.key")
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    headers = {"Content-Type": "application/json"}
    if a.api_key_file.exists():
        headers["Authorization"] = "Bearer " + a.api_key_file.read_text().strip()

    def chat(text):
        body = {"model": "strata", "messages": [{"role": "user", "content": text}], "temperature": 0,
                "reasoning_effort": a.reasoning, "max_tokens": a.max_tokens}
        t0 = time.perf_counter()
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=1800) as r:
            res = json.load(r)
        return res, time.perf_counter() - t0

    rows, correct = [], 0
    for q, gold in GSM8K:
        res, dt = chat(q + SUFFIX)
        msg = res["choices"][0]["message"]
        text = msg.get("content") or ""
        m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text) or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        got = float(m[-1].replace(",", "")) if m else None
        ok = got is not None and abs(got - gold) < 1e-6
        correct += ok
        rows.append({"q": q, "gold": gold, "got": got, "ok": ok, "s": round(dt, 2), "usage": res.get("usage"),
                     "finish": res["choices"][0].get("finish_reason"), "content": text,
                     "reasoning": msg.get("reasoning_content")})
        print(f"{'OK ' if ok else 'BAD'} gold={gold} got={got} {dt:.1f}s {res.get('usage')}", flush=True)
    for q in CHINESE:
        res, dt = chat(q)
        msg = res["choices"][0]["message"]
        rows.append({"q": q, "s": round(dt, 2), "usage": res.get("usage"), "content": msg.get("content"),
                     "reasoning": msg.get("reasoning_content"), "finish": res["choices"][0].get("finish_reason")})
        print(f"ZH {dt:.1f}s {res.get('usage')}\n{msg.get('content')}\n", flush=True)
    res, dt = chat(CODE)
    text = res["choices"][0]["message"].get("content") or ""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    run = subprocess.run([sys.executable, "-c", (m.group(1) if m else text) + CODE_TESTS], capture_output=True,
                         text=True, timeout=30)
    code_ok = run.returncode == 0
    rows.append({"q": CODE, "s": round(dt, 2), "usage": res.get("usage"), "content": text, "code_ok": code_ok,
                 "stderr": run.stderr[-2000:], "finish": res["choices"][0].get("finish_reason")})
    print(f"CODE {'OK ' if code_ok else 'BAD'} {dt:.1f}s {res.get('usage')}\n{text}\n{run.stderr[-500:]}", flush=True)
    print(f"GSM8K {correct}/{len(GSM8K)}, code {'pass' if code_ok else 'FAIL'}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"reasoning": a.reasoning, "gsm8k_correct": correct, "code_ok": code_ok, "rows": rows},
                                ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
