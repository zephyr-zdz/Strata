#!/usr/bin/env python3
"""A chat prompt as the engine's --tokens-file: one user message rendered through the pack's chat template the way
serve.server renders an OpenAI request (reasoning_effort, default none = thinking off), then tokenized.

    python bench/bw2-01/tokens.py --text "..." --out prompt.txt [--reasoning none] [--pack DIR]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from strata_tokenizer import Tokenizer  # noqa: E402
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=Path("/data/users/zephyr-zdz/models/strata-nvfp4/packs/radixark-nvfp4"))
    ap.add_argument("--text", required=True)
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = a.pack / "tokenizer"
    vocab = json.loads((d / "vocab.json").read_text())
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = Tokenizer(toks, (d / "merges.txt").read_text().splitlines(), json.loads((d / "token_type.json").read_text()))
    req = {"messages": [{"role": "user", "content": a.text}], "reasoning_effort": a.reasoning}
    messages, tools, kwargs = openai_to_messages(req)
    ids = tok.encode(ChatTemplate(d / "chat_template.jinja").render(messages, tools, **kwargs), parse_special=True)
    a.out.write_text(",".join(map(str, ids)) + "\n")
    print(f"{a.out}: {len(ids)} tokens")


if __name__ == "__main__":
    main()
