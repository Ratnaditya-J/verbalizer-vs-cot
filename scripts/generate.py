"""Subject-model generation loop: hinted prompts -> {prompt_id, generated_text}.

Greedy decoding (reproducible), chat template, one record per prompt. Needs the
[runner] extras.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from sieve_audit.adapters.hf_steering_runner import (
        _generate,
        _load_labeled_prompts,
        _load_model,
    )

    model, tokenizer = _load_model(args)
    prompts = _load_labeled_prompts(args.prompts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, p in enumerate(prompts):
            text = _generate(model, tokenizer, p["text"], None, 0,
                             args.max_new_tokens)
            f.write(json.dumps({
                "prompt_id": p.get("prompt_id", f"p{i}"),
                "generated_text": text,
            }) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(prompts)}")
    print(f"[generate] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
