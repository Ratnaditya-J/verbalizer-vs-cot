"""Negative-control records: the verbalizer must not hallucinate hidden state.

Takes the LABELED hinted examples and their UNHINTED twins' generations, and
emits verbalizer prompts for the twins carrying the hinted run's labels. On an
unhinted input the model has no hint to rely on, so there is no hint-reliance
state to read: a faithful verbalizer's claims must NOT decode these transplanted
labels (expected verdict: not_decodable / surface_confounded). If they do
decode, the verbalizer is picking up a per-question confound (e.g. difficulty),
not the hidden property - and the organism result is not usable evidence.

Usage:
  control_records.py subset  <labeled.jsonl> <unhinted_prompts.jsonl> <out.jsonl>
      unhinted prompts restricted to the labeled ids (for scripts/generate.py)
  control_records.py records <labeled.jsonl> <items.jsonl> <unhinted_gens.jsonl> <out.jsonl>
      verbalizer prompts: unhinted text + the twin-run CoT, hinted-run label
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from verbalizer_vs_cot.organisms.labeling import split_cot


def _rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> int:
    mode = sys.argv[1]
    if mode == "subset":
        labeled, unhinted, out = sys.argv[2:5]
        keep = {r["prompt_id"] for r in _rows(labeled)}
        rows = [r for r in _rows(unhinted) if r["prompt_id"] in keep]
        Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[control] {len(rows)}/{len(keep)} unhinted twins -> {out}")
        return 0
    if mode == "records":
        labeled, items, gens, out = sys.argv[2:6]
        label_by_id = {r["prompt_id"]: r["label"] for r in _rows(labeled)}
        item_by_id = {r["prompt_id"]: r for r in _rows(items)}
        n = 0
        with Path(out).open("w") as f:
            for g in _rows(gens):
                pid = g["prompt_id"]
                if pid not in label_by_id or pid not in item_by_id:
                    continue
                cot = split_cot(g["generated_text"])
                f.write(json.dumps({
                    "prompt_id": pid,
                    "text": item_by_id[pid]["unhinted_text"] + "\n" + cot,
                    "cot": cot,
                    "label": label_by_id[pid],   # transplanted: must NOT decode
                    "family": item_by_id[pid]["family"],
                }) + "\n")
                n += 1
        print(f"[control] {n} control verbalizer prompts -> {out}")
        return 0
    raise SystemExit(f"unknown mode {mode!r} (use subset | records)")


if __name__ == "__main__":
    raise SystemExit(main())
