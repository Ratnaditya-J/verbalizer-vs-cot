"""labeled_records.jsonl -> the prompts file the sieve verbalizer adapter reads.

The verbalizer's input is the full transcript the subject model produced
(hinted prompt + its own CoT); ``cot`` is the exact removable span for the
Tier-2 `cot` gate. Usage: records_to_prompts.py <labeled_records> <out>.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    records_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(l) for l in records_path.read_text().splitlines() if l.strip()]
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps({
                "prompt_id": r["prompt_id"],
                "text": r["prompt"] + "\n" + r["cot"],
                "cot": r["cot"],
                "label": r["label"],
                "family": r["family"],
            }) + "\n")
    print(f"[records->prompts] {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
