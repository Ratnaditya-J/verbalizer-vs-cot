#!/usr/bin/env bash
# J-lens causal addendum: steering (sufficiency) + ablation (necessity) on the
# claim-recovered direction. Frozen plan: preregistration_jlens_causal.yaml
# (hash e8fe7915..., committed before this ran). Expects vectors_jlens.npz and
# OPENROUTER_API_KEY in the environment (LLM judge).
set -euo pipefail
cd /workspace/verbalizer-vs-cot
MODEL="Qwen/Qwen2.5-7B-Instruct"
OUT=runs/jlens_causal
mkdir -p "$OUT"

python - <<'PY'
import json
from verbalizer_vs_cot.organisms.hint_unfaithfulness import generate_items
AUDIT = {"peer", "professor", "user_belief"}
items = [it for it in generate_items(600, seed=42) if it.family in AUDIT][:30]
with open("runs/jlens_causal/steer_prompts.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps({"prompt_id": it.prompt_id, "text": it.hinted_text,
                            "family": it.family}) + "\n")
print(f"[jlens-causal] {len(items)} audit-family prompts")
PY

# sufficiency: steering with matched arms (probe + 3 random + orthogonal + wrong_layer)
python -m sieve_audit.adapters.hf_steering_runner steer \
  --model "$MODEL" --dtype bfloat16 \
  --vectors runs/jlens/vectors_jlens.npz \
  --steer-prompts "$OUT/steer_prompts.jsonl" \
  --alpha-mode relative \
  --max-new-tokens 512 --out "$OUT/steer.jsonl"

# necessity: directional ablation (baseline / probe / ablate_random)
python -m sieve_audit.adapters.hf_steering_runner ablate \
  --model "$MODEL" --dtype bfloat16 \
  --vectors runs/jlens/vectors_jlens.npz \
  --eval-prompts "$OUT/steer_prompts.jsonl" \
  --max-new-tokens 512 --out "$OUT/ablate.jsonl"

# three judges on both
python scripts/judge_hintfollow.py \
  --steer-prompts "$OUT/steer_prompts.jsonl" --generations "$OUT/steer.jsonl" \
  --llm-judge "openrouter:openai/gpt-5.2" --out "$OUT/judged_steer.jsonl"
python scripts/judge_hintfollow.py \
  --steer-prompts "$OUT/steer_prompts.jsonl" --generations "$OUT/ablate.jsonl" \
  --llm-judge "openrouter:openai/gpt-5.2" --out "$OUT/judged_ablate.jsonl"

echo "STAGE_JLENS_CAUSAL_DONE"
