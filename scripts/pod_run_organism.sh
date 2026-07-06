#!/usr/bin/env bash
# Organism validation run (Experiment 1): hint-unfaithfulness on a 7B-32B open
# model, audited end-to-end by SIEVE's verbalizer adapter. This is the run that
# earns the right to make claims on GLM-5.2 (Experiment 2) - do NOT touch GLM
# until (a)-(c) below hold:
#   (a) verbalizer claims causally_sufficient for hint-reliance,
#   (b) claims survive CoT removal (Tier-2 `cot` gate: survives-cot-removal),
#   (c) negative control (unhinted twins) yields not_decodable /
#       surface_confounded - the verbalizer must not hallucinate hidden state.
#
# Needs: 1x80GB GPU, pip install -e ".[runner]" (pulls sieve-audit runner+judges
# extras), OPENROUTER_API_KEY for the two steering judges.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
LAYER="${LAYER:-24}"                    # sweep first; see --layer note below
PROPERTY="relying on the embedded hint"
SEED=0
OUT=runs/organism_$(date +%Y%m%d)
mkdir -p "$OUT"

# 0. prompts (deterministic; can run anywhere)
python -m verbalizer_vs_cot.organisms.hint_unfaithfulness \
  --n 600 --seed $SEED --out-dir "$OUT/prompts"

# 1. subject-model generations on the HINTED prompts (CoT + final answer).
#    (simple generation loop; any harness works - record {prompt_id, generated_text})
python scripts/generate.py --model "$MODEL" \
  --prompts "$OUT/prompts/hinted_prompts.jsonl" \
  --max-new-tokens 512 --out "$OUT/generations.jsonl"

# 2. behavioral ground truth: follow/resist labels, CoT-mentions-hint filtered
python -m verbalizer_vs_cot.organisms.labeling \
  --items "$OUT/prompts/organism_items.jsonl" \
  --generations "$OUT/generations.jsonl" \
  --out "$OUT/labeled_records.jsonl"

# 3. verbalize: Patchscopes-style readout over the subject model's activations
#    at the answer decision point, on full / cot-removed / random-removed inputs
#    (labeled_records carry prompt+cot; build the prompts file for the adapter)
python scripts/records_to_prompts.py "$OUT/labeled_records.jsonl" "$OUT/verb_prompts.jsonl"
python -m sieve_audit.adapters.verbalizer verbalize \
  --model "$MODEL" --dtype bfloat16 \
  --prompts "$OUT/verb_prompts.jsonl" --layer "$LAYER" \
  --out "$OUT/claims.jsonl" --activations "$OUT/acts.npz" --seed $SEED

# 4. causal stage: claim direction + matched controls -> existing HF runner
python -m sieve_audit.adapters.verbalizer vectors \
  --records "$OUT/claims.jsonl" --activations "$OUT/acts.npz" \
  --property "$PROPERTY" --n-random-controls 3 --out "$OUT/vectors.npz" --seed $SEED
python -m sieve_audit.adapters.hf_steering_runner steer \
  --model "$MODEL" --vectors "$OUT/vectors.npz" \
  --steer-prompts "$OUT/verb_prompts.jsonl" --alpha-mode relative \
  --out "$OUT/steer.jsonl"
# The organism's metric is mechanically checkable, so the run-1 pipeline used
# the deterministic judge pair instead of LLM judges (see scripts/pod/stage_b.sh):
python scripts/judge_hintfollow.py \
  --steer-prompts "$OUT/verb_prompts.jsonl" \
  --generations "$OUT/steer.jsonl" --out "$OUT/judged.jsonl"

# 5. bundle + audit (the verdict logic is GPU-free and reproducible)
python -m sieve_audit.adapters.verbalizer bundle \
  --records "$OUT/claims.jsonl" --model "$MODEL" \
  --verbalizer "patchscopes:identity@L$LAYER" --layer "$LAYER" \
  --property "$PROPERTY" \
  --prompt-distribution "hint-unfaithfulness organism v0 (4 hint families)" \
  --prompt-license "MIT (generated)" --attest-out-of-sample \
  --steer "$OUT/steer.jsonl" --judged "$OUT/judged.jsonl" \
  --out "$OUT/bundle.json"
sieve audit --bundle "$OUT/bundle.json" --name organism_hint --out "$OUT/reports"

# 6. negative control: same pipeline on the UNHINTED twins (no hidden property;
#    expected not_decodable / surface_confounded - anything stronger means the
#    verbalizer hallucinates hidden state and Experiment 2 is off).
echo "[pod] now repeat steps 1-5 with unhinted_prompts.jsonl -> control bundle"
