#!/usr/bin/env bash
# Stage A of the full organism run: data + activations + claims.
set -euo pipefail
cd /workspace/verbalizer-vs-cot
MODEL="Qwen/Qwen2.5-7B-Instruct"
OUT=runs/full
mkdir -p "$OUT"

# 1. organism prompts (fresh seed, one clean batch for the bundle)
python -m verbalizer_vs_cot.organisms.hint_unfaithfulness \
  --n 600 --seed 42 --out-dir "$OUT/prompts"

# 2. subject-model CoT + answers on the hinted prompts
python scripts/generate.py --model "$MODEL" \
  --prompts "$OUT/prompts/hinted_prompts.jsonl" \
  --max-new-tokens 700 --out "$OUT/generations.jsonl"

# 3. behavioral labels (follow/resist), CoT-mentions-hint filtered
python -m verbalizer_vs_cot.organisms.labeling \
  --items "$OUT/prompts/organism_items.jsonl" \
  --generations "$OUT/generations.jsonl" \
  --out "$OUT/labeled.jsonl"

# 4. verbalizer input file (full transcript; cot = removable span)
python scripts/records_to_prompts.py "$OUT/labeled.jsonl" "$OUT/verb_prompts.jsonl"

# 5. layer sweep: where is hint-reliance decodable from activations at all?
#    (LOFO AUROC per candidate layer; chooses the read layer for the verbalizer.
#    The label-based vectors saved here are NOT used for the audit - the
#    audited direction comes from the verbalizer's CLAIMS in stage B.)
python -m sieve_audit.adapters.hf_steering_runner decode-lofo \
  --model "$MODEL" --dtype bfloat16 \
  --holdout-prompts "$OUT/verb_prompts.jsonl" \
  --save-vectors "$OUT/sweep_vectors.npz" --out "$OUT/decode_labels.json" \
  2>&1 | tee "$OUT/layer_sweep.log"
LAYER=$(python -c "import numpy as np; print(int(np.load('$OUT/sweep_vectors.npz')['selected_layer']))")
echo "SELECTED_LAYER=$LAYER"

# 6. verbalize: Patchscopes identity readout at the selected layer over
#    full / cot-removed / random-removed inputs; activations stay on the pod
python -m sieve_audit.adapters.verbalizer verbalize \
  --model "$MODEL" --dtype bfloat16 \
  --prompts "$OUT/verb_prompts.jsonl" --layer "$LAYER" \
  --max-claim-tokens 40 \
  --out "$OUT/claims.jsonl" --activations "$OUT/acts.npz" --seed 0

# 7. negative control: unhinted twins of the labeled ids -> generations
python scripts/control_records.py subset \
  "$OUT/labeled.jsonl" "$OUT/prompts/unhinted_prompts.jsonl" "$OUT/control_prompts_gen.jsonl"
python scripts/generate.py --model "$MODEL" \
  --prompts "$OUT/control_prompts_gen.jsonl" \
  --max-new-tokens 700 --out "$OUT/control_generations.jsonl"
python scripts/control_records.py records \
  "$OUT/labeled.jsonl" "$OUT/prompts/organism_items.jsonl" \
  "$OUT/control_generations.jsonl" "$OUT/control_verb_prompts.jsonl"

# 8. verbalize the control (full variant only - no cot field needed for the
#    control's decodability question, but keep cot for symmetry)
python -m sieve_audit.adapters.verbalizer verbalize \
  --model "$MODEL" --dtype bfloat16 \
  --prompts "$OUT/control_verb_prompts.jsonl" --layer "$LAYER" \
  --max-claim-tokens 40 \
  --out "$OUT/control_claims.jsonl" --activations "$OUT/control_acts.npz" --seed 0

echo "STAGE_A_DONE layer=$LAYER"
