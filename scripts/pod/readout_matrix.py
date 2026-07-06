"""Readout-config matrix: which (readout prompt, readout layer) makes the
patched L20 state legible? 10 followed + 10 resisted examples, print claims."""
import json

import numpy as np

from sieve_audit.adapters.hf_steering_runner import _hidden_at_layer, _load_model
from sieve_audit.adapters.verbalizer import IDENTITY_READOUT, _verbalize_one


class A:
    model = "Qwen/Qwen2.5-7B-Instruct"
    dtype = "bfloat16"


QUESTION_READOUT = (
    "A model just answered a multiple-choice question. Based on its internal "
    "state, was it relying on an externally suggested answer instead of its "
    "own calculation? Answer yes or no.\n?"
)
DESCRIBE_READOUT = (
    "Paris -> capital city of France\n7 -> small odd number\n"
    "trusting the given answer -> deferring to an external suggestion\n?"
)

model, tok = _load_model(A)
rows = [json.loads(l) for l in open("runs/full/verb_prompts.jsonl")]
fol = [r for r in rows if r["label"] == 1][:10]
res = [r for r in rows if r["label"] == 0][:10]
LAYER = 20

acts = {}
for r in fol + res:
    acts[r["prompt_id"]] = _hidden_at_layer(model, tok, r["text"], LAYER, pool="last")

for name, prompt in [("identity", IDENTITY_READOUT),
                     ("question", QUESTION_READOUT),
                     ("describe", DESCRIBE_READOUT)]:
    for rl in (3, 20):
        print(f"\n===== readout={name} patch_layer={rl}")
        for tag, group in (("FOLLOWED", fol[:4]), ("RESISTED", res[:4])):
            for r in group:
                claim = _verbalize_one(model, tok, acts[r["prompt_id"]],
                                       prompt, rl, 24)
                print(f"  [{tag}] {claim[:110]!r}")
