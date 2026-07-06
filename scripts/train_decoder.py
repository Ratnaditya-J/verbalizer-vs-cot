"""Train the LatentQA-style decoder: subject activations -> yes/no claim.

The decoder is the SUBJECT model plus a LoRA adapter (no second model's
parametric priors between the activation and the claim). Training pairs are
(activation at subject layer L, behavioral label) from the decoder's TRAINING
families only - never the audit families (the train/eval leakage firewall in
sieve-audit refuses to verbalize any family listed in the checkpoint's
metadata). The training signal is the behavioral label (followed vs resisted),
never the hint text, mirroring the organism's labeling discipline.

Mechanics: the scaled activation is injected at the final token of the fixed
question prompt (same injection as inference), and a LoRA on the decoder's
attention projections is trained with a class-weighted logistic loss on
logit(Yes) - logit(No) at the next-token position. Early stopping on a
stratified validation split WITHIN the training families; the audit families
are never touched.

Outputs a checkpoint dir: peft adapter + metadata.json (question, inject
layer, patch scale, train families, subject model, layer, seed, val AUROC).
Weights and activations stay LOCAL per the data policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class BatchInject:
    """Forward hook: overwrite the final-token hidden state of EVERY batch row
    with a (scaled) per-row injected vector. The training-time counterpart of
    the adapter's PatchHook (which is batch-size-1, prefill-only)."""

    def __init__(self):
        self.batch = None      # torch (B, d_model), set per step

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() != 3 or self.batch is None:
            return output
        modified = hidden.clone()
        modified[:, -1, :] = self.batch.to(hidden.dtype).to(hidden.device)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


def _first_token_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks and toks[0] not in ids:
            ids.append(toks[0])
    return ids


def _auroc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, nargs="+", required=True,
                        help="labeled.jsonl file(s): prompt, cot, family, label")
    parser.add_argument("--train-families", nargs="+", required=True)
    parser.add_argument("--subject-model", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layer", type=int, required=True,
                        help="subject layer the activations are read from")
    parser.add_argument("--inject-layer", type=int, required=True)
    parser.add_argument("--patch-scale", type=float, default=3.0)
    parser.add_argument("--question", required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="checkpoint dir")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from sieve_audit.adapters.hf_steering_runner import (
        _hidden_at_layer,
        _input_device,
        _load_model,
    )

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- training records: TRAIN families only, hard-checked ---
    records = [r for p in args.labeled for r in _rows(p)]
    families = sorted({r["family"] for r in records})
    train_fams = sorted(args.train_families)
    unknown = sorted(set(train_fams) - set(families))
    if unknown:
        raise SystemExit(f"train families {unknown} not present in the data")
    records = [r for r in records if r["family"] in train_fams]
    if not records:
        raise SystemExit("no training records after the family filter")
    y = np.array([int(r["label"]) for r in records])
    print(f"[train] {len(records)} records from families {train_fams} "
          f"({int(y.sum())} followed / {int((1 - y).sum())} resisted)")

    # --- subject activations (plain model, same read as the audit) ---
    class _A:
        model, dtype = args.subject_model, args.dtype
    model, tokenizer = _load_model(_A)
    acts = []
    for i, r in enumerate(records):
        text = r["prompt"] + "\n" + r["cot"]
        acts.append(_hidden_at_layer(model, tokenizer, text, args.layer, pool="last"))
        if (i + 1) % 50 == 0:
            print(f"  activations {i + 1}/{len(records)}")
    X = np.stack(acts).astype(np.float32)

    # --- stratified train/val split (within train families) ---
    idx = np.arange(len(y))
    val_idx = np.concatenate([
        rng.choice(idx[y == c], max(1, int(args.val_frac * (y == c).sum())),
                   replace=False)
        for c in (0, 1)
    ])
    val_mask = np.zeros(len(y), bool)
    val_mask[val_idx] = True
    print(f"[train] split: {int((~val_mask).sum())} train / {int(val_mask.sum())} val")

    # --- decoder: subject weights + LoRA on attention projections ---
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "v_proj"])
    decoder = get_peft_model(model, lora)
    decoder.print_trainable_parameters()

    ids = tokenizer(args.question, return_tensors="pt").input_ids.to(
        _input_device(model))
    yes_ids = _first_token_ids(tokenizer, ("Yes", " Yes", "yes"))
    no_ids = _first_token_ids(tokenizer, ("No", " No", "no"))
    print(f"[train] yes tokens {yes_ids}, no tokens {no_ids}")

    hook = BatchInject()
    layer_module = decoder.get_base_model().model.layers[args.inject_layer]
    handle = layer_module.register_forward_hook(hook)
    opt = torch.optim.AdamW(
        [p for p in decoder.parameters() if p.requires_grad], lr=args.lr)
    pos_weight = torch.tensor((y[~val_mask] == 0).sum() / max((y[~val_mask] == 1).sum(), 1),
                              device=_input_device(model), dtype=torch.float32)

    def _logit_diff(h_batch: np.ndarray) -> "torch.Tensor":
        hook.batch = torch.from_numpy(args.patch_scale * h_batch)
        batch_ids = ids.expand(len(h_batch), -1)
        logits = decoder(input_ids=batch_ids).logits[:, -1, :].float()
        l_yes = torch.logsumexp(logits[:, yes_ids], dim=-1)
        l_no = torch.logsumexp(logits[:, no_ids], dim=-1)
        return l_yes - l_no

    def _val_auroc() -> float:
        decoder.eval()
        scores = []
        with torch.no_grad():
            for s in range(0, int(val_mask.sum()), args.batch_size):
                hb = X[val_mask][s:s + args.batch_size]
                scores.extend(_logit_diff(hb).cpu().numpy().tolist())
        decoder.train()
        return _auroc(y[val_mask], scores)

    tr_idx = idx[~val_mask]
    best, best_epoch = -1.0, -1
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.max_epochs):
        rng.shuffle(tr_idx)
        losses = []
        for s in range(0, len(tr_idx), args.batch_size):
            b = tr_idx[s:s + args.batch_size]
            diff = _logit_diff(X[b])
            target = torch.from_numpy(y[b].astype(np.float32)).to(diff.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                diff, target, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        val = _val_auroc()
        print(f"[train] epoch {epoch}: loss {np.mean(losses):.4f}, val AUROC {val:.3f}")
        if val > best:
            best, best_epoch = val, epoch
            decoder.save_pretrained(str(args.out))
        elif epoch - best_epoch >= args.patience:
            print(f"[train] early stop (best val AUROC {best:.3f} @ epoch {best_epoch})")
            break
    handle.remove()

    metadata = {
        "question": args.question,
        "inject_layer": args.inject_layer,
        "patch_scale": args.patch_scale,
        "train_families": train_fams,
        "subject_model": args.subject_model,
        "decoder_model": args.subject_model,
        "layer": args.layer,
        "seed": args.seed,
        "val_auroc_train_families": best,
        "n_train": int((~val_mask).sum()),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "target_modules": ["q_proj", "v_proj"]},
        "checkpoint_id": f"organism-v2-seed{args.seed}",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=1))
    print(f"[train] checkpoint (LOCAL ONLY) -> {args.out}; "
          f"best val AUROC {best:.3f} (train families only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
