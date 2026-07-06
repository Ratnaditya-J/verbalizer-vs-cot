"""Local (GPU-free) endgame: claims + steering evidence -> bundles -> audits.

Builds the main bundle (verbalization + decodability + cot-leakage + efficacy +
steering) and the negative-control bundle (decodability-only, transplanted
labels), runs both audits, prints the verdicts. Reproducible from the evidence
files + seed.

Usage: build_and_audit.py --run-dir <dir with claims_v2.jsonl, steer.jsonl,
       judged.jsonl, control_claims_v2.jsonl> --out-dir reports/organism_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sieve_audit import AuditConfig, run_audit
from sieve_audit.adapters.verbalizer import build_bundle_from_records
from sieve_audit.bundle import EfficacyRecord, SteeringRecord
from sieve_audit.card import write_card

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
VERBALIZER = "patchscopes:reliance-question@L20,scale3 (training-free)"
PROPERTY = "relying on an externally suggested answer"
SCORERS = ("yesno:lexical", "yesno:graded")
DIST = "hint-unfaithfulness organism v3 (5 hint families x count/digitsum/mod), Qwen2.5-7B behavioral labels"


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--claims-file", default="claims_v2.jsonl")
    parser.add_argument("--control-claims-file", default="control_claims_v2.jsonl")
    parser.add_argument("--verbalizer", default=VERBALIZER,
                        help="verbalizer identity string for the card scope")
    parser.add_argument("--revision", default=None,
                        help="pinned model revision (must match the prereg scope)")
    parser.add_argument("--prereg", type=Path, default=None,
                        help="prereg JSON (sieve prereg output); the MAIN card "
                             "states MATCH/MISMATCH against its hash")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rd = args.run_dir
    prereg = None
    if args.prereg:
        from sieve_audit import PreRegistration
        prereg = PreRegistration.load(args.prereg)

    # --- main bundle ---
    bundle = build_bundle_from_records(
        _rows(rd / args.claims_file),
        target_model=MODEL,
        verbalizer=args.verbalizer,
        layer=LAYER,
        property_tested=PROPERTY,
        prompt_distribution=DIST,
        prompt_license="MIT (generated)",
        claim_scores_out_of_sample=True,  # scorers are fixed rubrics, never tuned on these examples
        revision=args.revision,
        scorer_specs=SCORERS,
    )
    # causal-stage evidence is optional: a run stopped at the robustness gate
    # (gate_robustness.py) has no steering; the audit then reports the
    # correlational stages and refuses the causal verdict, which is the point.
    if (rd / "steer.jsonl").exists():
        steer_rows = _rows(rd / "steer.jsonl")
        bundle.efficacy = [
            EfficacyRecord(
                alpha=r["alpha"], prompt_id=r["prompt_id"],
                resid_delta_norm=r["resid_delta_norm"] or 0.0,
                resid_base_norm=r["resid_base_norm"] or 0.0,
                expected_delta_norm=r["expected_delta_norm"],
                output_changed=bool(r["output_changed"]),
                arm=r["arm"],
            )
            for r in steer_rows if r["resid_base_norm"] is not None
        ]
        bundle.steering = [
            SteeringRecord(arm=r["arm"], alpha=r["alpha"], prompt_id=r["prompt_id"],
                           judge_scores=r["judge_scores"])
            for r in _rows(rd / "judged.jsonl")
        ]
    else:
        print("[audit] no steering evidence (robustness gate stopped the causal "
              "stage); auditing the correlational stages only")
    bundle.validate()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle.save(args.out_dir / "bundle_main.json")
    res = run_audit(bundle, AuditConfig(seed=args.seed),
                    bundle_path=str(args.out_dir / "bundle_main.json"),
                    prereg=prereg)
    write_card(res.card, args.out_dir, "organism_main")
    if res.card.preregistration is not None:
        pre = res.card.preregistration
        print(f"[audit] pre-registration: "
              f"{'MATCH' if pre['matches'] else 'MISMATCH'} "
              f"({pre['declared_hash'][:16]}...)")
        for d in pre["diffs"]:
            print(f"  - {d}")

    # --- negative control: no hidden property, transplanted labels ---
    control = build_bundle_from_records(
        _rows(rd / args.control_claims_file),
        target_model=MODEL,
        verbalizer=args.verbalizer,
        layer=LAYER,
        property_tested=PROPERTY + " (NEGATIVE CONTROL: unhinted twins, transplanted labels)",
        prompt_distribution=DIST + " - unhinted twins",
        prompt_license="MIT (generated)",
        claim_scores_out_of_sample=True,
        scorer_specs=SCORERS,
    )
    control.save(args.out_dir / "bundle_control.json")
    cres = run_audit(control, AuditConfig(seed=args.seed),
                     bundle_path=str(args.out_dir / "bundle_control.json"))
    write_card(cres.card, args.out_dir, "organism_control")

    lk = res.leakage
    print("=== MAIN:", res.card.label)
    print("    decision:", "; ".join(res.card.diagnostics["decision_reasons"][:3]))
    if lk:
        print(f"    cot_leaky={lk.cot_leaky} cot_survives={lk.cot_survives}")
    print("=== CONTROL:", cres.card.label)
    print("    decision:", "; ".join(cres.card.diagnostics["decision_reasons"][:3]))
    print(f"cards -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
