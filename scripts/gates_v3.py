"""Run-3 gate evaluator: Gates 0-2 of preregistration_organism_v3.yaml.

Evaluated on the held-out audit families with bootstrap 95% CIs. Thresholds
come from the frozen prereg and are hard-coded here VERBATIM; this script is
committed before the run so its one interpretive choice (the 'epsilon' of
no_constant_collapse: std > 0.01 and >= 3 distinct values) is frozen too.

  Gate 0 (sanity): full-input AUROC >= 0.75 AND the SIEVE decodability stage
          (leave-one-family-out surface baselines) reports beats_baselines -
          the same comparison run 2 won.
  Gate 1 (robustness): cot-removed AUROC >= 0.65 AND no constant collapse.
  Gate 2 (research question): CoT-specific effect = AUROC(random_removed)
          - AUROC(cot_removed), paired bootstrap.
          H2 if effect >= 0.10 with CI lower bound > 0;
          H1 if |effect| <= 0.05 AND cot_removed AUROC >= 0.65;
          else INCONCLUSIVE.

Exit codes for the pipeline: 0 = Gate 1 passed (proceed to the causal stage,
whatever Gate 2 said); 2 = Gate 0 failed (run void); 3 = Gate 1 failed
(gate1_fail_span_fragile - STOP, the CoT test is again uninformative).
Writes the full evaluation to <claims>.gates.json for the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sieve_audit import AuditConfig, run_audit
from sieve_audit.adapters.verbalizer import build_bundle_from_records, scalarize_claims

FULL_AUROC_MIN = 0.75
COT_REMOVED_FLOOR = 0.65
H2_MARGIN = 0.10
H1_PARITY_BAND = 0.05
COLLAPSE_STD = 0.01
COLLAPSE_DISTINCT = 3
N_BOOT = 2000


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def _boot(y, s, rng) -> tuple[float, float, float]:
    point = _auroc(y, s)
    reps = []
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        reps.append(_auroc(y[i], s[i]))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def _boot_diff(y, s_a, s_b, rng) -> tuple[float, float, float]:
    """AUROC(s_a) - AUROC(s_b), paired over examples."""
    point = _auroc(y, s_a) - _auroc(y, s_b)
    reps = []
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        reps.append(_auroc(y[i], s_a[i]) - _auroc(y[i], s_b[i]))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--property", required=True)
    parser.add_argument("--verbalizer-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = [json.loads(l) for l in args.claims.read_text().splitlines() if l.strip()]
    y = np.array([int(r["label"]) for r in rows])
    scorers = ("yesno:lexical", "yesno:graded")
    cond: dict[str, np.ndarray] = {}
    for key, field in (("full", "claim_text"),
                       ("cot_removed", "claim_cot_removed_text"),
                       ("random_removed", "claim_cot_random_removed_text")):
        s, _ = scalarize_claims([r[field] for r in rows], args.property, scorers)
        cond[key] = np.asarray(s)

    report: dict = {"n": len(rows), "n_followed": int(y.sum()), "conditions": {}}
    for key, s in cond.items():
        p, lo, hi = _boot(y, s, rng)
        report["conditions"][key] = {
            "auroc": p, "ci": [lo, hi], "std": float(s.std()),
            "distinct": int(len(np.unique(np.round(s, 3)))),
        }
        print(f"[gates] {key:15s} AUROC {p:.3f} [{lo:.3f}, {hi:.3f}] "
              f"std {s.std():.3f} distinct {report['conditions'][key]['distinct']}")

    # ---- Gate 0: sanity on full inputs (incl. the engine's baseline fight) ----
    full = report["conditions"]["full"]
    bundle = build_bundle_from_records(
        rows, target_model="Qwen/Qwen2.5-7B-Instruct",
        verbalizer=args.verbalizer_name, layer=20,
        property_tested=args.property,
        prompt_distribution="gate0 decodability check (audit families)",
        prompt_license="MIT (generated)", claim_scores_out_of_sample=True,
        scorer_specs=scorers,
    )
    decod = run_audit(bundle, AuditConfig(seed=args.seed)).decodability
    gate0 = (full["auroc"] >= FULL_AUROC_MIN) and decod.beats_baselines
    report["gate0"] = {
        "passed": bool(gate0),
        "auroc_min": FULL_AUROC_MIN,
        "beats_baselines_lofo": decod.beats_baselines,
        "baseline_aurocs": decod.baseline_aurocs,
    }
    print(f"[gates] GATE 0 {'PASS' if gate0 else 'FAIL'} "
          f"(full {full['auroc']:.3f} >= {FULL_AUROC_MIN}; "
          f"beats baselines: {decod.beats_baselines}; "
          f"baselines {decod.baseline_aurocs})")
    if not gate0:
        _write(args, report, outcome="gate0_fail_signal_lost")
        return 2

    # ---- Gate 1: robustness precondition (the run-2 killer) ----
    cr = report["conditions"]["cot_removed"]
    alive = cr["std"] > COLLAPSE_STD and cr["distinct"] >= COLLAPSE_DISTINCT
    gate1 = (cr["auroc"] >= COT_REMOVED_FLOOR) and alive
    report["gate1"] = {"passed": bool(gate1), "floor": COT_REMOVED_FLOOR,
                       "alive": bool(alive)}
    print(f"[gates] GATE 1 {'PASS' if gate1 else 'FAIL'} "
          f"(cot_removed {cr['auroc']:.3f} >= {COT_REMOVED_FLOOR}, alive={alive})")
    if not gate1:
        _write(args, report, outcome="gate1_fail_span_fragile")
        print("[gates] STOP: even variant-trained decoders are span-fragile at "
              "7B/L20; the CoT test is not evaluable at this scale/layer")
        return 3

    # ---- Gate 2: the research question ----
    eff, eff_lo, eff_hi = _boot_diff(y, cond["random_removed"], cond["cot_removed"], rng)
    report["gate2"] = {"cot_specific_effect": eff, "ci": [eff_lo, eff_hi]}
    if eff >= H2_MARGIN and eff_lo > 0:
        outcome = "H2_info_is_cot_dependent"
    elif abs(eff) <= H1_PARITY_BAND and cr["auroc"] >= COT_REMOVED_FLOOR:
        outcome = "H1_verbalization_beats_cot"
    else:
        outcome = "INCONCLUSIVE"
    report["gate2"]["outcome"] = outcome
    print(f"[gates] GATE 2: CoT-specific effect {eff:+.3f} [{eff_lo:+.3f}, "
          f"{eff_hi:+.3f}] -> {outcome}")
    _write(args, report, outcome=outcome)
    print("[gates] Gate 1 passed: proceed to the causal stage (Gate 3)")
    return 0


def _write(args, report: dict, outcome: str) -> None:
    report["outcome"] = outcome
    out = args.claims.with_suffix(".gates.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"[gates] evaluation -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
