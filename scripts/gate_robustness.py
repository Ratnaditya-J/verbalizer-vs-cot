"""Pre-registered robustness gate: does the decoder produce a live signal?

Run 1's readout collapsed to a constant under any span removal, so the cot
verdict was never earned. This gate (thresholds frozen in prereg_v2.yaml
BEFORE the run) refuses to proceed to the causal stage unless the trained
decoder's claim scores are non-degenerate:

  PASS requires, on the audit-family claims:
    1. std(full-input claim scores) >= 0.10 and >= 3 distinct values,
    2. std(cot-removed scores) > 0 and std(random-removed scores) > 0
       (no collapse-to-constant under either removal),
  and reports (informational, not gated): the score shift under cot removal
  vs matched random removal.

Exit code 0 = PASS, 1 = STOP (the cot test is uninformative; do not spend the
causal stage; report the failure honestly like run 1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sieve_audit.adapters.verbalizer import scalarize_claims

STD_FULL_MIN = 0.10
DISTINCT_MIN = 3


def _scores(rows: list[dict], key: str, prop: str, scorers) -> np.ndarray:
    s, _ = scalarize_claims([r[key] for r in rows], prop, tuple(scorers))
    return np.asarray(s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--property", required=True)
    parser.add_argument("--scorers", nargs="+",
                        default=["yesno:lexical", "yesno:graded"])
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.claims.read_text().splitlines() if l.strip()]
    full = _scores(rows, "claim_text", args.property, args.scorers)
    cot = _scores(rows, "claim_cot_removed_text", args.property, args.scorers)
    rand = _scores(rows, "claim_cot_random_removed_text", args.property, args.scorers)

    n_distinct = len(np.unique(np.round(full, 3)))
    checks = {
        f"std(full)={full.std():.3f} >= {STD_FULL_MIN}": full.std() >= STD_FULL_MIN,
        f"distinct(full)={n_distinct} >= {DISTINCT_MIN}": n_distinct >= DISTINCT_MIN,
        f"std(cot_removed)={cot.std():.3f} > 0": cot.std() > 0,
        f"std(random_removed)={rand.std():.3f} > 0": rand.std() > 0,
    }
    for desc, ok in checks.items():
        print(f"[gate] {'PASS' if ok else 'FAIL'}  {desc}")
    print(f"[gate] info: mean|shift| cot-removal {np.abs(full - cot).mean():.3f}, "
          f"random-removal {np.abs(full - rand).mean():.3f}")
    if all(checks.values()):
        print("[gate] ROBUSTNESS GATE PASSED: the cot test is informative; "
              "proceed to the causal stage")
        return 0
    print("[gate] STOP: degenerate claim signal (run-1 failure mode); the cot "
          "verdict is NOT earned and the causal stage should not be spent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
