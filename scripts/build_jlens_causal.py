"""J-lens causal addendum: merge correlational + causal evidence, audit.

Rebuilds the correlational sections deterministically (frozen procedure of
preregistration_jlens.yaml), adds efficacy/steering (sufficiency) and ablation
(necessity) from the pod run, and audits against the ADDENDUM prereg
(prereg_jlens_causal.json, hash e8fe7915...). Also runs the pre-registered
secondary scalarization diagnostics (non-gating).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from jlens_audit import (  # noqa: E402
    DIST,
    LAYER,
    MODEL,
    PROPERTY,
    REVISION,
    VERBALIZER_NAME,
    JLensReader,
    _auroc,
    _boot,
    scalarize,
    select_lexicon,
)

ADAPTER = "verbalizer_vs_cot.scripts.jlens_audit:0.1"
DIRECTION_SOURCE = (
    "Jacobian lens (neuronpedia/jacobian-lens qwen2.5-7b-it wikitext, "
    "sha256 3b3ab44cd67c2ad1) applied to L20 last-token activations; "
    "claims scalarized by the frozen train-family lexicon procedure of "
    "preregistration_jlens.yaml (jlex:logprob + jlex:rank); causal stage "
    "on the claim-recovered mean-diff direction per "
    "preregistration_jlens_causal.yaml"
)


def _rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> int:
    from sieve_audit import AuditConfig, PreRegistration, run_audit
    from sieve_audit.bundle import (
        AblationRecord,
        DecodabilityEvidence,
        EfficacyRecord,
        EvidenceBundle,
        LeakageEvidence,
        SteeringRecord,
        VerbalizationEvidence,
    )
    from sieve_audit.card import write_card

    rng = np.random.default_rng(0)
    reader = JLensReader("runs/jlens/Qwen2.5-7B-Instruct_jacobian_lens.pt",
                         "runs/jlens/qwen_unembed.pt")
    train = np.load("runs/organism_v5/acts_topup.npz")
    audit = np.load("runs/organism_v4/acts_v4.npz")
    texts = json.loads(Path("runs/organism_v4/acts_v4.npz.texts.json").read_text())
    y = audit["main_labels"].astype(int)
    fams = audit["main_families"]

    lp_tr = reader.logprobs(train["main_full"])
    y_tr = train["main_labels"].astype(int)
    lexicon = select_lexicon(lp_tr, y_tr)
    scores = {c: scalarize(reader.logprobs(audit[f"main_{c}"]), lexicon)
              for c in ("full", "cot_removed", "random_removed")}
    main_scores = [float(v) for v in scores["full"]]

    # --- causal evidence from the pod run ---
    steer_rows = _rows("runs/jlens_causal/steer.jsonl")
    efficacy = [
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
    steering = [
        SteeringRecord(arm=r["arm"], alpha=r["alpha"], prompt_id=r["prompt_id"],
                       judge_scores={k: v for k, v in r["judge_scores"].items()
                                     if isinstance(v, (int, float)) and math.isfinite(v)})
        for r in _rows("runs/jlens_causal/judged_steer.jsonl")
    ]
    steering = [r for r in steering if len(r.judge_scores) >= 2]
    ablation = [
        AblationRecord(arm=r["arm"], prompt_id=r["prompt_id"],
                       judge_scores={k: v for k, v in r["judge_scores"].items()
                                     if isinstance(v, (int, float)) and math.isfinite(v)})
        for r in _rows("runs/jlens_causal/judged_ablate.jsonl")
    ]
    ablation = [r for r in ablation if len(r.judge_scores) >= 2]

    bundle = EvidenceBundle(
        model=MODEL, revision=REVISION, layers=[LAYER],
        direction_source=DIRECTION_SOURCE,
        prompt_distribution=DIST, prompt_license="MIT (generated)",
        behavioral_metrics=[PROPERTY], adapter=ADAPTER,
        decodability=DecodabilityEvidence(
            texts=texts["main_texts"], labels=[int(v) for v in y],
            probe_scores=main_scores, families=[str(f) for f in fams],
            probe_scores_out_of_sample=True,
        ),
        efficacy=efficacy,
        steering=steering,
        ablation=ablation,
        leakage=LeakageEvidence(
            labels=[int(v) for v in y],
            probe_scores_full=main_scores,
            probe_scores_leak_removed=[float(v) for v in scores["cot_removed"]],
            probe_scores_random_removed=[float(v) for v in scores["random_removed"]],
            probe_scores_cot_removed=[float(v) for v in scores["cot_removed"]],
            probe_scores_cot_random_removed=[float(v) for v in scores["random_removed"]],
        ),
        verbalization=VerbalizationEvidence(
            target_model=MODEL, verbalizer=VERBALIZER_NAME, layer=LAYER,
            token_selection="last", property_tested=PROPERTY,
            texts=texts["main_texts"], cot_texts=texts["main_cots"],
            labels=[int(v) for v in y],
            verbalizer_claim_scores=main_scores,
            families=[str(f) for f in fams],
            claim_scores_out_of_sample=True,
        ),
    )
    bundle.validate()
    out = Path("reports/jlens_audit")
    bundle.save(out / "bundle_causal.json")
    prereg = PreRegistration.load("prereg_jlens_causal.json")
    res = run_audit(bundle, AuditConfig(seed=0),
                    bundle_path=str(out / "bundle_causal.json"), prereg=prereg)
    write_card(res.card, out, "jlens_causal")
    pre = res.card.preregistration
    print(f"[audit] prereg: {'MATCH' if pre['matches'] else 'MISMATCH'} "
          f"({pre['declared_hash'][:16]}...)")
    print(f"[audit] LABEL: {res.card.label}")
    print(f"[audit] causal summary: {res.card.diagnostics['causal_summary']['combined'][:300]}")
    if res.necessity and not res.necessity.inconclusive:
        n = res.necessity
        print(f"[audit] necessity: {'NECESSARY' if n.necessary else 'not necessary'} "
              f"(probe drop {n.probe_drop.point:+.3f} [{n.probe_drop.lo:+.3f},{n.probe_drop.hi:+.3f}], "
              f"random drop {n.random_drop.point:+.3f})")
    elif res.necessity:
        print(f"[audit] necessity inconclusive: {res.necessity.notes}")
    print("[audit] decision:", "; ".join(res.card.diagnostics["decision_reasons"][:4]))

    # --- secondary, non-gating scalarization diagnostics ---
    lp_full = reader.logprobs(audit["main_full"])
    lex128 = select_lexicon(lp_tr, y_tr, k=128)
    s128 = scalarize(lp_full, lex128)
    p, lo, hi = _boot(y, s128, rng)
    print(f"[secondary] K=128 lexicon full AUROC {p:.3f} [{lo:.3f}, {hi:.3f}]")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000, random_state=0)
    lr.fit(lp_tr[:, lexicon], y_tr)
    s_lr = lr.predict_proba(lp_full[:, lexicon])[:, 1]
    p, lo, hi = _boot(y, s_lr, rng)
    print(f"[secondary] LR-on-lexicon full AUROC {p:.3f} [{lo:.3f}, {hi:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
