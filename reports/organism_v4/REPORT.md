# Organism run 4 - linear-probe ceiling test on CoT-removed activations

**Outcome (pre-registered): `gate0_fail_pipeline_broken`.** The full-condition
probe did not reproduce the ceiling (0.686 [0.619, 0.746] < 0.83 required), so
per the frozen prereg the cot-removed and random-removed numbers are NOT
interpretable and neither H2 nor B is licensed. The run is a valid stop, not a
finding about the research question.

Pre-registration: `prereg_v4.json`, hash
`f5ec686dd3ca9637ec0ea2cdf74057dd68108b5bb33c695c3c7516ba012b7aeb`, frozen and
pushed before capture. The audit card states **MATCH**.

## Measurements (reported with CIs regardless, per prereg)

| Condition (LOFO probe, audit families) | AUROC |
| --- | --- |
| full | 0.686 [0.619, 0.746] |
| cot_removed | 0.551 [0.486, 0.618] (not interpretable: Gate 0 failed) |
| random_removed | 0.523 [0.449, 0.595] (not interpretable: Gate 0 failed) |
| specificity control (unhinted twins) | 0.536 [0.468, 0.607] - clean |

All anchors: run-1 training-free 0.615; run-2 decoder full 0.843; run-3
decoder full 0.911, cot-removed 0.500 (constant), random-removed 0.561;
linear ceiling (run-1 protocol) 0.88; TF-IDF 0.690. SIEVE verdict on the
run-4 bundle: `surface_confounded` (probe 0.686 does not beat tfidf with
margin) - consistent with the Gate-0 stop.

## Why Gate 0 failed (diagnosis, post-hoc)

The 0.88 "ceiling" anchor was measured under run 1's protocol: LOFO over ALL
FIVE hint families (each probe trains on four families, 485 examples). Run 4,
to stay comparable with the decoder runs, probed the three AUDIT families only
- so each fold trains a mean-diff direction on just TWO families (~190
examples). That estimator is much weaker: the prereg baked in an anchor from a
different fold structure. A second, independently interesting observation: on
these exact activations and families, run 3's trained decoder read 0.911 -
i.e. the mean-diff probe is NOT the maximal-power readout its premise assumed;
a trained readout beats it by a wide margin here.

## What this licenses and what it does not

- Licensed: "the run-4 protocol cannot reproduce its own sanity anchor; the
  probe-ceiling question needs a redesigned protocol." Nothing about H2/B.
- NOT licensed: any claim about CoT-removed signal presence/absence. The
  cot_removed 0.551 and random_removed 0.523 numbers sit above/near chance
  with overlapping CIs and MUST NOT be quoted as evidence either way.

## The corrected design (needs a NEW pre-registration; not run here)

1. Probe with LOFO over all five families (run-1's fold structure - legitimate
   for a probe, which is trained fresh per fold and has no decoder-style
   train/eval firewall concern), restoring the 0.88-comparable setting.
2. Use a stronger linear readout (logistic regression on z-scored activations,
   as in the surface-baseline machinery) alongside mean-diff, pre-registering
   both.
3. Recalibrate Gate 0's sanity anchor to the SAME fold structure the run uses.

## Hygiene

Pod `73w8bqk6lkp9sd`: capture only (~10 min, ~$1), terminated and verified
(0 pods). Activations stay local. GLM remains locked.
