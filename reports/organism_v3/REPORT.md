# Organism run 3 - on-distribution CoT-redundancy test

**Outcome (pre-registered, verbatim): `gate1_fail_span_fragile`.**

> "Even with variant-augmented training, the 7B decoder remains span-fragile;
> the CoT-redundancy test is not evaluable at this scale/layer."
> Methodological null; report and stop.

Pre-registration: `prereg_v3.json`, hash
`a562790516349329d8356ac1fb053c3deb446623fa834031d0830ea24af7c64d`, frozen and
pushed before any evidence existed. The audit card states **MATCH** against it.

## What was run

The one change vs run 2: the decoder (same architecture - subject-model-as-
decoder, LoRA on the injection pathway at L20, patch scale 3) was trained on
**all three input variants** (full / cot-removed / matched-random-removed),
train families only (`code_comment`, `metadata`; audit families `peer`,
`professor`, `user_belief`; firewall asserted). 1,256 training records
x 3 variants; behavioral labels only. Val AUROC within train families: 0.830.

## The gates (frozen thresholds; run in order)

| Gate | Criterion | Result |
| --- | --- | --- |
| 0 (sanity) | full AUROC >= 0.75 and beats TF-IDF (LOFO) | **PASS** - 0.911 [0.878, 0.940] vs tfidf 0.690 |
| 1 (robustness) | cot-removed AUROC >= 0.65 and no constant collapse | **FAIL** - 0.500 [0.500, 0.500], std 0.000, ONE distinct claim value |
| 2 (H1/H2) | - | not evaluated (Gate 1 failed) |
| 3 (causal) | - | not spent (per prereg) |

Reference: random-removed inputs stayed *alive* (AUROC 0.561 [0.512, 0.611],
std 0.160) while cot-removed inputs collapsed completely.

## All anchors

| Signal | AUROC |
| --- | --- |
| Linear probe ceiling at L20 (run 1 sweep) | 0.88 |
| Run-1 training-free verbalizer (full inputs) | 0.615 (lost to tfidf 0.775) |
| Run-2 trained decoder (full inputs) | 0.843 [0.802, 0.881] |
| **Run-3 variant-trained decoder (full inputs)** | **0.911 [0.879, 0.942]** |
| TF-IDF surface baseline (audit families) | 0.690 |
| Run-3 cot-removed | 0.500 [0.500, 0.500] (constant) |
| Run-3 random-removed | 0.561 [0.512, 0.611] |

Specificity control (unhinted twins, transplanted labels, same decoder):
`surface_confounded` - claims AUROC 0.601 matched by tfidf 0.603, i.e. no
privileged hidden-state signal where there is no hint. Within the prereg's
allowed control outcomes (`not_decodable`/`surface_confounded`).

## Findings licensed by this run

1. **Variant-augmented training does not fix span-fragility at 7B/L20.** The
   decoder was trained on cot-removed activations with behavioral labels and
   still emits one constant claim for every cot-removed audit input. The
   CoT-redundancy question (does verbalization beat reading the CoT?) remains
   unanswered at this scale/layer after three runs; answering it needs a
   different design, not a threshold change.
2. **Variant training helped where it could:** full-input transfer improved
   from 0.843 to 0.911 - the strongest verbalizer decodability yet on this
   organism, still causally unvalidated (Gate 3 was correctly not reached).

## Post-hoc observation (NOT a pre-registered finding)

The training dynamics are *suggestive* of H2 (the information being
CoT-dependent): if hint-reliance were absent from cot-removed activations,
a decoder trained on them could do no better than predicting the base rate,
which is exactly the constant-output behavior observed - while the harder-but-
information-bearing random-removed inputs stayed weakly alive (0.561). The
Tier-2 deltas point the same way without reaching significance (cot drop
0.411 [0.381, 0.442] vs random drop 0.350 [0.296, 0.402]; CIs overlap). This
is an interpretation, not a result: under the frozen prereg, Gate-1 failure
means the CoT test is *not evaluable*, and no H2 claim is licensed. A future
pre-registration could test H2 directly (e.g. train a probe, not a decoder,
on cot-removed activations; or read at multiple layers/positions).

## Costs and hygiene

Run 3 pod: ~1.2h on 1xH100 (~$4); pod `1mdmystb6zk1dm` terminated and
verified (0 pods listed). Activations, weights, and generations stayed on the
pod/local disk; only bundles and cards are committed. Experiment 2 (GLM-5.2)
remains locked: it requires `H1_and_causally_sufficient`, which no run has
produced.
