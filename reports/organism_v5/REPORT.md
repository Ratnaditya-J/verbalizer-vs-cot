# Organism run 5 - corrected probe-ceiling test (final planned run)

**Outcome (pre-registered): `inconclusive` - neither H2 nor B band met.**
Licensed statement, verbatim from the prereg: "Neither band met, or CIs
overlap: report all six AUROCs, no headline claim."

Pre-registration: `prereg_v5.json`, hash
`d35a84b0bada94a37b90da7b1b7fdb00c8a28003a011555888f5fc7246b3461c`, frozen and
pushed before capture. Audit card states **MATCH**. Gate 0 - the sanity gate
run 4 failed - **passed decisively** under the corrected five-family fold
structure, so unlike run 4, everything below is interpretable.

## The six pre-registered numbers (five-family LOFO, n=485, 167 followed)

| Condition | mean_diff | logistic_regression (stronger) |
| --- | --- | --- |
| full | 0.906 [0.877, 0.933] | **0.974 [0.962, 0.984]** |
| cot_removed (**C**) | 0.565 [0.513, 0.614] | **0.612 [0.562, 0.659]** |
| random_removed (**R**) | 0.583 [0.530, 0.636] | **0.687 [0.638, 0.734]** |

CoT-specific effect **R - C = +0.075 [+0.007, +0.145]** (paired bootstrap).
Specificity control: clean (mean_diff 0.620, LR 0.602 - at the generic
tfidf-level correlate seen in every run, below the 0.70 void line).

Gate outcomes: Gate 0 PASS (0.974 >= 0.83, CI-lo 0.962 > 0.71, LR >= MD).
Gate 1: C = 0.612 meets the H2 ceiling (<= 0.62) but R - C = 0.075 misses the
0.10 CoT-specific margin -> H2 not licensed; C < 0.75 -> B not licensed ->
**inconclusive**.

## All anchors

run-1 training-free verbalizer 0.615 · run-2 decoder 0.843 · run-3 decoder
full 0.911, cot-removed 0.500 (constant), random-removed 0.561 · run-4
mismatched-fold probe 0.686 (voided) · mean-diff 5-family ceiling 0.88 ·
TF-IDF 0.690 · **run-5 LR full 0.974 (the new ceiling)**.

## What the structure shows (within the licensed outcome)

1. **The headline contrast: probe C = 0.612 vs run-3 decoder 0.500-constant.**
   The trained decoder was partially readout-limited: a logistic probe
   extracts a live, above-chance signal from the very inputs on which the
   decoder emitted a constant. The engine's Tier-2 gate independently grants
   `survives-cot-removal` (post-removal AUROC CI-lo 0.562 clears chance). So
   the signal is NOT gone - H2's "reads chance" is false as an absolute.
2. **But it is a shadow of the full signal.** 0.612 against a 0.974 same-
   instrument ceiling: removing the CoT destroys most of the recoverable
   information. Removing a matched random span also hurts badly (0.687), and
   the CoT-SPECIFIC increment beyond that is real but small (+0.075, CI
   excluding zero yet below the pre-registered 0.10 margin).
3. **Instrument findings.** LR = 0.974 redefines the ceiling (the 0.88
   mean-diff anchor and even the 0.911 decoder were not the top); mean-diff
   under-reads this representation by ~0.07 everywhere.

Honest synthesis (interpretation, not a licensed headline): hint-reliance at
7B/L20 is *largely, but not entirely,* carried by the CoT tokens in context -
a weak CoT-independent trace exists, too faint for the H1/B bands and far too
faint to support a verbalization-beats-CoT claim, and never causally tested.

## Closure

Per the prereg's final_run_note, this closes Experiment 1: five runs, five
frozen preregs, every stop and every verdict on a config-hashed card. GLM
(Experiment 2) was never unlocked (requires H1_and_causally_sufficient). The
next step is the write-up, not a sixth run.

Pod: capture-only (~8 min, ~$0.50), terminated, verified 0 pods; balance
$42.45. Activations and probe weights stay local.
