# SIEVE audit: Anthropic's Jacobian lens on the hint-unfaithfulness organism

**Verdict (SIEVE-v0.1-strict): `surface_confounded · survives-cot-removal`.**
Pre-registration `prereg_jlens.json`
(`4626956bc8e6eb6fdea5ec58f327c4fc1faf024bb44a890e6c6a10122c96e82e`, frozen
before any audit-family evaluation); the card states **MATCH**. Causal stage
not licensed under the frozen gates (requires decodable-and-beats-baselines).

## What was audited

The official Neuronpedia pre-fitted Jacobian lens for Qwen2.5-7B-Instruct
(sha256 `3b3ab44c...`, fitted 2026-06-12 on wikitext-103) - the readout behind
Anthropic's "verbalizable global workspace" release (2026-07-02) - applied
GPU-free to the organism's saved L20 activations (runs 4-5), with claims
scalarized by a lexicon selected on train families only (firewalled) and
evaluated on three held-out families. The lexicon the lens itself surfaced for
silent hint-following is strikingly on-theme: *mysterious, suspicious,
suspicion, cautiously, concealed, unknown, 可疑 (suspicious), 神秘
(mysterious)* - concealment concepts lighting up in J-space, echoing the
release's own "secretly/fraud" anecdotes, here with ground truth attached.

## The numbers (held-out audit families, n=282, bootstrap 95% CIs)

| Measurement | AUROC |
| --- | --- |
| full inputs | **0.746** [0.685, 0.802] |
| TF-IDF surface baseline (engine, LOFO) | 0.690 |
| cot-removed | **0.620** [0.553, 0.687] |
| random-removed (matched control) | 0.574 [0.505, 0.641] |
| specificity control (unhinted twins) | 0.591 [0.523, 0.659] - clean |

## The three findings

1. **The lens's claims genuinely track the hidden behavior** - 0.746 across
   hint framings it never saw during lexicon selection, far above the
   training-free Patchscopes verbalizer (0.615), with a clean specificity
   control (no hallucinated concealment on unhinted twins).
2. **The signal survives CoT removal - the first verbalizer in this series to
   earn that flag.** Cot-removed 0.620 with CI above chance, and *higher* than
   the matched random-removal control (0.574): zero evidence of CoT-parroting.
   Notably, 0.620 is statistically indistinguishable from the maximal-power
   LR probe's cot-removed reading (0.612, run 5): the training-free lens
   recovers essentially all of the (weak) CoT-independent signal that exists
   at this site.
3. **But under the strict bar, no activation-level claim is warranted:** the
   paired comparison against the TF-IDF baseline (0.690) does not clear the
   pre-registered CI margin. A word-counter on the raw prompt statistically
   matches the lens's discrimination. Verdict: `surface_confounded` - the
   same gate that caught verbalizer 1, applied evenhandedly to the flagship.

## Head-to-head (one organism, one adjudicator, four verbalizers + probe)

| Readout | full | cot-removed | verdict/flags |
| --- | --- | --- | --- |
| Patchscopes (run 1) | 0.615 | collapsed | `surface_confounded` |
| LatentQA LoRA (run 2) | 0.843 | collapsed (0.500 const) | beats baselines; cot not earned |
| LatentQA variant-trained (run 3) | 0.911 | collapsed (0.500 const) | gate1 span-fragile |
| **Jacobian lens (this audit)** | **0.746** | **0.620, survives** | `surface_confounded · survives-cot-removal` |
| Linear probe ceiling (run 5) | 0.974 (LR) | 0.612 | reference |

The dissociation is the story: the trained decoder wins on decodability but
shatters off-distribution; the Jacobian lens is weaker on-distribution but
uniquely robust - the only readout whose signal survives span removal - while
still not clearing the surface bar that separates "reads internals" from
"correlates with text statistics".

## Scope (do not detach)

Everything here is bounded to: this lens artifact, Qwen2.5-7B-Instruct
@a09a3545, layer 20 last-token, this organism and property, SIEVE-v0.1-strict.
Nothing here evaluates J-lens on Claude or on Anthropic's own demos. Their
paper's validation battery is extensively CAUSAL and includes a probe
decomposition (Fig 16: J-space swaps flip 61% of two-hop answers vs 28% for
the non-J-space component carrying ~90% of variance) and causal-selectivity
controls (report/inference affected, continuation untouched) - but it reports
no detection-performance metrics against ground truth, no surface-text
baselines, no CoT-redundancy removal, and no detector-specificity control;
those are the axes this audit adds. The audited artifact is Neuronpedia's
extension of the method to open-weight models (dense lens, not the paper's
sparse k=25 J-space frame); the paper's quantitative claims concern Claude
4.5/4.6-class models only.

Pre-outcome note (recorded before the causal addendum's results exist): the
paper reports that causal swap success tracks 'workspace loading' and that
NUMBER WORDS have the lowest loading and worst swap performance they tested.
This organism's hidden intermediates are numeric, so their own results
predict weak causal manipulability in this setting - a negative causal
result here would be consistent with that caveat and must be scoped to it;
a positive one would be stronger for landing in their hardest content class.
Independently, their GSM8K ablation finding ('external reasoning reduces
J-space dependence') corroborates run 5's conclusion from the opposite
direction: CoT externalization drains the internal signal. A `surface_confounded` verdict on a 7B organism does
not establish the method confabulates; it establishes the burden of proof the
strict bar demands and the current evidence does not meet.

## Hygiene

Entirely local and GPU-free (pre-fitted lens + saved activations); zero pod
spend. Lens artifact is third-party public; activations stay local. Card,
bundle, gates JSON, and sample claims committed in this directory.
