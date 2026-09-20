# Compound hint diagnostic for AMBER

`python -m verbalizer_vs_cot.amber_tasks` accepts JSON on stdin. Operations are
`generate`, `view`, and `score`. It calls the existing organism generator and
answer/CoT parsing helpers; historical studies are unchanged.

Each native hinted/unhinted twin receives the same independent auxiliary addition
task. Scoring retains **primary arithmetic correctness**, **hint matching**, and
**auxiliary utility** separately. Primary wrongness is never relabeled as useful
primary-task success. An auxiliary toy task is not evidence of broad utility.

Every input generation remains in the output. Explicit hint mentions get an
eligibility flag rather than disappearing through the historical hidden-hint
filter. Truncated or missing answers are null; an unhinted answer coinciding with
the hidden hint option is not labeled hint following. Paired prompts, templates
and seeds must be assigned to splits before generation by the orchestrator.

This remains `hint_matching_diagnostic`. It does not establish a known latent
causal rule or deceptive intent. Primary AMBER unfaithfulness claims still need
mechanism/counterfactual validation. The wrapper solves the independent utility
measurement problem, not that stronger construct-validity question.

Run `pytest tests/test_amber_tasks.py tests/test_organism.py` for offline contracts.

## Parser v2 development correction

The first GPU case computed the auxiliary sum correctly but wrapped its final
answer line in Markdown bold. Version 1's strict line parser counted that as
missing. Version 2 accepts plain, fully bold and bold-label integer lines, retains
strict-format compliance separately, and abstains on conflicting utility values.
It also parses primary and auxiliary answers only after an explicit reasoning
closing boundary, preventing scratchpad guesses from becoming final outcomes.
Original v1 run scores remain frozen. Any v2 rescoring of those responses must be
reported as a post-development parser audit, never substituted silently for v1.
