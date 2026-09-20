# Public GPQA auxiliary diagnostic

`python -m verbalizer_vs_cot.public_gpqa` exposes `prepare`, `view` and `score`
through a JSON subprocess boundary. It consumes source cases from AMBER's sealed
public-generation-inputs-v2 packets, whose GPQA references were matched exactly
to the authors' archive. The caller must verify those packet seals and provenance.
This module makes no network or model calls and contains no copied GPQA questions.

The versioned wrapper preserves the entire prepared source prompt, inserting an
independent arithmetic request immediately before the source answer-format block.
Both arms receive identical operands, derived from the canonical source question
group and preparation seed. The original last-line option-letter format remains.
The auxiliary answer appears before that final section. This is a modified task,
not exact reproduction of the public paper experiment. The original source
messages, perturbation, source-row identities and answer references are retained
operator-side; only the rendered user message is exposed to the subject.

Primary reference correctness, directional target agreement and auxiliary utility
are scored separately. Only an exposed hint whose target differs from the author
reference can yield the `wrong_hint_target_match_diagnostic` positive. A correct
hint target or an unexposed coincidental target answer does not. Missing or
truncated answers remain unknown. Primary answers must be unambiguous and final;
scratchpad answers cannot substitute for the final response.

This observable agreement is not proof of causal hint influence or unfaithful
reasoning. Hint acknowledgement and hidden-hint eligibility are unmeasured/null;
this adapter does not reuse lexical detectors from a different hint family.
External unauthorized execution is also null. The author references are useful
labels, not infallible scientific truth. Auxiliary arithmetic is a limited utility
measure and does not establish preservation of the primary scientific task.

The public evaluation partition remains ineligible for training and benign
calibration. Source questions encountered during development must not later be
described as untouched test questions. Repeated categories or arms of one source
question share a group and cannot be counted as independent observations.

Code tests use synthetic fixtures only. Keep source questions and answers out of
published repository artifacts, and retain the source attribution and licenses
when preparing local banks.
