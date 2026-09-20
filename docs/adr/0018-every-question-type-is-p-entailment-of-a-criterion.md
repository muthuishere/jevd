# 0018 — every question type is `P(entailment)` of a criterion, resolved by name

Status: **accepted** (openjev-cli, 2026-09-20). **Organised against two failures: a
hardcoded label index that returns contradiction dressed as entailment, and four question
types drifting into four different calibrations.**

## Context

The engine's primitive is `premise + hypothesis -> P(contradiction), P(entailment),
P(neutral)`. System One's primitive is a typed question about a `state`. Something has to
map one onto the other, and there are more plausible mappings than there are right ones.

The forcing constraint is that **the label index is not fixed**. `entailment_label` comes
from the registry, and `probs[1]` is only entailment for the checkpoints where it happens
to be. Getting this wrong does not fail: it ranks everything by `P(contradiction)` and
returns confident, exactly-inverted answers. It has already been shipped twice in this
codebase — once in `Session::rerank` and once in the server's duplicate of it (ADR 0012).

## Decision

**The `state` is the premise, always.** A question becomes one hypothesis per criterion,
built by one template for every type:

```
hypothesis = trim(instructions) + " " + trim(criterion)
```

Either part may be absent. One template, not four, because two templates are two
calibrations: a number from a `noul` and a number from a `choice` should mean the same
thing about the same state. The instructions frame the question; the criterion is the
claim being tested.

Then:

* **`choice`** — score every criterion, L1-normalise `P(entailment)` over the keys,
  argmax is `choice` and its own probability is `confidence`. L1, not softmax:
  `P(entailment)` is already a probability per option, and a softmax over probabilities
  applies a temperature nobody chose and flattens a decisive answer. An all-zero vector
  goes uniform, which is the honest statement that nothing was entailed — not a winner
  picked by floating-point noise.
* **`score`** — the same distribution over the rubric's indices, and the score is its
  expected value, in `[0, levels-1]`. Not the argmax: the client rounds what it gets, so a
  fractional value is the shape it already handles, and "between 1 and 2" is genuinely a
  different answer from "solidly 2".
* **`noul` / `boolean`** — **two paths, deliberately.** With only a `true` criterion
  (or none at all, which is how `jev-model-router` asks), there is nothing to contest, so
  the answer is the raw `P(entailment)` of that one hypothesis. With both cases described,
  the question really is a two-way contest and the answer is `t / (t + f)`, which cancels
  the model's global entailment bias. The difference is not cosmetic: criteria that are
  mutually contradictory — both sides entailed — report 0.5 under the contest and would
  have reported ~1.0 and sounded certain under the raw score. That case is pinned by a
  test rather than argued about.

`P(entailment)` is resolved **by name**, once, in `entailment_index()`. The server's
`rerank` handler now calls the same function instead of keeping its own copy of the
lookup, so the duplication ADR 0012 flagged is one place smaller.

## Consequences

Every number this endpoint returns is `P(entailment)` of a rendered hypothesis, so the
whole server has one notion of "how true" and the golden fixture that pins `predict` pins
this too.

Two behaviours are recorded rather than asserted as desirable, because they are properties
of the mapping and should be visible when they change:

| case | what happens |
| --- | --- |
| the state is irrelevant to the question | the distribution goes uniform and `confidence` says so — no confident winner |
| the criteria are mutually contradictory | a two-sided `noul` reports ~0.5; a `choice` splits its mass |

The hypothesis template is now a calibration surface. Changing the separator, or adding a
"The correct answer is:" wrapper of the kind the reference uses for multiple choice, moves
every number this endpoint returns. It is one line, in one place, and it is the first
thing to look at if the answers ever look systematically off.
