# 0012 — `rerank` scores the question as premise, the option as hypothesis

Status: **accepted** (openjev-core + openjev-cli, 2026-09-20). **Fixes a silent wrong
answer that every weightless test passed.**

## Context

`Session::rerank` built its pairs as `(option, question)` — option as premise, question
as hypothesis. `server.rs` duplicated the same order independently.

The reference is the other way round:

```python
def rerank(self, question, options, hyp_fmt="The correct answer is: {}") -> int:
    p = self.predict([(question, hyp_fmt.format(o)) for o in options])
    return int(p[:, ENT].argmax())
```

Entailment is directional. *"Does 'carbon dioxide' entail 'Which gas do plants absorb?'"*
is a different question from *"Does 'Which gas do plants absorb?' entail 'carbon
dioxide'?"*, and the two produce different rankings.

Nothing caught it. The unit test `rerank_sorts_by_p_entailment_and_keeps_the_original_index`
passed under both orders, because the fake backend selected its canned hidden state from
`tokens[0]` — the premise word — and under the wrong order that happened to be the option.
The test was reading the bug as its fixture.

## Decision

Both core and the server score `(question, option)`. The fake backend now keys on the
**last** token, so the two orders are distinguishable, and a dedicated test asserts what
actually reached the backend rather than restating the encoder.

The reference's `hyp_fmt = "The correct answer is: {}"` is **not** adopted. Our `rerank`
is a general ranking verb, not a multiple-choice harness, and the herdr plugin passes
options that are already statements. Measured on real weights, the direction is what moves
the ranking; the wrapper is a prompt choice a caller can make for itself.

## Consequences

`rerank` returns a different — correct — order than it did. Any calibration done against
the old order is void. The plugin's tier and effort classifiers are the main consumers and
were ranking by a reversed entailment the whole time.

The server keeping its own copy of the pair order remains a second place to be wrong, kept
only so batching can coalesce reranks with predicts. That duplication is now the thing
this ADR is organised against, and the golden fixture is what would catch it next time.
