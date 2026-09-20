# 0019 — `usage` tokens are counted by the encoder that ran, or reported as zero

Status: **accepted** (openjev-core + openjev-cli, 2026-09-20). **Organised against the
failure where a number that looks like a measurement is an estimate, and nothing
downstream can tell.**

## Context

`usage.tokens` was `0` on every response from `/v1/predict`, `/v1/rerank` and
`/v1/grade` — a placeholder STATUS listed as a known gap. `/v1/systemone` has to report
`usage.input_tokens`, and clients budget against it: `fast-jev-compaction` fits a `state`
to a token budget and splits its questions by what is left.

Two tempting wrong answers. Estimating from character counts produces a plausible number
that is wrong by whatever the tokenizer does with the caller's text, and a consumer
budgeting against it cannot see the error. Threading a count out of `predict` puts it on a
path where the backend could report one thing and the encoder another.

## Decision

`Session::count_pair_tokens(pairs)` in `openjev-core` renders and encodes each pair and
returns its token count — **the same encoder, the same template, the same trim** as the
forward pass. The worker calls it once per coalesced batch and slices each job its own
share, which `Outcome.tokens` carries back. Every pair endpoint now reports a real count,
not just the new one.

It is a second tokenisation pass. That costs microseconds against a ~50 ms forward, and
buying correctness with it is not a trade worth thinking about.

Two numbers stay zero, and are documented as constants rather than left looking incidental:

* **`output_tokens: 0`** — a cross-encoder emits no tokens. It emits one distribution over
  labels per pair. There is no honest number, so there is no number.
* **`cost: 0`** — this runs on your machine. Inventing a rate to fill the field would be
  making up a price for electricity.

`/v1/latents` still reports 0: latents do not go through the pair encoder, and a count
there would be a different measurement wearing the same field name.

## Consequences

`usage.tokens` changing from a constant 0 to a real number is a visible change for anyone
who was reading it. It was documented as a gap, so nothing should have been.

`Outcome` gained a field, which touches the worker — the one file this change shares with
work in flight on the backends. It is additive and the batching logic is untouched.
