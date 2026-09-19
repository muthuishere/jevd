# 0008 — `truncate: "tail"` is refused, not approximated

Status: **accepted** (openjev-cli, 2026-09-19). **Divergence from design 02 §3.2 that
clients must know about.**

## Context

Design 02 defines `truncate: "error" | "tail"`, default `error`. Core owns tokenisation
(`tokenize::Encoder`) and exposes no truncating encode: there is no public way to ask for
"the last N tokens of this pair under the model's template".

## Decision

`truncate: "error"` is the default and works. `truncate: "tail"` returns
**422 `unprocessable`** naming the limitation.

Truncating on *characters* at the CLI boundary would be a lie with a plausible shape: the
template wraps the text, the tokenizer is not a character counter, and the failure mode
of getting it slightly wrong is a confident, wrong, unfalsifiable label — the exact
failure `truncate: "error"` is the default to prevent.

## Consequences

Clients must handle 422 for over-long input and split it themselves. When core grows a
truncating encode, this becomes an additive change: the same field, the same value, a
200 instead of a 422. No client change is needed for that transition.
