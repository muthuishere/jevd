# 0004 — The answer bar is separate from, and above, the routing bars

**Status:** accepted

## Context

Three confidence numbers now exist: upgrade (0.30), downgrade (0.60), and the bar for
answering a task outright. The cheap move is to reuse the downgrade bar — answering
locally is, after all, the cheapest possible option, so it looks like the extreme of the
same axis.

## Decision

`answer.min_confidence` is its own key, defaulting to **0.85**, and config **refuses** a
value below `min_downgrade_confidence`.

They are not the same axis. The routing bars trade money against capability, and a wrong
call is recoverable inside the same turn — a task on too small a model produces a bad
answer that a human immediately sees. The answer bar decides whether a human gets a
machine's two-word reply **instead of their agent's work**: a different kind of mistake,
with a different kind of cost, and one where the user may not notice they were shortchanged.

If the answer bar were the lower of the two, the plugin would be more willing to replace
your task than to make it cheaper. That is exactly backwards, so it is a validation error,
not a style guideline.

## Why 0.85 specifically

After normalisation (see 0005), 0.85 means:

- a two-way answer must be roughly **6:1** — no genuine coin flip reaches that;
- a four-way pick must beat a uniform 0.25 by a **factor of three**.

Both are comfortably outside what an unsure cross-encoder produces, and comfortably
inside what a clear-cut question produces.

## Consequences

The number will be wrong for somebody, so it is config, and the declined near-misses are
journalled: a bar nobody can see the misses for is a bar nobody can tune. Shadow mode
(ADR 0006) exists to collect exactly that evidence before the bar is allowed to act.
