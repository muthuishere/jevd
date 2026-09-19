# 0003 — Shape is checked before confidence, and from the text alone

**Status:** accepted

## Context

The short circuit's failure mode is answering a generative task with a probability. The
tempting design is one gate: ask the model how confident it is that it can answer, and
short-circuit above a threshold. That design cannot work, because a model asked "can you
answer 'rewrite the parser'?" will often be very confident — about the wrong question.

## Decision

Two gates, ordered, and the ordering is the design:

1. **SHAPE** — decided in Go, from the text, with no model involved. A closed set:
   `boolean`, `pick_one`, `grade`, or `none`.
2. **CONFIDENCE** — decided by the model, against its own high bar.

**A task that is not decision-shaped is never short-circuited no matter how confident the
model is.** Confidence never gets the chance to rescue the wrong question.

A generative verb anywhere in the text vetoes unconditionally, before any shape is
considered. A false veto costs one agent call — which is what would have happened anyway.
A false pass costs the user their actual task. The asymmetry is total, so the veto is
broad.

## The one narrowing

A verb immediately preceded by a determiner or preposition is being used as a noun, and
the marking carries across a compound: "a **port change**", "the **test** suite". Without
this, "Does the state file survive a port change?" is vetoed — and so is roughly half of
every genuine question anyone asks about a codebase, since `change`, `test`, `review`,
`plan`, `design` and `build` are nouns at least as often as verbs.

## Consequences

- Shape detection is pure, fast, and exhaustively testable with no model. The table in
  `shape_test.go` is deliberately heavier on what must NOT be answered than on what must.
- New shapes must be new entailment framings, not new hopes. There is no shape for
  "summarise", because entailment cannot express one.
