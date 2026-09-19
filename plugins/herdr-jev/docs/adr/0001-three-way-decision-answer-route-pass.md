# 0001 — The decision is three-way: answer, route, pass through

**Status:** accepted

## Context

The obvious shape for this plugin is a router: classify a task, pick a model. But a
cross-encoder that can rank options against a question can also *answer* a question —
and a large share of what gets typed at a coding agent is a question, not a task. Asking
a 200B model, over the network, whether the build is green is an absurd trade when a 4B
model on loopback answers it in 40ms.

## Decision

Three stages, in a fixed order:

1. **Answer locally** — decision-shaped AND above the answer bar.
2. **Route** — pick the tier and effort, start the agent with them.
3. **Pass through** — anything else, untouched.

Stage 3 is not an error path. It is the **default that stages 1 and 2 must earn their
way out of**. Every branch that cannot prove what it wants falls into it.

## Consequences

- A broken, slow, absent or confused openjev is indistinguishable from not having
  installed the plugin. That is the intended failure mode.
- The three stages are one function (`dispatch.Decide`) that returns a plan and performs
  no side effects, so `--dry-run` is the same code path minus one call rather than a
  parallel implementation that can drift.
- "How much did this save" becomes countable, because every dispatch records which stage
  it ended in.
