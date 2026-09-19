# 0006 — The short circuit ships off, with a shadow mode

**Status:** accepted

## Context

Answering a task instead of invoking the user's agent is the largest behavioural claim
this plugin makes. A plugin that starts doing it the moment it is installed has made that
claim on the user's behalf, without being asked, and the first time it is wrong it will
be wrong invisibly — the user gets a plausible two-word answer and no transcript.

## Decision

- `answer.enabled = false` by default. Model/effort routing is on; answering is opt-in.
- `answer.shadow` logs **"WOULD have answered locally (0.94)"** and dispatches to the
  agent anyway.
- Shadow **wins** over enabled when both are set. Shadow exists precisely so the short
  circuit can be watched without acting, so a config with both must watch.
- `savings` counts the shadow would-haves separately, so the question "what am I leaving
  on the table" has a number.

## Consequences

The intended adoption path is: install → shadow for a week → read `herdr-jev why` →
enable with evidence. That is slower than shipping it on, and it is the difference
between a bar the owner believes and a bar the owner turns off after one bad answer.

`doctor` reports the off state as a FAILING check with the exact fix, because "it never
answers anything" is otherwise indistinguishable from a broken install — and by design,
nothing else in the system will complain.
