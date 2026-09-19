# 0002 — Port the asymmetric-confidence policy rather than invent one

**Status:** accepted
**Source:** `jev-model-router` (claude-code-templates), `hooks/policy.ts` + `tests/policy.test.ts`

## Context

The policy question — "how sure must a classifier be before it changes what a task runs
on" — looks like it has one answer and actually has two. The reference implementation had
already found that, plus four edge cases that are individually easy to get wrong and
collectively the difference between a router that works and one that quietly costs money.

## Decision

Port `policy.ts` faithfully to Go, including its rationale, and port its test file with
it. Keep every rule:

- **Asymmetric bars.** Spending more: 0.30. Spending less: 0.60. The two mistakes do not
  cost the same, so they do not clear the same bar.
- **`risky > 0.7` forces the deep tier past both bars**, and is a floor-raise only.
- **No confidence reported → may only move up.**
- **Unknown model id / numeric effort → no knowable direction**, so the gentler bar and
  no change respectively.
- **It never blocks.**

## What we changed, and why

| change | reason |
|---|---|
| backend is `openjev` on loopback | the reference ships prompt text to a hosted API. Ours does not leave the machine. |
| tier/effort/risky as NLI hypotheses | a cross-encoder has no typed-question API; entailment probability is the calibrated confidence the policy already wanted |
| tiers are per **agent kind** | claude/codex/gemini name models differently; a hardcoded haiku/sonnet/opus routes exactly one agent |
| both switches default **on** | the reference keeps main-model routing off because switching mid-session invalidates the prompt cache. We choose at process START, so there is no cache to invalidate |
| dropped `ALIAS_IDS` and `pendingDecisions` | both are Claude-Code-engine specifics: we have no `turn.step` to resolve an alias for, and no queue of prompts racing a turn |

## Consequences

The ported tests are the highest-value tests in the repo: every case in them is a bug
someone already found. They pass unmodified in intent, and the two Gateway-wire-shape
tests are replaced by our own normalisation tests, which test the same property (a
confidence must be derivable, and absent when it is not).
