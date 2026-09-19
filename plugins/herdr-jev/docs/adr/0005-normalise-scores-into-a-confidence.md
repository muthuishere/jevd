# 0005 — Normalise scores; a raw entailment score is not a confidence

**Status:** accepted

## Context

`/v1/rerank` returns an independent entailment probability per option. It is tempting to
read the winner's raw score as the confidence — it is already a 0..1 number with the
right shape.

It is the wrong number. Two options at 0.90 each are **not** a confident answer; they are
a tie in which the model finds both equally plausible. Reading the raw score calls that
0.90 and answers it.

## Decision

Divide by the sum of the positive scores. That turns *"how well does each option fit"*
into *"how much better is the best"*, which is the question every bar in this plugin is
actually asking.

Three consequences follow directly:

- **A boolean asks both the claim and its negation.** P(yes)=0.55 read alone looks like a
  weak yes and is indistinguishable from a coin flip. Asked as a pair, 0.55/0.52
  normalises to 0.51 and is refused.
- **Effort takes the EXPECTED rung, not the argmax.** The rubric is ordered, so a task
  split evenly between rungs 1 and 2 genuinely wants 1.5; rounding to whichever won by
  0.01 throws away the ordering the rubric is made of. The confidence alongside it is
  still the winner's share, because that is what the bars read.
- **`risky` is NOT normalised.** There is no set to normalise over: the question is "how
  likely is this true", not "which of these". The raw entailment probability is the
  answer, and it is exactly the `noul` the reference policy reads.

## Consequences

An all-zero or all-equal response degrades to "no signal" rather than to a confident
arbitrary pick, and every bar in the system inherits that property for free.
