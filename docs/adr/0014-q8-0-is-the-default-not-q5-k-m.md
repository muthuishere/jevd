# 0014 — Q8_0 is the default quantisation, not Q5_K_M

Status: **accepted** (openjev-core, 2026-09-20). **Reverses design 01 §5's default.**
Closes design risk **R1**.

## Context

Design 01 §5 shipped Q5_K_M as the default and said so before anything had been measured,
with R1 as the open question: *"quantisation moves the label"*. R1's own stated gate was
label agreement — *"anything below ~98 % agreement demotes the default to Q8_0"*.

Measured, 35-pair golden fixture, Metal, against the bf16 reference:

| trunk | size | hidden relL2 | max abs prob deviation | label agreement |
|---|---|---|---|---|
| F16     | 8.4 GB | 3.7e-3 | 0.0029 | 35/35 |
| **Q8_0**    | **4.5 GB** | **6.0e-3** | **0.0097** | **35/35** |
| Q5_K_M  | 3.1 GB | 1.9e-2 | 0.0218 | 35/35 |
| Q4_K_M  | 2.7 GB | 2.8e-2 | 0.0253 | 35/35 |

The reference disagrees with *itself* by 0.0021 (bf16 vs fp32) on the same pairs.

## Decision

**Default Q8_0.** F16, Q5_K_M and Q4_K_M are published alongside with these numbers
attached, and `dtype` stays configuration.

R1's agreement gate does not discriminate: every level scores 35/35. So the gate is the
probability deviation instead, and the reason is that agreement is the wrong statistic for
how this model is actually consumed. `grade` returns `P(entailment)` against a threshold
that defaults to 0.5, and the herdr plugin's asymmetric-confidence policy compares
probabilities to bars. Those callers feel 0.02 of drift directly, whatever the argmax did.

Q8_0 sits at 4.6x the reference's own bf16-vs-fp32 spread. Q5_K_M sits at 10x it, and at
that point the number is no longer characterisable as rounding.

## Consequences

1.4 GB more to download than Q5_K_M, and ~5 GB resident. On the 16 GB machine design 01
worried about that is still comfortable; a caller who would rather have the gigabyte back
can set `dtype` and now has the number they are trading away.

**What this does not establish.** 35 pairs with zero label flips bounds the true flip rate
only loosely — the one-sided 95 % bound is roughly 8 %, which is not a small number for a
classifier. "Q4_K_M never flips a label" is *not* proven here and is not claimed. What is
proven is a monotone ordering of trunk error, and that the ordering matches intuition. A
real agreement claim needs a few hundred pairs from SNLI/ANLI dev, which is R1's original
experiment and is still worth running.
