# 0015 — batching buys nothing here, because an NLI prefill already saturates the GPU

Status: **accepted**, measured 2026-09-20. **A negative result, and it contradicts the
reasoning that motivated the work.**

## The expectation

The baseline throughput table is dead flat across batch size — 16.3 pairs/s at batch 1,
16.2 at 16, 15.7 at 32 — because the backend declared `!Caps::BATCH` and looped. The
obvious reading, and the one taken: a 32-token forward through a 4B model leaves the GPU
mostly idle, the cost is per-call overhead, and coalescing sequences into one decode
should reclaim 3-4x.

So multi-sequence decode was implemented: `n_seq_max > 1`, one `llama_batch` carrying N
sequences under N seq_ids, one `decode`, `embeddings_seq_ith` per sequence.

## The measurement

Q8_0, Metal, n_ctx 16384 total, warm:

| tokens | max_seqs=1 | max_seqs=8 | max_seqs=16 |
|---|---|---|---|
| 32  | 16.3 | 16.5 | 15.7 |
| 128 | 11.9 | 11.5 | — |
| 512 | 3.7  | 3.6  | — |

Eight sequences of 32 tokens in one decode take 465 ms. One takes 61 ms. That is 8x — the
decode cost scales linearly with the number of sequences, and coalescing reclaims nothing.

## Why

The premise was wrong. This is a **prefill**, not a decode. Generating one token from a 4B
model is memory-bound and leaves the arithmetic units idle, which is why batching is such a
large win for generation. Prefilling 32 tokens is a matmul of `[32, 2560]` against every
weight matrix in the model, and it is already compute-bound. There is no idle capacity for
a second sequence to occupy, so a second sequence costs a second sequence's worth of work.

Roughly: 4B params x 2 FLOP x 32 tokens ~ 256 GFLOP for one pair. At 61 ms that is ~4
TFLOP/s sustained, which is the right order for this GPU on ggml's Metal kernels. Eight
pairs is 2 TFLOP and takes 465 ms — the same rate. The hardware was never waiting.

## Decision

**The batching code stays, `Caps::BATCH` stays, and `server.max_seqs` stays at 1.**

Correct-but-pointless code that carries a silent-wrong-answer risk is a bad trade, so the
keep/delete question deserves an argument rather than a shrug.

**The risk is real and specific.** ADR 0001 measured the recurrent-state leak with exactly
one sequence per `llama_decode`, and named batched sequences as the single condition that
would reopen it. Several sequences sharing one context is exactly that condition. A leak
there is a confident wrong label, not a crash.

**It is now measured, and asserted.** Target second in a shared decode behind distractors
of 8 / 64 / 256 / 1024 tokens: `relL2 0.000e0`, every length. Target first, distractor
after: `0.000e0`. Four identical copies in one call: `0.000e0` against each other and
against the standalone result. Plus a mixed-length batch of real pairs through the whole
golden suite, also `0.0000e0`. Two tests now carry this —
`recurrent_state_does_not_leak_within_one_batched_decode` for state leakage and
`a_mixed_length_batch_agrees_with_the_same_pairs_alone` for padding and pooling — so the
property is coverage, not folklore.

**So it stays**, because with those assertions in place the residual risk is a regression
that two tests would catch, and what is bought is a proven-correct multi-sequence path that
a faster runtime, a smaller trunk or a bigger GPU could switch on with one config key. The
cost of keeping it is those two tests, which had to exist anyway — R2 does not stop
mattering just because batching is off.

**It defaults to 1** because it costs KV budget and returns nothing measurable today.

**It should be deleted if** the tests ever have to be weakened to keep it green, or if a
year passes with no workload where it wins. Either would mean the price stopped being paid
and the code became decoration.

## Consequences

The throughput number in `STATUS.md` does not move. The honest summary is that openjev
runs at the speed a 4B model prefills at on this machine, and the levers that remain are
about doing **less work**, not about scheduling the same work better:

* **fewer bits** — Q5_K_M and Q4_K_M reduce both bandwidth and dequantisation cost, at a
  measured accuracy price (ADR 0014). This is the next thing to measure.
* **fewer parameters** — a 0.8B trunk measured 57 pairs/s. A 4x speedup is available for
  whatever accuracy the 0.8B checkpoint actually has, and that is a product decision.
* **fewer tokens** — cost is roughly 50 ms fixed plus 0.45 ms/token. Short hypotheses are
  cheap; the fixed term dominates at NLI lengths and it is the model, not us.

What this does **not** license is the opposite conclusion. Batching is worthless *for this
model on this runtime on this hardware*. On a machine where the GPU is genuinely underfed —
a larger GPU, a smaller trunk, or a llama.cpp that fuses per-sequence recurrent work — the
same code may well pay. It is one flag away, and the golden suite already proves it safe.
