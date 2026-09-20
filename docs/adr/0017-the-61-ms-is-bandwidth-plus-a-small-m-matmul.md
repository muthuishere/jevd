# 0017 — the 61 ms is bandwidth plus a small-M matmul, and both of those are the hardware

Status: **accepted**, measured 2026-09-20. **Confirms ADR 0015's conclusion and falsifies
its arithmetic — and one of the things it falsifies is worth 2.6x.**

## The failure this is organised against

ADR 0015 closed with "openjev runs at the speed a 4B prefills at on this machine," and
justified it with: *256 GFLOP in 61 ms is ~4 TFLOP/s, which is the right order for this
GPU.* Nobody had measured this GPU. "We are at peak" asserted against a number that was
never taken is how optimisation stops one measurement too early — and it is exactly as
unfalsifiable as the guess it replaced.

So the denominator got measured first. `scripts/metal-peak.swift` / `task bench:hw`.

## What this GPU actually does

Apple M5 Pro, 20-core GPU, 48 GB unified. MPS fp16 matmul on real buffers, best of 5:

| shape | TFLOP/s |
| --- | --- |
| M=4096 K=4096 N=4096 | **31.0 – 31.4** |
| M=512 K=2560 N=9728 | 23.5 – 31.5 |
| M=128 K=2560 N=9728 | 21.4 – 26.1 |
| **M=32 K=2560 N=9728** | **7.4 – 7.7** |
| **M=32 K=2560 N=2560** | **7.1 – 9.6** |
| blit-copy bandwidth | **267 – 271 GB/s** |

**"4 TFLOP/s is the right order for this GPU" is false — the GPU does 31.** What is true,
and is the whole story, is that **31 TFLOP/s is not available at M=32**. An NLI pair is 32
tokens, so every matmul in the model has M=32, and at that shape Apple's own kernels — the
fairest possible upper bound on what ggml could reach — deliver 7–9 TFLOP/s. The ceiling
is set by the shape of the problem, not by the quality of the kernel.

## Where the 61 ms goes

llama.cpp b9620 (`llama-bench`), Q8_0, Metal, `-ngl 99`, warm, `-r 5`:

| prompt | t/s | ms | | prompt | t/s | ms |
| --- | --- | --- | --- | --- | --- | --- |
| pp1 | 50.4 | **19.8** | | pp32 | 566 | 56.5 |
| pp4 | 181 | 22.0 | | pp48 | 815 | 58.9 |
| pp8 | 297 | 26.9 | | pp64 | 973 | 65.8 |
| pp12 | 219 | 54.7 | | pp128 | 1459 | 87.8 |
| pp16 | 291 | 55.1 | | pp256 | 1689 | 151.6 |
| pp20 | 363 | 55.1 | | pp512 | 1734 | 295.3 |
| pp24 | 433 | 55.4 | | pp1024 | 1733 | 591.0 |

The same shape through `openjev-core`'s own harness (`tests/bench.rs`,
`report_where_the_milliseconds_go`, p50 of 30 warm forwards):

| tokens | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p50 ms | **22.2** | 22.8 | 24.2 | 30.2 | 59.2 | **60.0** | 65.5 | 80.8 | 128.9 |

Three facts fall out, and together they close the budget.

**1. A one-token forward costs 20 ms, and that is the floor.** 4.16 GiB of weights must
cross the bus once no matter how few tokens you ask about: 4.47 GB / 19.8 ms = **226 GB/s,
85% of the 267 GB/s this machine measures.** A 1-token forward through a 4B model is
bandwidth-saturated. **You cannot answer with a 4B model without reading a 4B model**, and
on this hardware reading it takes 17–20 ms. Nothing in software moves that.

**2. pp32 is 94% accounted for by bandwidth plus the M=32 matmul rate.**

```
weight streaming   4.47 GB / 267 GB/s                    = 16.7 ms
arithmetic         2 x 4.21e9 x 32 FLOP / 7.4 TFLOP/s    = 36.4 ms
                                                   total = 53.1 ms   (measured 56.5)
```

There is no pool of waste. The residual is ~6% and it is scheduling.

**3. Tokens 12 through 48 are free.** pp12 through pp48 all cost 55–59 ms; the in-process
table shows 16 and 32 tokens costing the same to within noise, with a marginal cost of
**-0.2 ms per token** across that span. The step sits between 8 and 12 tokens. A shorter
hypothesis buys nothing once the pair clears ~12 tokens, and real NLI pairs — the golden
fixture's median is 20 — sit entirely inside the flat region. Above 48 tokens the marginal
cost climbs to its asymptote of 0.58 ms/token.

## What openjev adds on top: 3.8 ms, and none of it is reclaimable

60.3 ms in our harness against llama.cpp's own 56.5 for the identical shape. That 3.8 ms is
batch construction, the memory clear, the embedding copy out of the context, and the head.
**The head is `[3, 2560] x [2560]` — 15 kFLOP, four nanoseconds of arithmetic.** It was
worth confirming it is not being done stupidly; it is not, and it could not matter if it
were.

## The two knobs that were supposed to help, and did not

**`n_ubatch`.** The code sizes `n_batch`/`n_ubatch` to the whole context, on the reasoning
that a coalesced group should be one graph. Swept 256 / 512 / 1024 / 4096 / 16384 with
`max_seqs=8`: flat within noise at both batch 1 and batch 8. `llama-bench` shows the same
at pp32 (565 / 565 / 554 t/s for ub 512 / 2048 / 8192) — the largest setting is *marginally
worse*, by 2%, which is not worth a config key. **Left alone.**

**Flash attention.** pp32 with `-fa 0` and `-fa 1`: 564.7 vs 567.0 t/s. Of 32 layers only 8
are `full_attention`, and they attend over 32 tokens, where the attention matrix is 32x32.
There is nothing there to accelerate. **Left alone.**

**Pooling.** `LlamaPoolingType::None` with a manual last-token read instead of `Last`: no
difference outside thermal drift. **Left alone.**

Everything in this section was measured and reverted. The only permanent change this work
produced in the engine is none, which is the correct outcome when the profile indicts the
hardware.

## The one thing that is not the hardware: batching is 2.6x off, and ADR 0015 is why nobody looked

ADR 0015 measured eight 32-token sequences in one decode at 465 ms — 8x the single — and
concluded the GPU was never waiting. **`llama-batched-bench` on b9620, same model, same
shape, disagrees:**

| sequences | T_pp | ms per pair |
| --- | --- | --- |
| 1 | 0.056 s | 56.0 |
| 4 | 0.142 s | 35.5 |
| 8 | **0.207 s** | **25.9** |
| 16 | **0.321 s** | **20.1** |

Stock llama.cpp coalesces eight recurrent sequences into one ubatch and gets 2.2x; sixteen
gets 2.8x. **Our path does not**: re-measured today at `max_seqs=8`, 545–565 ms, i.e. 68 ms
per pair, unchanged from ADR 0015. The single-sequence path is at the hardware's rate; the
batched path is 2.6x off it.

That is a falsification of 0015's *mechanism*, not its headline. "The hardware was never
waiting" is true at batch 1 and false at batch 8 — and 0015's explanation, being a
satisfying one, is precisely what stopped the next measurement from being taken. The
in-process forward at 256 tokens costs 129 ms against 60 ms at 32, which says the same
thing from the other side: **the GPU will do four pairs' worth of tokens for twice one
pair's price.** It is idle at M=32 and it says so.

**Cause not identified, and not guessed at.** `n_ubatch` is ruled out (swept, flat).
Pooling type is ruled out (swept, flat). `clear_kv_cache()` per group is the remaining
suspect and **could not be A/B'd**, because removing it makes the next group's `decode`
fail — the design depends on it. Naming it as the suspect without the measurement would be
the same mistake this ADR exists to correct.

**This is throughput, not latency**, so it does not touch the 61 ms a single `predict`
pays. It is worth up to 2.8x on `rerank`, which is the call that actually scores N options,
and on any batched grading. It is the one open engineering win on this trunk.

## Decision

**No change to the engine.** The single-pair path is within ~15% of what the kernels can
do, and what remains is bandwidth (17 ms, irreducible) and an M=32 matmul (36 ms, at
Apple's own ceiling for that shape).

**The realistic best for one 32-token pair on this trunk, on this machine, is ~50 ms.**
Against 61 ms today, that is an 18% prize for work that would have to be done inside
ggml-metal. It is not worth taking, and it is not remotely the target.

**Say the number plainly: 61 ms is the floor for 4B on this hardware, to within 20%.
Milliseconds require a smaller model.** See ADR 0018.

Two things were fixed, both in the measurement apparatus rather than the engine:

* `task bench` now passes `--test-threads=1`. Two benchmarks were sharing one GPU and
  measuring each other — 30% high, with p95 at 3x p50. A benchmark that races itself is
  worse than no benchmark.
* `scripts/metal-peak.swift` / `task bench:hw` exists so the denominator is a command
  anyone can re-run, not a paragraph in an ADR.

## Consequences

* ADR 0015's decision stands; its arithmetic does not. Batching stays, defaults to 1, and
  now has a named, measured 2.6x gap against stock llama.cpp behind that default.
* Quantisation's flatness (ADR 0014) is now explained rather than merely observed. Q4_K_M
  cuts the 17 ms bandwidth term to ~10 ms and adds dequantisation to the 36 ms arithmetic
  term, which is the larger one. A small win against a small loss is a wash, and the
  measurement said wash.
* "Fewer tokens" is a weaker lever than ADR 0015 implied. Below ~12 tokens it is worth 2x;
  from 12 to 48 it is worth nothing at all.
