# Status — 2026-09-20

The previous status said *"no real forward pass has ever run"*. That is no longer true.
openjev runs the real checkpoint, and its answers match the reference implementation.

## It works

`AlexWortega/openjev @ 4b5f9a67` / `qwen3.5-4b-nli-v2`, converted to GGUF by us, served by
`openjev-core` on Metal, graded against 35 pairs scored by `transformers` +
`Qwen3_5ForSequenceClassification`.

| | |
| --- | --- |
| label agreement | **35 / 35** at every quantisation level |
| token ids vs reference | **exact on all 35 pairs** — template, trim and BOS all match |
| max abs probability deviation (Q8_0) | **0.0097** |
| the reference disagreeing with itself (bf16 vs fp32) | 0.0021 |
| pooled hidden state, relL2 vs reference (Q8_0) | 5.8e-3 – 6.3e-3 |
| mixed-length batch vs the same pairs alone | **0.0000e0** |
| recurrent-state leak after a 1024-token distractor | **0.000e0** (bit-identical, on Metal) |
| same, *within one batched decode*, both orders | **0.000e0** at every distractor length |
| all-zero hidden states (ADR 0003 guard) | none, on any pair |

The whole product path is exercised, not just the library: `openjev predict/rerank/grade`
and `POST /v1/predict|rerank|grade` on a running `openjev serve` all return the fixture's
numbers. `rerank` puts *carbon dioxide* first for the photosynthesis question; `grade`
scores a faithful answer 0.976 entailment and a wrong one 0.982 contradiction.

### System One, served (2026-09-20)

`POST /v1/systemone` serves the TypeSafe System One shape alongside
`/v1/predict|rerank|grade`, which are unchanged. Four question types — `noul`, `choice`,
`score`, and `boolean` (the Vercel AI Gateway's name for a `noul`). Clients written for
TypeSafe's API point at it by changing a base URL.

The owner's example runs end to end in `task check`: `tests/systemone_e2e.rs` builds a
real `Session` — real registry entry, template, tokenizer and linear head — over a stub
backend with chosen hidden states, and asserts the exact response shape through the real
router, worker and answer arithmetic. `tests/systemone_live.rs` (`task systemone`) runs the
identical request against a running server with real weights; **it has not been run** —
there is no converted checkpoint on this machine — so the wire shape is proven and the
checkpoint's judgement on that example is not.

Three deliberate divergences, all documented in `docs/adr/0017`: `provider` reports
`openjev` and never `TypeSafe`; errors use this server's one envelope rather than
TypeSafe's; `usage.output_tokens` and `cost` are honest zeros.

**`usage.tokens` is no longer always 0.** The gap this file used to list is closed:
`Session::count_pair_tokens` counts with the encoder that ran, and every pair endpoint
reports a real number (ADR 0019). `/v1/latents` still reports 0 and says why.

The contract lives at `tests/golden/nli.json` and is enforced by
`task golden OPENJEV_ARTEFACTS=<dir> OPENJEV_TOKENIZER=<path>`.

## Quantisation — measured, not assumed

Design 01 defaulted to Q5_K_M before anything had been measured. **The default is now
Q8_0** (ADR 0014).

| trunk | size | hidden relL2 | max abs prob deviation | labels |
| --- | --- | --- | --- | --- |
| F16 | 8.4 GB | 3.7e-3 | 0.0029 | 35/35 |
| **Q8_0** | **4.5 GB** | **6.0e-3** | **0.0097** | **35/35** |
| Q5_K_M | 3.1 GB | 1.9e-2 | 0.0218 | 35/35 |
| Q4_K_M | 2.7 GB | 2.8e-2 | 0.0253 | 35/35 |

Label agreement does not discriminate — every level scores 35/35 — so the gate is the
probability deviation, because that is what `grade`'s threshold and the plugin's
confidence bars actually consume. Q8_0 sits at 4.6x the reference's own spread; Q5_K_M at
10x, which is past calling it rounding.

**35 pairs with zero flips bounds the real flip rate only loosely** — the one-sided 95%
bound is around 8%. "Q4_K_M never flips a label" is not proven and is not claimed.

## Speed

Q8_0, Metal, M5 Pro, warm, one worker, `tests/bench.rs`:

| tokens | p50 | p95 | pairs/s |
| --- | --- | --- | --- |
| 32 | 61 ms | 61 ms | 16.3 |
| 128 | 84 ms | 84 ms | 11.9 |
| 512 | 272 ms | 277 ms | 3.7 |

Inside the 8–18 pairs/s band the earlier estimate predicted for a 4B. The
"50 ms fixed + 0.45 ms/token" model fitted to three points is wrong in the middle: the real
curve is 20 ms at one token, a step between 8 and 12, then flat to 48. See ADR 0017.

CPU (same Q8_0, all cores): **2.3 pairs/s** at 32 tokens, 0.3 at 512 — 7x and 13x slower
than Metal. CPU is a fallback, not a deployment target. CUDA and Vulkan remain unexercised.

**Quantisation does not buy speed** — pairs/s, same machine, warm:

| trunk | 32 tok | 128 tok | 512 tok |
| --- | --- | --- | --- |
| F16 | 15.2 | 11.4 | 4.1 |
| **Q8_0** | **16.7** | **12.4** | 4.0 |
| Q5_K_M | 15.7 | — | — |
| Q4_K_M | 15.1 | 10.9 | 3.3 |

Three times smaller weights, identical throughput — and Q4_K_M is *slower* than Q8_0,
because dequantisation is work too. **Q8_0 dominates: most accurate of the quantised
options and the fastest.** The usual accuracy-for-speed trade is simply not on offer on
this architecture, which is the same fact ADR 0015 reaches from the batching side.

### Where the 61 ms goes — profiled, not guessed (ADR 0017)

The denominator got measured first, because ADR 0015 asserted "we are at this GPU's rate"
against a number nobody had taken. `task bench:hw` (`scripts/metal-peak.swift`, MPS on real
buffers, best of 5) on this M5 Pro:

| | |
| --- | --- |
| fp16 matmul, M=4096 | **31.0 – 31.4 TFLOP/s** |
| fp16 matmul, **M=32** (an NLI pair) | **7.1 – 9.6 TFLOP/s** |
| memory bandwidth (blit) | **267 – 271 GB/s** |

So "4 TFLOP/s is the right order for this GPU" was **false** — it does 31. What is true is
that 31 is not on offer at M=32, where even Apple's own kernels give 7–9. The ceiling is
the shape of an NLI pair, not the quality of the kernel.

**A one-token forward costs 20 ms, and that is the floor.** 4.16 GiB of weights cross the
bus once whatever you ask: 4.47 GB / 19.8 ms = **226 GB/s, 85% of measured peak**. You
cannot answer with a 4B model without reading a 4B model.

**pp32 at 56.5 ms is 94% accounted for:**

```
weight streaming   4.47 GB / 267 GB/s                  = 16.7 ms
arithmetic         2 x 4.21e9 x 32 FLOP / 7.4 TFLOP/s  = 36.4 ms
                                                 total = 53.1 ms   (measured 56.5)
```

openjev adds 3.8 ms on top of llama.cpp's own number for the identical shape — batch build,
memory clear, embedding copy, head. The head is `[3,2560] x [2560]`: 15 kFLOP, four
nanoseconds. There is no pool of waste anywhere in this path.

**Tokens 12 to 48 are free.** `tests/bench.rs` now reports the decomposition
(`report_where_the_milliseconds_go`, p50 of 30 warm forwards):

| tokens | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p50 ms | **22.2** | 22.8 | 24.2 | 30.2 | 59.2 | **60.0** | 65.5 | 80.8 | 128.9 |

16 and 32 tokens cost the same; the marginal cost across that span is **-0.2 ms/token**.
The step is between 8 and 12. A shorter hypothesis buys 2x below ~12 tokens and *nothing at
all* from 12 to 48 — and the golden fixture's median pair is 20 tokens, squarely inside the
flat region. Above 48 the marginal cost climbs to its 0.58 ms/token asymptote.

**Three knobs were swept and all three are flat**, and all three were reverted:

| knob | result |
| --- | --- |
| `n_ubatch` 256 / 512 / 1024 / 4096 / 16384 | flat within noise; the largest is 2% *worse* |
| flash attention on/off, pp32 | 567.0 vs 564.7 t/s — only 8 of 32 layers attend, over 32 tokens |
| pooling `Last` vs `None` + manual read | no difference outside thermal drift |

**The realistic best for one 32-token pair on this trunk is ~50 ms.** Against 61 ms today
that is an 18% prize, and it would have to be won inside ggml-metal. **61 ms is the floor
for 4B on this hardware, to within 20%.**

### Batching: flat for us, 2.6x for stock llama.cpp — the one open engineering win

**ADR 0015's decision stands; its mechanism does not.** "The GPU was never waiting" is true
at batch 1 and false at batch 8. `llama-batched-bench` on llama.cpp b9620, same model, same
shape:

| sequences | T_pp | ms per pair | ours |
| --- | --- | --- | --- |
| 1 | 0.056 s | 56.0 | 60.3 |
| 4 | 0.142 s | 35.5 | 65 |
| 8 | **0.207 s** | **25.9** | **68** |
| 16 | **0.321 s** | **20.1** | 65 |

Stock llama.cpp coalesces eight recurrent sequences into one ubatch and gets 2.2x; sixteen
gets 2.8x. Our path gets nothing, and is 2.6x off. The in-process table says the same from
the other side: 256 tokens in one sequence costs 129 ms against 60 ms for 32 — the GPU will
do four pairs' worth of tokens for twice one pair's price, because at M=32 it is idle and
says so.

**Cause not identified, and not guessed at.** `n_ubatch` ruled out (swept, flat), pooling
type ruled out (swept, flat). `clear_kv_cache()` per group is the remaining suspect and
**could not be A/B'd** — remove it and the next group's `decode` fails, because the design
depends on it. Naming a suspect without the measurement is the mistake ADR 0017 exists to
correct.

This is **throughput, not latency**: it does not touch the 61 ms a single `predict` pays.
It is worth up to 2.8x on `rerank`, which is the call that scores N options.

### The highest-leverage optimisation remaining

**A smaller trunk — and the one this status used to name does not exist.**

`AlexWortega/openjev` has exactly **three** checkpoints and **two** sizes: the 4B v1, the 4B
v2 (ours), and a 35B MoE. The 0.8B and 2B *were trained* — `results/qwen0.8b_mnli_gpqa.json`
and `results/qwen2b_full.json` are keyed on `ckpt/qwen3.5-0.8b-nli` and `ckpt/qwen3.5-2b-nli`,
the author's local training paths, at MNLI-m 0.869 and 0.886 — **and neither was ever
uploaded.** The "0.8B at 57 pairs/s, available today" line in the previous status described
a checkpoint that cannot be downloaded, and is withdrawn. The 0.6B / 2B / 4B on openjev.com
are stock general LLMs (`Qwen3-0.6B`, `MiniCPM5-2B`, `Qwen3.5-4B`) run in-browser for typed
option-logits — not NLI heads.

**What does exist runs in our runtime today and answers in 10 ms** (ADR 0018). llama.cpp has
**no DeBERTa support at all**, which rules out the whole mDeBERTa/DeBERTa-v3 family; it does
convert `ModernBertForSequenceClassification` *with* its 3-label head. Both candidates
converted and measured here — warm p50 over a running server, median of 100 calls:

| | ModernCE-base-nli | nli-distilroberta-base | openjev 4B |
| --- | --- | --- | --- |
| params | 149.6M | 82.1M | 4.21B |
| GGUF F16 | 301 MB | 167 MB | 8.4 GB |
| **warm p50** | **10.4 ms** | **8.6 ms** | 61 ms |
| **agreement vs the 4B, 35 pairs** | **32 / 35** | **29 / 35** | — |
| llama.cpp vs its own HF result | 35/35, mean abs dp 0.0015 | 35/35, 0.0018 | — |

**6x, for 3 pairs in 35.** Two of ModernCE's three disagreements are low-confidence or
arguably the 4B's error; **one (pair 32) is confidently wrong and a confidence gate will not
catch it.** Two silent traps are named in ADR 0018: ModernCE's `config.json` declares the
wrong `id2label` (trusting it gives 2/35 instead of 32/35), and `llama-server`'s
`/v1/rerank` returns only `logit[0]`.

**Not built.** It is a second architecture, a second thing that breaks, and signing it off
needs a fixture larger than 35 pairs. The numbers are in ADR 0018 so the call is the
owner's.

## Weights and conversion

`scripts/convert-model.sh <workdir>` is the reproducible path, and it documents both
obstacles rather than leaving them to be rediscovered:

* **Issue #27019 is already fixed on llama.cpp master** (checked 2026-09-20), independently
  of draft PR #27132, which is stale and should *not* be applied. `conversion/qwen.py`
  carries `_LinearAttentionVReorderBase`, which handles both the `ssm_conv1d` kernel dim
  and the `in_proj_a/b` layout. The converter is not the blocker the design assumed.
* **What actually blocks** is that the converter registers
  `Qwen3_5ForConditionalGeneration` / `ForCausalLM` and not
  `Qwen3_5ForSequenceClassification`, and then dies on `score.weight` with "Can not map
  tensor". Worked around by presenting the trunk under the generative architecture name and
  dropping the head, which openjev applies in Rust anyway.
* The converter/runtime pairing that *would* have been silently wrong — the
  linear-attention V-head order — is fine: master and the tree vendored by
  `llama-cpp-sys-2 0.1.156` both broadcast with the same tiled `ggml_repeat_4d`. The golden
  fixture is what proves it rather than that reading.

`models.toml` now pins `revision = 4b5f9a67...` and carries the Q8_0 sha256. Artefact
digests, for whoever uploads them:

```
b3af9b47…a305  openjev-4b-nli-v2-F16.gguf      8 424 385 248
f00d017a…7332  openjev-4b-nli-v2-Q8_0.gguf     4 482 394 848
87b945e2…2aff  openjev-4b-nli-v2-Q5_K_M.gguf   3 074 978 528
48a334bb…d309  openjev-4b-nli-v2-Q4_K_M.gguf   2 708 796 128
33af00c8…3db2  score.safetensors               (score.weight, bf16 [3, 2560], no bias)
```

## Three more silent-wrong-answer bugs, found by reading the reference

All three passed every weightless test, because weightless tests cannot see them.

1. **`rerank` scored `(option, question)` — backwards.** Entailment is directional, so this
   returned a different ranking, forever, with no error. The unit test passed under both
   orders because the fake backend keyed its canned state on `tokens[0]` — the premise word
   — which under the wrong order happened to be the option. *The test was reading the bug as
   its fixture.* (ADR 0012)
2. **The template did not trim its fields.** The reference renders
   `premise=p.strip(), hypothesis=h.strip()`. A test asserted the opposite and called it
   "not silently changing the input"; declining a transformation the model expects is not
   neutral. (ADR 0011)
3. **`padding_side` defaulted to left**, per design 01 §5. Correct for a dense causal
   decoder, wrong here: a recurrence cannot be masked, so leading pads advance the state at
   every real position that follows. Moot on today's path — llama.cpp pads nothing — and
   hardest to find on the day it stops being moot. (ADR 0013)

Plus: the server ranked by `unwrap_or(0)` when the entailment label was missing, silently
reversing the ranking to P(contradiction); and the startup banner printed `llama-cpp-2
0.1.0`, a version that crate has never had.

**And a fourth, from a different direction: `context >= 32768` SIGSEGVs on Metal.** Bisected
— 28672 fine, 32768 and 65536 dead, CPU fine at all of them, so it is a ggml-metal
allocation ceiling and not a model limit. `context` is user-editable registry data, so the
config surface itself invites the value that kills the process, with no error and no log.
openjev now refuses before allocating and names the value, the ceiling, the device, the
config key and `--device cpu`. It cannot be probed: the failure is a segfault, not an error
return, so there is no process left to report it. (ADR 0016)

## Still not proven

* **Nothing is published.** The GGUF and head exist only on this machine. `models.toml`
  points at `muthuishere/openjev-gguf`, which does not have them, so a fresh `openjev serve`
  404s. The end-to-end runs above used a hand-populated cache. **This is the one remaining
  blocker to a working first-run experience**, and it needs an HF account decision.
* **CUDA and Vulkan are wired and unexercised.** macOS only here. Unchanged from before, and
  deliberately not guessed at.
* **`usage.tokens` is always 0.** Core's `Session` still returns no token count.
* **35 pairs is a small fixture.** It spans the label space and the awkward cases, but a
  real agreement claim — especially for the smaller quants — wants a few hundred pairs from
  SNLI/ANLI dev. That is design risk R1's original experiment, and ADR 0018 is what finally makes it
  load-bearing: a fixture this small cannot sign off a second model that disagrees with the
  4B on three of its pairs.
* **R5 is closed on this machine only.** `otool -L` shows no `libllama`/`libggml` (only the
  system Metal frameworks), there is no `.metallib` beside the binary, and a copy of the
  binary run from an unrelated directory still does a real Metal forward pass. It has not
  been run on a *different machine*, which is the part that would prove the shader embedding
  rather than a warm local cache.
* The tier→model tables, the short circuit, and everything in `plugins/herdr-jev` are
  unchanged and still carry the caveats the last status listed. The plugin now talks to a
  `rerank` that ranks the right way round, which invalidates any calibration done before.

## Yours to decide

1. **Publish the artefacts.** Everything else is done; this is one upload and one sha256
   paste, and until it happens openjev only runs where the cache was filled by hand.
2. **Q8_0 or Q4_K_M.** The accuracy cost is measured and the speed gain is in the table.
3. **Is 16 pairs/s enough?** If not, the answer is a fast tier, not more engineering on
   this trunk — ADR 0017 shows there is at most 18% left in it. The fast tier is
   ModernCE-base-nli: **10.4 ms, 32/35 agreement, runs in the runtime we already ship**.
   The question is whether 3 disagreements in 35 — one of them confidently wrong — is a
   price worth 6x, and whether the fixture grows to a few hundred pairs before it is paid.
4. **Does the short circuit ship on or off?** Unchanged from the last status.
