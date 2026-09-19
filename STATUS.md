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
| all-zero hidden states (ADR 0003 guard) | none, on any pair |

The whole product path is exercised, not just the library: `openjev predict/rerank/grade`
and `POST /v1/predict|rerank|grade` on a running `openjev serve` all return the fixture's
numbers. `rerank` puts *carbon dioxide* first for the photosynthesis question; `grade`
scores a faithful answer 0.976 entailment and a wrong one 0.982 contradiction.

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

Inside the 8–18 pairs/s band the earlier estimate predicted for a 4B. Cost is roughly
**50 ms fixed + 0.45 ms/token**.

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

### What was optimised, and what it bought

**Batching: implemented, correct, and worth nothing.** (ADR 0015.)

| tokens | max_seqs=1 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- |
| 32 | 16.3 | 16.5 | 15.7 | 15.8 |
| 128 | 11.9 | 11.5 | 11.9 | 12.1 |
| 512 | 3.7 | 3.6 | 3.4 | — |

The flat baseline looked like per-call overhead waiting to be reclaimed. It was not.
Batching is a win for *generation*, where emitting one token is memory-bound and the
arithmetic units idle. An NLI *prefill* is a real matmul against every weight in the model
and is already compute-bound at ~4 TFLOP/s sustained — eight sequences take eight times as
long because there was never any idle capacity to fill.

The code stays, defaulted off via `server.max_seqs`, because it is *proven correct*: at
`max_seqs=16` the entire golden suite passes with results identical to the unbatched path.
That is a real demonstration that llama.cpp keeps a separate gated-DeltaNet state per
`seq_id`, and it is one flag away on any machine where the GPU is genuinely underfed.

`n_ctx` turned out to be a memory lever, not a speed one: 8192 and 16384 measure the same.

### The highest-leverage optimisation remaining

**A smaller trunk.** The 0.8B measured 57 pairs/s against this 4B's 16.3 — a 3.5x speedup
that is available today for whatever accuracy the 0.8B checkpoint actually has, and that is
a product decision rather than an engineering one. Nothing in the engine would change: the
registry already treats a model as configuration.

There is no second lever worth the name. The two obvious candidates were both measured and
both are worth nothing: batching (flat), and quantisation (flat, and negative below Q8_0).
`n_ctx` is memory, not speed. The ~50 ms fixed cost per forward is llama.cpp graph setup and
is worth attacking only if the fixed term ever matters more than it does at 32 tokens.

The honest summary: **openjev already runs at the speed a 4B prefills at on this machine.**
Getting materially faster means asking the hardware to do less arithmetic, and the only
lever with real leverage there is parameter count.

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
  SNLI/ANLI dev. That is design risk R1's original experiment and it is still worth running.
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
3. **Is 16 pairs/s enough?** If not, the answer is the 0.8B trunk, not more engineering on
   this one — and that is a question about how good the 0.8B has to be.
4. **Does the short circuit ship on or off?** Unchanged from the last status.
