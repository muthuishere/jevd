# 0001 — llama.cpp resets the gated-DeltaNet state per sequence (design risk R2 is closed)

Status: **accepted**, measured 2026-09-19. Supersedes nothing; closes R2 in
`docs/design/01-inference-backend.md` §6.

## The question

Design risk R2, and the single assumption v0.1 rests on: a Qwen3.5 hybrid trunk carries a
**gated-DeltaNet recurrent state** across the 24 linear-attention layers. If llama.cpp
leaks that state from one sequence into the next, the second sequence gets a hidden state
contaminated by text it never saw. The head then returns a **confident wrong label**.
There is no exception, no NaN, and no log line. It is the worst failure this system has,
and nothing else in the crate can detect it.

So it was tested before any product code was written.

## The experiment

Two independent runs, both on `unsloth/Qwen3.5-0.8B-GGUF` @ Q8_0 — the same hybrid
architecture as our 4B target (24 layers, `linear_attention` × 18 + `full_attention` × 6,
`linear_conv_kernel_dim 4`, `attn_output_gate: true`), chosen over the 4B only because it
downloads in a minute. Our own checkpoint is not converted yet, which is exactly why R2
had to be answered against a community GGUF.

Method: forward a target sequence **alone**, then forward it again **after** a long
unrelated sequence, and compare. A leaked state shows up as a deviation that **grows with
the length of what preceded it**. Float non-determinism shows up as a deviation that is
flat and tiny. The two are distinguishable, and distinguishing them is the whole point.

### Run 1 — `llama-embedding --pooling last` (llama.cpp b9620), before any Rust existed

Semantic control, to calibrate what a *real* difference looks like: the target versus an
unrelated sentence gives `1-cos = 7.0e-1`, `relL2 = 1.15`.

| distractor before the target | CPU (`-ngl 0`) | Metal (`-ngl 99`) |
|---|---|---|
| 7 words   | `relL2 0.000e0` | `relL2 8.3e-4`, `1-cos 3.5e-7` |
| 35 words  | `relL2 0.000e0` | `relL2 8.3e-4`, `1-cos 3.5e-7` |
| 140 words | `relL2 0.000e0` | `relL2 7.5e-4`, `1-cos 2.8e-7` |
| 560 words | `relL2 0.000e0` | `relL2 7.5e-4`, `1-cos 2.8e-7` |
| 1400 words| `relL2 0.000e0` | `relL2 7.5e-4`, `1-cos 2.8e-7` |

**On CPU the result is bit-identical** — byte-for-byte the same vector after a 1400-word
distractor as with no distractor at all. That is a complete answer on its own: the state
is reset, exactly, every time.

On Metal the deviation is **flat at ~8e-4 relative from 7 words to 1400 words** — a 200×
change in the state left behind moves the number not at all. It is also the same magnitude
when the "distractor" is a *verbatim copy of the target*. That is the signature of
non-associative float reduction under a different kernel tiling, not contamination. It sits
**six orders of magnitude** below the semantic control.

### Run 2 — through our own `Backend::forward`, via the C API (`llama-cpp-2` 0.1.156)

`crates/openjev-core/tests/real_weights.rs::recurrent_state_does_not_leak_between_sequences`,
on Metal, Apple M5 Pro:

```
distractor     8 tokens -> relL2 0.000e0  1-cos 1.110e-16
distractor    64 tokens -> relL2 0.000e0  1-cos 1.110e-16
distractor   256 tokens -> relL2 0.000e0  1-cos 1.110e-16
distractor  1024 tokens -> relL2 0.000e0  1-cos 1.110e-16
determinism floor:        relL2 0.000e0
```

Exactly zero, on Metal too, because our backend additionally clears memory between
sequences (below). `1-cos = 1.1e-16` is one f64 epsilon — it is the subtraction `1.0 - 1.0`,
not a difference.

## Decision

**R2 is closed. llama.cpp resets the recurrent state correctly for a Qwen3.5 hybrid, and
we do not rely on that alone.**

Three things, in order of how much they are trusted:

1. **v0.1 declares `!Caps::BATCH` and forwards one sequence per call.** Nothing shares a
   context with anything else. This is also why nothing pads on this path.
2. **`clear_kv_cache()` between sequences**, which on a hybrid model clears the recurrent
   and conv states along with the KV. A leaked state is unrepresentable rather than
   merely unlikely.
3. The measurement above, as the regression test, so a future llama.cpp bump that breaks
   the property fails loudly instead of quietly relabelling.

The cost of (1) and (2) is throughput we are not yet spending: a batching backend must
re-prove this before it may set `Caps::BATCH`.

## What would reopen this

- Setting `Caps::BATCH` on any backend — multi-sequence batches share a context and the
  experiment must be re-run per-sequence within one batch.
- A different hybrid architecture (`arch` is config; the recurrence is not).
- A llama.cpp bump. The test is cheap; run it.

## Notes

The tolerance in the test is `relL2 < 1e-2`, not `== 0`. Measured worst case across CPU,
Metal and the CLI path is `8.3e-4`; the semantic control is `1.15`. The threshold sits
three orders below a real difference and one above the observed float noise, so it catches
contamination without failing on a kernel-scheduling change.
