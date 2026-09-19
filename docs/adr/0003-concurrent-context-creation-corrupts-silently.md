# 0003 — Concurrent llama.cpp context creation corrupts silently; model load is serialised process-wide

Status: **accepted**, found by test 2026-09-19. Not anticipated by
`docs/design/01-inference-backend.md`; closest neighbour is R9.

## What happened

The gated integration tests opened five independent `LlamaCppBackend` instances — five
models, five contexts, five `Mutex`es, no shared Rust state — from five test threads in
one process. Exactly **one, randomly chosen, returned an all-zero hidden state**.

Reproducible: 3 of 3 parallel runs failed, each on a different test; 3 of 3 runs with
`--test-threads=1` passed. So it is not an input, a length or a device — it is
concurrency at open time.

The failure mode is the problem, not the frequency. There was:

- no error from `llama_decode`,
- no null pointer, no panic, no allocation failure,
- no log line, at any level,

just a 1024-float vector of zeros. Fed to the classification head, an all-zero hidden
state produces zero logits, and `softmax([0,0,0])` is a perfectly uniform distribution
whose argmax is `labels[0]` — `contradiction`, for our model, **every time**. A silent,
deterministic, plausible wrong label. This is the same class of failure as R2 and it
arrives from a completely different direction.

`llama_backend_init` was already behind a `OnceLock`, so the global init was not racing.
The race is in ggml's device/backend setup during **model load and context creation**,
which carries global state that `llama_backend_init` does not make re-entrant.

## Decision

1. **Model load and context creation are serialised process-wide** by a `static INIT_LOCK:
   Mutex<()>` in `backends/llamacpp.rs`, held across `load_from_file` + `new_context`.
   This is boot-only work, so steady-state throughput is untouched, and it removes the
   entire failure class rather than narrowing the window. 4 of 4 parallel runs clean
   after the change.

2. **The backend refuses an all-zero or non-finite pooled state**, in `forward`, as a
   `JevError::Backend`. The lock is the fix; this is the guard for the next instance of
   the same shape. A dead vector must never reach the head — the difference between a
   loud failure and a wrong answer is the whole product.

## Why not just document "use one instance"

Because the failure is silent. A rule that is only enforced by a sentence in a README gets
broken by the first person who runs two models side by side, or by a test harness, and
they get labels rather than an error. The constraint is real, so it is expressed in code.

## Consequences

- Opening N backends takes N serialised model loads. For `openjev serve` this is
  irrelevant: the design already specifies one model and one worker by default.
- Decode is **not** covered by this lock — each backend keeps its own `Mutex` over its own
  context, and concurrent `forward` across separate instances has run clean. That is the
  weaker claim, and it is deliberately the weaker claim: it has been exercised, not proved.
  If cross-instance decode ever shows the same signature, the honest fix is to widen this
  lock to cover `forward`, and to say out loud that the backend has no cross-instance
  parallelism.

## What would reopen this

- A llama.cpp or `llama-cpp-2` bump: re-run the integration tests **in parallel**, which
  is the default, so this needs no discipline.
- Any move to multiple concurrent contexts as a throughput strategy.
