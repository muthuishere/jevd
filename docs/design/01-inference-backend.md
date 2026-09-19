# openjev — inference backend design

**Organised against one question: what runs `AlexWortega/openjev @ qwen3.5-4b-nli-v2` on
an Apple Silicon laptop this month, with no Python at runtime, without us first writing a
research-grade kernel.**

Status: **built and measured against the reference, 2026-09-20.** This document is kept as
written so the reasoning can be checked against what happened; where reality diverged, an
ADR says so and the ADR wins.

Diverged so far:

| this doc says | reality | where |
|---|---|---|
| left-pad a batch (§5) | right-pad + explicit pool index; a recurrence cannot be masked | ADR 0013 |
| default Q5_K_M (§5) | default Q8_0, on measured probability drift | ADR 0014 |
| the converter is broken, apply PR #27132 (§0, R3) | #27019 is already fixed on master; what actually blocks is the unregistered `ForSequenceClassification` architecture and the unmappable `score.weight` | `scripts/convert-model.sh` |
| batching is the Engine's job and wins throughput (§2) | it wins nothing here — an NLI prefill is already compute-bound | ADR 0015 |

Closed risks: **R1** (ADR 0014), **R2** (ADR 0001, re-measured on the real 4B), **R3** and
**R4** (`scripts/convert-model.sh` + the golden fixture), **R6** and **R7**
(`tests/golden.rs`), **R5** (no `libllama`/`libggml` in `otool -L`, no `.metallib`
sidecar, and a copy of the binary run from an unrelated directory still does a real Metal
forward — though not yet on a second machine). Still open: **R8**, **R9**.

Target: `openjev-core` (Rust lib), `openjev-cli`
(binary `openjev`, `openjev serve`), `herdr-jev` (Go plugin, designed elsewhere).

## 0. Verified upstream facts (do not re-derive)

Given (owner-verified, trusted):

- `architectures: ["Qwen3_5ForSequenceClassification"]`, `model_type qwen3_5`,
  `problem_type single_label_classification`,
  `id2label {0: contradiction, 1: entailment, 2: neutral}`.
- `nli_template: "Premise: {premise}\nHypothesis: {hypothesis}"`.
- text trunk: hidden 2560, 32 layers, head_dim 256, 16 q-heads, 4 kv-heads,
  vocab 248320, rms_eps 1e-6, `attn_output_gate: true`, `full_attention_interval: 4`.
- `layer_types`: 24 `linear_attention` + 8 `full_attention`. Gated-DeltaNet style:
  `linear_conv_kernel_dim 4`, 16 key heads / 32 value heads, key+value head dim 128.
- A `vision_config` exists. `pad_token_id 248044`.
- candle 0.11.0 has qwen3 / qwen3_moe / qwen3_vl / quantized_qwen3 — **none** implement
  Qwen3.5 linear attention.
- candle `qwen3.rs` `Model::forward(&mut self, input, offset) -> Tensor` returns hidden
  states; `ModelForCausalLM` bolts on `lm_head`. That is the right hook for a head.
- Host: macOS arm64, Metal, rust 1.95.

Found by this design pass (2026-09-19), with sources:

- **candle git `main` has no `qwen3_5`, no `qwen3_next`, no `gated_deltanet`, no generic
  linear-attention module.** Not in release, not in main. A port is greenfield.
- **llama.cpp's *runtime* already executes qwen3_5 hybrid GGUFs correctly.** ggml has the
  ssm/conv1d/delta-rule ops and the hybrid recurrent+KV cache. The broken part is the
  *converter*: `convert_hf_to_gguf.py` mishandles `ssm_conv1d` kernel-dim reorder and
  fails to expand `in_proj_a` / `in_proj_b` (65536 → 131072). Issue
  [ggml-org/llama.cpp#27019](https://github.com/ggml-org/llama.cpp/issues/27019), fix PR
  #27132 open/unmerged as of writing. Pre-converted community GGUFs (unsloth) run fine.
- **ONNX is a dead end for this architecture.** As of 2026-02 there are no ONNX operators
  for any non-softmax attention variant; hybrid recurrence exports as dynamic-shape soup
  and blows the 2 GB protobuf limit. The only ONNX work in flight is vendor-specific
  packed-hybrid export inside ORT GenAI
  ([onnxruntime/mobius#737](https://github.com/onnxruntime/mobius/issues/737)), not a
  portable graph. `ort` itself is healthy (2.0.0-rc.12) — the crate is not the problem,
  the graph is.
- **llama.cpp does carry a classification head through GGUF**: `cls.output.weight`,
  `pooling_type = RANK`, `classifier.output_labels` metadata. But that path is built and
  tested for yes/no rerankers on dense-attention models, not a 3-way head on a hybrid
  recurrent trunk. We do not bet on it — see §1.
- No other Rust runtime is credible here: mistral.rs has no gated-DeltaNet, RustyLLM is a
  hobby GGUF runner, `aprender` has an *open issue* asking for GDN on GPU.

## 1. Decision: backend

> **D1 — v0.1 ships on llama.cpp (statically linked) as the *trunk only*, and applies the
> 3-way classification head in Rust ourselves. candle stays a post-1.0 option behind the
> same trait, not a launch dependency.**

The owner's "pure Rust" constraint is withdrawn as a hard requirement, and this design
takes that seriously: **v0.1 is not pure Rust.** It is one statically linked binary with
no Python, no dynamic library to install and no runtime download beyond weights — which
is the property that was actually wanted. Rewriting a research kernel to keep a language
badge is the expensive way to buy nothing.

The load-bearing trick, and the reason this is not just "option (d)":

**Do not ask GGUF to carry the classifier.** The head in this checkpoint is one
`score: Linear(2560 → 3, no bias)` — 30 720 bf16 values, 60 KB. We ship it as a separate
tiny safetensors file and do the matmul in Rust. llama.cpp is used purely as "hidden
states for a token sequence": `--embedding`, causal attention, last-token pooling, 2560
floats out. Then:

```rust
// the entire "classification head", in openjev-core
let logits = score_w.matmul(&h_last)?;   // [3]
let probs  = softmax(&logits)?;          // contradiction / entailment / neutral
```

This deletes the whole class of risk around `pooling = rank`, `cls.output.weight`,
`classifier.output_labels`, converter support for a 3-label head, and llama.cpp's
reranker flag interactions. It also makes `latents` (§4) free — we already hold the
hidden state — and makes a future backend swap cheaper, because every backend only has to
answer one question: *give me the last hidden state*.

### Why not the others

**(a) Full candle port of Qwen3.5 hybrid.** Honest estimate: **18–30 engineer-days** to
numerically-correct-on-Metal, by someone who has written kernels before; 40+ for someone
who has not. The hard parts, named:

1. *Gated delta rule recurrence.* Chunked form for prefill (you cannot afford token-by-
   token over a 512-token premise+hypothesis), sequential form for decode. Getting
   chunked and sequential to agree to 1e-3 is where the days go.
2. *Causal depthwise conv1d, kernel 4, with a carried state* across chunk boundaries.
   candle has conv1d; it does not have a stateful causal conv with a cache.
3. *Head-count asymmetry.* 16 key heads / 32 value heads, both head_dim 128, against a
   2560 hidden. The key-head→value-head broadcast is exactly the `in_proj_a/b` expansion
   llama.cpp's converter gets wrong — a published landmine.
4. *Hybrid cache.* 8 layers need a KV cache that grows with tokens; 24 need a fixed-size
   recurrent state `[num_v_heads, v_head_dim, k_head_dim]` plus conv state. One `Cache`
   type, two disciplines, and `seqlen_offset` means different things in each.
5. *Gated full attention* (`attn_output_gate: true`) — not candle's qwen3 attention; the
   output is gated before o_proj. Silent-wrong if missed.
6. *Metal.* candle's Metal path will not have a fused kernel for any of this, so the delta
   recurrence lands as a loop of small matmuls. Expect it to be slower than llama.cpp for
   a while, not faster.

Plus the thing nobody budgets: there is no reference to diff against on this machine
unless we also stand up a Python transformers run to produce golden logits. That is a
day, and it is mandatory (§5).

**(b) Text-only subset.** Yes — **the ViT is skippable, completely.** `Qwen3_5ForSequence
Classification` embeds text ids through the text trunk; the vision tower only contributes
when image placeholder tokens are present in the input, and the NLI template produces
none. Concretely: never emit image tokens ⇒ `visual` is never called ⇒ we do not load,
convert or implement it. This is a real saving (skip a whole ViT + merger + M-RoPE
interleave) and it applies to *every* backend option, so it is not a backend choice — it
is a constraint on all of them. **v0.1 is text-only and the config says so; images are a
`NotSupported` capability, not a silent wrong answer.**

**(c) ONNX via `ort`.** Rejected. No exported ONNX exists for this checkpoint, and none
plausibly can: no ONNX ops for non-softmax attention, dynamic-shape recurrence, 2 GB
protobuf ceiling on a 4B model. Exporting would mean unrolling the recurrence to a fixed
sequence length, which is worse than writing the kernel. Revisit only if ORT GenAI ships
a portable packed-hybrid op set.

**(d) GGUF via llama.cpp, unmodified.** This is the chassis we take, minus its classifier
path. The only genuine cost is that **no GGUF of this checkpoint exists** and the
converter is buggy. That cost is paid *once, by us, at publish time* — see §3.

**(e) Staged.** Adopted, and that is what D1 is. v0.1 llama.cpp + Rust head; v0.2+ candle
behind `InferenceBackend`, promoted only when it matches golden logits.

### The Python line

Python is **forbidden at runtime and allowed at publish time.** We run patched
`convert_hf_to_gguf.py` once on a maintainer machine, extract `score.weight` once, publish
both artefacts to a HF repo we control, and pin them by sha256. Users get a binary and a
download. This is the whole reason the converter bug is a footnote instead of a blocker.

## 2. The abstraction

Boundary rule: **`openjev-serve` and the CLI know about `Scores` and `Latents`. They never
know a backend name except to print it.**

```rust
pub trait InferenceBackend: Send + Sync {
    fn describe(&self) -> BackendInfo;                 // name, device, dtype, ctx, caps
    fn capabilities(&self) -> Caps;                    // bitflags: LATENTS, BATCH, IMAGES
    fn forward(&self, batch: &[EncodedInput]) -> Result<Vec<Hidden>, JevError>;
}
```

- **One method.** `forward` returns last-token hidden states, nothing else. `predict`,
  `rerank`, `grade`, `latents` are *free functions in core* built on `forward` + the head
  (§4). A backend that only implements `forward` gets all four operations for free, and
  cannot implement them inconsistently.
- **Sync trait, async shell.** The trait is blocking. Inference is CPU/GPU-bound and
  llama.cpp's context is `!Sync` for concurrent decode anyway. `openjev-core` wraps it in
  an `Engine` that owns a bounded worker pool (`N = 1` by default; a 4B model is not
  sharable across threads on one laptop) and exposes `async fn predict(...)` over a
  channel. Axum handlers await; the backend never sees a runtime. This also means a pure-
  CPU backend cannot stall the HTTP server's health endpoint.
- **Batching is the Engine's job, not the backend's.** The Engine coalesces requests
  arriving within a small window (`batch.max_size`, `batch.max_delay_ms`) into one
  `forward`. A backend that cannot batch declares `!Caps::BATCH` and the Engine loops —
  same API, worse throughput, no code change upstream. llama.cpp v0.1 is `!BATCH`
  (sequential per-sequence, which also means *no padding at all* — see §4).
- **Errors.** One `JevError` enum in core, `thiserror`, non-exhaustive. Variants that
  matter: `ModelNotFound`, `BackendUnavailable { needed, have }`, `NotSupported(Cap)`,
  `ContextOverflow { tokens, limit }`, `Device(...)`, `Backend(#[source] anyhow::Error)`.
  Backends are allowed to be sloppy internally (`anyhow`) and must map at the boundary.
  HTTP maps `NotSupported` → 501, `BackendUnavailable` → 503 with the resolution in the
  body, `ContextOverflow` → 413.
- **Degrading.** `NotSupported` is always explicit and always at the edge: capability is
  checked when the route is built, not when the tensor is allocated. If the live backend
  lacks `LATENTS`, `/v1/latents` is registered and returns a 501 naming the backend and
  what would provide it. We never silently substitute.
- **Registration** is a `BackendFactory` registry keyed by name, populated by cargo
  features (`backend-llamacpp`, `backend-candle`). Unbuilt backends are absent from the
  registry, which is how §3's "binary lacks the backend" check is implemented.

## 3. Model is config, not code

> **D2 — a model is a config entry. `openjev/qwen3.5-4b-nli-v2` is the default value of a
> field, and appears in exactly one place in the source: a built-in TOML string.**

A **built-in registry** of known-good models is compiled in (one embedded TOML), and a
user file at `$XDG_CONFIG_HOME/openjev/models.toml` (macOS: `~/.config/openjev/`, we do
not use `~/Library/Application Support` — CLI convention beats Apple convention) is
merged over it by id. User wins. `--model <id>` selects; `--model-file <path>` loads an
ad-hoc entry.

```toml
[models."openjev-4b-nli-v2"]          # the default; nothing else is special about it
default      = true
repo         = "AlexWortega/openjev"
revision     = "<pinned-commit-sha>"   # never "main"
subfolder    = "qwen3.5-4b-nli-v2"
arch         = "qwen3_5"               # informational + backend gating
task         = "sequence-classification"
template     = "Premise: {premise}\nHypothesis: {hypothesis}"
labels       = ["contradiction", "entailment", "neutral"]   # index == id2label key
tokenizer    = { file = "tokenizer.json", pad_token_id = 248044, padding_side = "left" }
head         = { kind = "linear", in = 2560, out = 3, bias = false, file = "score.safetensors" }
context      = 8192
modalities   = ["text"]

  [models."openjev-4b-nli-v2".backends.llamacpp]     # preference order = declaration order
  weights   = { repo = "muthuishere/openjev-gguf", file = "openjev-4b-nli-v2-Q5_K_M.gguf",
                sha256 = "...", size_bytes = 2_900_000_000 }
  min_version = "b7000"                # llama.cpp build that runs qwen3_5
  [models."openjev-4b-nli-v2".backends.candle]
  weights   = { repo = "AlexWortega/openjev", glob = "*.safetensors" }
  requires  = ["qwen3_5-hybrid"]       # a capability token the backend advertises
```

Resolution at startup: pick the model entry → walk `backends` in order → keep the first
whose factory is in the registry *and* whose `requires` are all advertised *and* whose
device constraints hold (§5). Nothing matches ⇒ hard fail, exit 78, with the exact reason
per candidate:

```
error: model 'openjev-4b-nli-v2' needs a backend this binary does not have
  llamacpp  not compiled in (build with --features backend-llamacpp)
  candle    present, but lacks capability 'qwen3_5-hybrid'
```

That is the answer to "configured model needs a backend the binary lacks": **refuse
loudly at boot, name the missing feature, never fall back to a different model.** Falling
back silently would make `/v1/predict` return confident wrong labels from the wrong
weights, which is the worst failure this system can have.

A **different architecture later** is a new `arch` + a new `backends` block. `template`,
`labels`, `head` and `tokenizer` are already data, so a 2-label entailment model, a
different prompt shape, or a 7-label taxonomy is a TOML edit. The only thing that is
code is a *new head kind* (e.g. `kind = "mlp"`), which is one match arm.

## 4. Weight acquisition

- **Crate: `hf-hub`**, blocking API, `ureq`/rustls (no OpenSSL, keeps the static binary
  static). It already speaks HF's resolve URLs, revisions and auth (`HF_TOKEN`).
- **Cache.** Respect `HF_HOME` / `HF_HUB_CACHE` when set — a developer with a 200 GB HF
  cache should not get a second copy. Unset ⇒ our own dir, `$XDG_CACHE_HOME/openjev`
  (macOS `~/.cache/openjev`, not `~/Library/Caches` — same reasoning as §3). Layout
  `<cache>/models/<model-id>/<revision>/{weights.gguf,score.safetensors,tokenizer.json}`,
  revision-addressed so two pinned revisions coexist.
- **Resumable.** Download to `<file>.partial`, HTTP `Range` on retry, `fsync` + atomic
  rename on completion. A truncated 3 GB download must never be readable as a model.
- **Integrity.** sha256 from the config, streamed during download, checked before the
  rename. Mismatch ⇒ delete the partial and fail; do not keep and warn. On subsequent
  runs, trust the filename+size and a stamp file (`.ok` containing the digest) rather than
  re-hashing 3 GB every boot; `--verify` forces a full re-hash.
- **Consent before bytes.** First run prints the size and *stops* unless it is a TTY and
  the user says yes, or `--yes` / `OPENJEV_ASSUME_YES=1`:
  ```
  openjev: model 'openjev-4b-nli-v2' is not in the cache.
           will download 2.9 GB (gguf) + 60 KB (head) + 11 MB (tokenizer)
           from huggingface.co/muthuishere/openjev-gguf @ <sha>
           into ~/.cache/openjev/models/openjev-4b-nli-v2/<rev>/
  Continue? [y/N]
  ```
  Non-TTY (a herdr plugin spawning us) without `--yes` ⇒ exit with that text on stderr and
  a distinct code, so the caller can surface it rather than hang on a prompt.
- **Progress** only when stderr is a TTY: one `indicatif` bar, bytes + rate + ETA. Not a
  TTY ⇒ a line every 10 % (herdr logs stay readable). `--quiet` kills both.
- **Offline.** `OPENJEV_OFFLINE=1` (and honour `HF_HUB_OFFLINE=1`): resolve from cache
  only; a miss is an error naming the exact path expected, never a network call.
- **Pinning.** `revision` is a commit sha in the registry, never a branch. `openjev model
  pin <id> --revision <sha>` writes a user override. A config carrying a branch name gets
  a startup warning — reproducibility is the point.
- `openjev model ls / show / fetch / rm / path` round it out; `fetch` is the "prewarm in
  the Dockerfile" verb.

## 5. Inference semantics

**Prompt.** `template` from config, `{premise}` / `{hypothesis}` substituted with no
trimming and no extra BOS beyond what the tokenizer adds. The `\n` is literal — the
config is TOML, so it is a real newline in a basic string. Any drift here changes the
label distribution, so the template is covered by a golden test, not a comment.

**Tokenizer.** `tokenizers` crate, `tokenizer.json` from the model subfolder (not the base
Qwen repo — vocab is 248320, it is not stock). We do **not** use llama.cpp's embedded
tokenizer even when using llama.cpp: one tokenizer for every backend, so backend swaps
cannot move the decision boundary.

**Pooling — this is the thing to get right.** The head reads the **last non-pad token's**
hidden state. Two rules:

1. **v0.1 (llama.cpp): do not pad at all.** Each pair is its own sequence, `pooling =
   last` per sequence, state reset between sequences (mandatory for the recurrent layers —
   a leaked DeltaNet state is a silently wrong answer, not a crash).
2. **Any padded batch (candle later): left-pad with 248044**, so the last position is
   always real text and pooling is `h[:, -1, :]` with no index arithmetic. Right-padding
   plus naive last-token pooling pools a pad embedding and produces plausible garbage —
   the classic cross-encoder bug. Left padding also requires the position ids / RoPE
   offsets to start after the padding, and the causal mask to exclude pads; a
   right-padded implementation with per-row `seq_len` gather is the only acceptable
   alternative and must be tested against the left-padded one.

**Operations** (all thin, all over `forward` + head):

| op | input | output |
|---|---|---|
| `predict(pairs)` | `[(premise, hypothesis)]` | per pair: 3 probs + argmax label |
| `rerank(question, options)` | 1×N | options sorted by `P(entailment)` (score = entailment prob, documented) |
| `grade(premise, hypothesis)` | 1 pair | scalar in [0,1] = `P(entailment)`, plus the full distribution |
| `latents(texts)` | N | the 2560-d pooled hidden states, f32 |

`rerank` and `grade` are *defined in terms of* `predict` — one label convention, one
softmax, one place to be wrong.

**Device — auto, with an explicit override.**

> **D3 — device is detected, not configured; configuration only overrides.**

Precedence: `--device` flag > `OPENJEV_DEVICE` env > config `runtime.device` > `auto`.
Values: `auto | cuda[:N] | metal | cpu`. `auto` probes in order **cuda → metal → cpu**,
keeping the first that (i) the host actually has (a real device handle, not a compile-time
feature), (ii) the selected backend supports, and (iii) has enough free VRAM for the
declared weight size + a margin. Fallback is a *chain*, and every demotion is logged at
warn with the reason:

```
warn: metal requested but backend 'candle' reports no metal kernel for qwen3_5 linear attention; falling back to cpu
```

An *explicit* device that fails is a hard error, never a silent demotion — if the user
said `--device cuda`, a CPU run at 40× the latency is not what they asked for.

**dtype** follows the device, and is also overridable (`--dtype`):
`cuda` → bf16 (f16 on pre-Ampere); `metal` → **f16, not bf16** (bf16 support on Metal is
uneven across candle ops and macOS versions — f16 is the safe default, bf16 is opt-in and
gated by a golden-logit check); `cpu` → f32 (bf16 on CPU is emulated and slow). On the
GGUF path the trunk dtype is the quantisation (Q5_K_M default), and the head matmul is
always f32 — 60 KB, so precision there is free.

**Startup line**, always, one line, stderr:

```
openjev 0.1.0  model=openjev-4b-nli-v2@<rev>  backend=llamacpp(b7xxx)  device=metal  dtype=q5_k_m/f32-head  ctx=8192  labels=3
```

**Memory, 4B, honest:** f16 ≈ 8.2 GB weights + activations ⇒ ~9–10 GB, fine on a 32 GB
Mac, tight on 16 GB. Q5_K_M ≈ 2.9 GB ⇒ ~4 GB resident, comfortable on 16 GB. Q4_K_M ≈ 2.4
GB if needed. The recurrent state is small (24 layers × 32 v-heads × 128 × 128 ≈ 100 MB at
f16) but **constant**, not proportional to context — which is the nice property of this
architecture and the reason long premises are cheap. Default ship: **Q5_K_M**, with Q8_0
published for anyone who wants to check quantisation drift.

## 6. Risk register

Ranked by *probability × damage*, each with a mitigation and an experiment that fits in an
hour.

**R1 — Quantisation moves the label.** A 3-way head on a 2560-d vector is far more
sensitive to trunk drift than next-token generation. Q4 could flip neutral↔entailment on
borderline pairs. *Mitigation:* publish Q8_0 and f16 alongside Q5_K_M; make dtype config;
gate the default on measured agreement. *Experiment (1h):* take 200 pairs from SNLI dev,
score under Q5_K_M vs f16, report label-agreement and max prob delta. Anything below ~98 %
agreement demotes the default to Q8_0.

**R2 — llama.cpp's `--embedding` path does not behave on a hybrid recurrent model.**
Pooling, state reset between sequences, or per-token hidden output could be wrong or
unimplemented for `qwen3_5`. This is the single assumption the whole v0.1 rests on.
*Mitigation:* the Rust head means we only need *hidden states*, the cheapest thing to ask
for; if `--embedding` misbehaves, the fallback is to read the final hidden state via the
C API directly rather than the server. *Experiment (1h, do this first):* pull an existing
unsloth Qwen3.5 GGUF, run `llama-embedding --pooling last`, confirm you get a non-zero
vector of the right width and that the *same* text twice gives bit-identical vectors
(proves state reset).

**R3 — The GGUF converter bug bites our checkpoint.** #27019 is exactly our tensors
(`in_proj_a/b`, `ssm_conv1d`). A bad conversion is silently wrong, not loud. *Mitigation:*
convert with PR #27132 applied; validate against golden logits before publishing; publish
only digest-pinned artefacts. *Experiment (1h):* convert, then compare our Rust-head
logits against Python `transformers` on 20 fixed pairs — agreement to 1e-2 on probs is the
ship gate. **This golden-logit fixture is the project's backbone and is worth building on
day one, because every later backend is graded against it.**

**R4 — No GGUF of this checkpoint exists, and we become its maintainer.** Upstream
retrains ⇒ we re-convert. *Mitigation:* a `scripts/convert.sh` in-repo, a documented
publish flow, and the revision pin so users are never surprised by an upstream push.
*Experiment:* none needed — this is a process cost, budget half a day per model revision.

**R5 — Static-linking llama.cpp with Metal into one binary is fiddlier than advertised.**
`llama-cpp-2`'s build.rs, cmake, and Metal shader embedding (`GGML_METAL_EMBED_LIBRARY`)
must all cooperate, or we ship a binary that needs a `.metallib` beside it — which breaks
the whole premise. *Mitigation:* prove it before writing product code. *Experiment (1h):*
hello-world crate, `cargo build --release`, `otool -L` shows no libggml/libllama, move the
binary to `/tmp` and run it there.

**R6 — Template / tokenizer drift.** Wrong newline, wrong tokenizer.json, or an extra BOS
and every number is quietly off. *Mitigation:* golden fixture (R3) covers tokens, not just
probs — assert the exact token id sequence for a known pair. *Experiment (30m):* tokenize
one pair in Python and in Rust, diff the id vectors.

**R7 — Padding/pooling bug when candle batching lands.** Right-padding + last-token
pooling is the classic silent failure. *Mitigation:* §5 rule; assert in tests that a
batch of N gives bit-comparable results to N single runs. *Experiment (1h, when
relevant):* batch of 3 with wildly different lengths vs 3 singles.

**R8 — Metal bf16/f16 numerics in candle.** Deferred with the candle port, but it is why
§5 defaults Metal to f16. *Experiment:* the R3 fixture again, on the candle backend.

**R9 — 4B on a 16 GB machine under `serve` with concurrency.** One model, one worker by
default; `batch.max_size` bounded; reject with 503 rather than swap. *Experiment (30m):*
hold 8 concurrent requests, watch RSS.

**R10 — Vision config tempts someone to load the ViT.** Cheap and already mitigated:
`modalities = ["text"]`, images are `NotSupported`, converter drops `visual.*` tensors.

## 7. What v0.1 is, concretely

`openjev serve` → axum on 127.0.0.1, `/v1/predict`, `/v1/rerank`, `/v1/grade`,
`/v1/latents`, `/healthz` (reports model, backend, device, dtype, ready). One static
binary. First run downloads ~3 GB after asking. Backend `llamacpp`, device `auto`, model
from the built-in registry. candle behind a feature flag, off, graded against R3's
fixture before it is ever the default.
