# openjev — Rust runtime survey

**Organised against one failure: shipping a Rust crate that runs fast on the owner's Mac
and nowhere else — or that runs everywhere until the model config changes and the crate
turns out to have hardcoded an architecture list.**

Survey date **2026-09-19**. Everything below was checked today against crates.io, docs.rs
and the GitHub API. Where something is unverified it says so.

Decision already taken upstream of this doc (see `01-inference-backend.md`): llama.cpp is
the **trunk only**. We pull last-token hidden states out of it and apply the 3-label
`Linear(2560->3)` head in Rust ourselves. So the requirement is **embedding extraction**,
not text generation. Every crate below is judged on that, not on how nice its chat API is.

## 0. Verified facts (do not re-derive)

- **llama.cpp runs Qwen3.5 natively, at runtime, today.** `src/llama-arch.cpp` registers
  `LLM_ARCH_QWEN35` → `"qwen35"` and `LLM_ARCH_QWEN35MOE` → `"qwen35moe"`;
  `src/models/models.h` defines `struct llama_model_qwen35 : llama_model_base` (line 2336)
  and `llama_model_qwen35moe` (2487). Landed in **PR #19435 "[Model] Qwen3.5 dense and MoE
  support (no vision)", merged 2026-02-08** — seven months of bake, not a fresh branch.
- **The broken piece is only the Python converter.** Issue
  [#27019](https://github.com/ggml-org/llama.cpp/issues/27019) — *open*, created 2026-08-13,
  last touched 2026-09-15, now labelled `stale`. `convert_hf_to_gguf.py` mishandles the
  `ssm_conv1d` kernel-dim reorder and fails to expand `in_proj_a`/`in_proj_b`
  (65536 → 131072). The reporter's own words: *"Runtime (llama-server) executes the
  existing unsloth-converted Qwen3.5 GGUF correctly."*
  PR [#27132](https://github.com/ggml-org/llama.cpp/pull/27132) is the fix — **open, still
  a draft, not merged** as of today. It notes `--no-mtp` is needed during conversion.
- **A community GGUF exists to test against today**: `unsloth/Qwen3.5-4B-GGUF` (full quant
  ladder, UD-IQ2 → BF16) and `unsloth/Qwen3.5-4B-MTP-GGUF`. Not our NLI checkpoint, but the
  same arch at the same size — enough to prove the whole Rust path end to end before the
  converter is fixed.
- **candle still has no Qwen3.5.** `huggingface/candle` `main`, `candle-transformers/src/models/`
  today lists `qwen2 · qwen2_moe · qwen3 · qwen3_moe · qwen3_vl/ · quantized_qwen2 ·
  quantized_qwen3 · quantized_qwen3_moe`. No `qwen3_5`, no gated-DeltaNet, no generic
  linear-attention block. The load-bearing fact holds. A candle port is greenfield kernel work.
- **llama.cpp ships prebuilt shared libraries for every target we care about.** Release
  `b11054` (2026-09-19, ~1–3 releases/day) carries 33 assets:
  `macos-arm64`, `macos-x64`, `ubuntu-x64`, `ubuntu-arm64`, `ubuntu-cuda-12.8-x64`,
  `ubuntu-cuda-13.3-{x64,arm64}`, `ubuntu-vulkan-{x64,arm64}`, `ubuntu-rocm-10.0-x64`,
  `ubuntu-sycl-*`, `win-cpu-*`, `win-cuda-*`, `android-arm64`.
  The macOS arm64 tarball is **11 MB** and contains `libllama.dylib`, `libggml{,-base,-cpu,
  -metal,-blas,-rpc}.dylib` with full SONAME symlink chains, plus CLI tools.
- **Two things the prebuilt tarball does NOT contain**: any `.h` header, and any
  `llama-embedding` binary. The embedding CLI is gone from current releases; the tools
  shipped are `llama-cli, llama-server, llama-completion, llama-bench, llama-tokenize,
  llama-quantize, llama-perplexity, llama-imatrix, …`. Pooled embeddings are still first
  class **via the C API and via `llama-server`'s `/embedding`** — `llama_context_params.pooling_type`
  takes `LLAMA_POOLING_TYPE_LAST`, which is exactly "embedding of the last token, for a
  causal model used as an encoder". That is our path; the missing CLI costs us nothing.
- Header absence is a non-issue: `include/llama.h` is a single stable public header we
  vendor from the matching tag. It is not generated.

## 1. The recommendation

**Build on `llama-cpp-2` (utilityai/llama-cpp-rs), compiled from its vendored submodule,
behind our own `Backend` trait — and do NOT chase the prebuilt-linking story in v1.**

`llama-cpp-2 0.1.156`, published **2026-09-02**. 134 versions, 1.31 M downloads, releases
roughly every 1–3 weeks through 2026 (0.1.152 Jul-21, .153 Jul-28, .154 Aug-05, .155
Aug-31, .156 Sep-02). MIT/Apache-2.0. Maintainer Marcus Dunn. This is the only Rust
llama.cpp binding with a 2026 release cadence that tracks upstream.

It gives us exactly what the trunk decision needs:

- `LlamaContext::embeddings_seq_ith(i) -> Result<&[f32], EmbeddingsError>` and
  `embeddings_ith(i)`. Docs are explicit: *"the size is the pooling-derived output width"*,
  and it errors when the context was built without embeddings enabled or with
  `LLAMA_POOLING_TYPE_NONE`. That is a real, documented, pooled-embedding API — not a
  by-product of a generation loop.
- `LlamaPoolingType` in `context::params`, settable on `LlamaContextParams`. `Last` is what
  we set; no fork, no patch.
- Backend features on `llama-cpp-sys-2`: `cuda`, `cuda-no-vmm`, `metal`, `vulkan`, `rocm`,
  `opencl`, `openmp`, `mkl`, plus `dynamic-link`, `dynamic-backends`, `system-ggml`,
  `system-ggml-static`, `static-stdcxx`. Every platform in the brief is a cargo feature,
  from one codebase. This is the whole reason it wins.
- **Architecture-agnostic by construction.** It is a thin safe wrapper over `llama_*`. It
  loads whatever GGUF you hand it; the arch table lives in llama.cpp, not in the crate. The
  crate's own README: *"does not attempt to create a stable API with all the Rust idioms,
  instead providing safe wrappers around nearly direct bindings."* Model repo, revision,
  subfolder, quant and future architecture all stay **config**. Nothing to disqualify.
- Its vendored `llama.cpp` submodule as of the current release post-dates the Feb-2026
  Qwen3.5 merge by seven months, so `qwen35` is in the box. **Verify this at first build**
  by loading `unsloth/Qwen3.5-4B-GGUF` — that is the gate, not this paragraph.

### The cost, stated honestly

`llama-cpp-sys-2/build.rs` is 1459 lines of cmake + bindgen driving
`llama-cpp-sys-2/llama.cpp` (a real submodule, pinned). **There is no
"link against an already-built libllama" escape hatch.** `system-ggml` /
`system-ggml-static` substitute *ggml only*; `dynamic-link` merely flips
`BUILD_SHARED_LIBS` on the build it is still performing. Knobs are
`LLAMA_LIB_PROFILE`, `LLAMA_BUILD_SHARED_LIBS`, `LLAMA_STATIC_CRT`,
`CMAKE_BUILD_PARALLEL_LEVEL`.

So we pay:

- **A cmake + C++ toolchain on every build machine**, and a cold build of llama.cpp
  (minutes on a laptop; substantially worse on a CUDA image, where nvcc compiles the ggml
  CUDA kernels). Warm rebuilds are cached in `OUT_DIR` and cheap.
- **CI matrix per backend.** `cargo test --features cuda` needs the CUDA toolkit present.
  This is the real tax and it is the price of the feature-flag portability we are buying.
- The `common` default feature pulls ~14 MB of JSON-schema grammar helper we do not need.
  Turn it off; we neither sample nor constrain.

We take that cost because the alternative — hand-rolling FFI to get prebuilt linking — buys
a nicer build story and hands back the safe wrapper, the embedding API, the pooling enum,
the batch/KV plumbing and the upstream tracking. That is a bad trade in v1.

### Organised against: the crate going stale

`llama-cpp-2` is one maintainer. If it stops tracking upstream, we must not be rewritten.
**Therefore: `openjev-core` defines its own `trait Trunk { fn hidden_last(&mut self, tokens)
-> Vec<f32> }` and `llama-cpp-2` lives behind exactly one module implementing it.** Nothing
else in the crate may name a `llama_cpp_2::` type. The escape hatch (§4) then costs days,
not a rewrite. This is non-negotiable and belongs in the build contract.

## 2. Runner-up and the rest

| candidate | verdict |
|---|---|
| **Direct FFI over `llama.h` + upstream prebuilts** | **Runner-up.** The only real contender. See §4. |
| `mistral.rs` | Live (7.7 k ★, pushed 2026-09-08, 384 open issues), genuinely has Qwen3.5 + gated-DeltaNet work (recent PRs name "GDN rollback", and the README's own examples use `unsloth/Qwen3.5-4B-GGUF`), and advertises *"embeddings in one engine."* **Loses on portability and on shape.** README claims *Metal on Apple Silicon; per-GPU CUDA or CPU on Linux; CPU on Windows* — **no Vulkan, no ROCm** anywhere in the docs. And it is an *engine*, not a library: a model-registry/dispatch layer we would have to get our architecture accepted into, with a heavy candle-kernel build. Its embedding path is embedding-*model* shaped, not "give me the last hidden state of an arbitrary causal trunk" — unverified whether a generic hidden-state hook exists at all, and depending on it being added is precisely the disqualifier in the brief. |
| `llama_cpp` / `llama_cpp-rs` (edgenai) | Higher-level, async, nicer API. Dead for our purposes: no 2026 release cadence, and a high-level generation-shaped API is the wrong shape for hidden-state extraction. |
| `llama-cpp-4`, `delysis/llama-cpp-rs` | Forks/mirrors of `llama-cpp-2` with the same design and less traffic. Take the original. |
| `rust-llama.cpp` (mdrokz) | 2023-era. Ignore. |
| `drama_llama` | No signal in 2026. Treat as dead. |
| `candle` | No `qwen3_5`, no gated-DeltaNet, not in `main` (re-verified above). Would mean writing the delta-rule + short-conv kernels for Metal *and* CUDA ourselves. That is the research project we decided not to do. |
| `ort` (ONNX Runtime) | Dead end at the format layer, not the crate layer: there is no ONNX operator set for non-softmax attention, so the hybrid recurrence has to export as dynamic-shape soup. Excellent crate, wrong graph. |
| `burn` | Real multi-backend story (wgpu/Vulkan, CUDA, Metal) and the most architecturally attractive answer on paper. Loses for the same reason as candle plus one more: no Qwen3.5, *and* no GGUF quant ecosystem. We would be porting weights and writing kernels. |
| wgpu / Vulkan runtime from scratch | Not a candidate. Named only to close the door. |

## 3. Does the owner's JVM work port?

`mochallama` (`core/.../panama/{LlamaBridge,NativeLoader,ChatEngine}.java`) is the right
*pattern* and zero reusable *code*.

The reusable idea, from `NativeLoader`: stage **every** file in each library's SONAME
symlink chain (`libggml.dylib`, `libggml.0.dylib`, `libggml.0.24.0.dylib`) side by side,
then `System.load` them in strict dependency order —
`ggml-base → ggml-cpu → ggml-blas → ggml → llama → llama-common → llamabridge` — so
`@rpath`/`$ORIGIN` resolves siblings with no `DYLD_LIBRARY_PATH` hackery. The upstream
tarball inspected above ships exactly that symlink chain, which confirms the approach.

What does not port: `LlamaBridge` is a Panama FFM downcall layer against a hand-written
`extern "C"` shim, solving a problem Rust does not have (Rust calls the C API directly;
bindgen or a hand-written `extern "C"` block is a day's work, not a bridge library). And
mochallama is generation-shaped — chat, sampling, tool calls — where we need one forward
pass and a hidden vector. `muthuishere/llama-bindings` is the same story a layer lower.

**Verdict: read `NativeLoader` for the load-order + rpath discipline if and when we do §4.
Port nothing.**

## 4. The escape hatch: direct FFI + upstream prebuilts

Worth writing down precisely, because it is the runner-up and it is where we go if the
build tax bites or `llama-cpp-2` stalls.

Shape: vendor `include/llama.h` from the matching `bXXXXX` tag, run `bindgen` over it in
`build.rs` (headers only — no cmake, no C++ compiler), `println!("cargo:rustc-link-lib=dylib=llama")`
against a `libs/<platform>/` directory populated at *install* time by a downloader that
picks the right tarball from the `b<build>` release, mirroring `NativeLoader`'s
platform-key → asset mapping.

**Buys:** no C++ toolchain, no cmake, no nvcc anywhere; seconds not minutes; a CI matrix
that is just "download a different tarball"; backend choice becomes a *runtime* download
decision rather than a compile-time cargo feature — which is strictly better for shipping
one binary that finds CUDA if it is there. Upstream cuts 1–3 releases a day, so the artifact
supply is not a risk.

**Costs, and they are why it is not v1:**
- We own the safety layer. `llama-cpp-2`'s `EmbeddingsError`, lifetime-bound
  `&[f32]` slices into context memory, `!Send`/`!Sync` markers, batch and KV-cache
  invariants — all of that is real work we would redo and get subtly wrong.
- We own upstream churn. llama.cpp does not promise ABI stability; `llama-cpp-2`
  absorbing that every two weeks is the product we are buying.
- macOS Metal shipping: `libggml-metal.dylib` needs its shader resources found at runtime;
  also codesigning/notarisation for six-plus staged dylibs. A submodule build sidesteps this.
- Version skew becomes ours: the pinned header must match the downloaded tarball exactly, or
  we get silent struct-layout corruption rather than a link error.

**Do this in v2 if and only if** the CUDA CI build time or the toolchain requirement is
measured to actually hurt. Not on aesthetics.

## 5. Expected performance — ESTIMATE, not measured

An NLI pair is **prefill-only**: ~40–120 tokens in, one forward pass, read the last hidden
state. No decode, no KV growth, no sampling. That is the cheapest possible shape and it is
throughput-bound on prefill, not latency-bound on token generation. Two structural bonuses:
the 2560×248320 `lm_head` is never computed in embedding mode (~0.64 GFLOP/token saved,
~8% of the model), and the 24 linear-attention layers carry a fixed-size recurrent state
instead of a growing KV cache.

Rough arithmetic: ~2·4e9 ≈ **8 GFLOP per token** at dense 4B, minus the head.
Assume Q4_K_M (~2.5 GB weights) and ~80 tokens per pair, batched into 1024–2048-token
prefill chunks. Then apply a **1.5–2× haircut**: ggml's gated-DeltaNet/SSM kernels are far
newer and less tuned than its attention path, and the 24:8 split means most layers run the
less-optimised code.

| target | prefill tok/s (est.) | pairs/s (est.) | single-pair latency (est.) |
|---|---|---|---|
| M-series Metal (M3/M4 Pro class) | 600–1 400 | **8–18** | 80–200 ms |
| CUDA (4090 / L40S class) | 5 000–12 000 | **60–150** | 20–60 ms |
| CPU only (10–16 perf cores) | 80–250 | **1–3** | 350–1 000 ms |

Read these as order-of-magnitude. The numbers that matter are the ratios: **CUDA ≈ 8× Metal
≈ 60× CPU**, and CPU-only is usable for interactive single-pair work but not for bulk
scoring. Batch aggressively — at 80 tokens a pair, a single 2048-token prefill batch swallows
~25 pairs, and per-call overhead dominates below that.

**Success check before any of this is believed:** `unsloth/Qwen3.5-4B-GGUF` loaded through
`llama-cpp-2` with `pooling_type = Last` and `embeddings = true`, returning a 2560-wide
`f32` slice from `embeddings_seq_ith(0)`, on Metal and on CPU, with `llama-bench` prefill
numbers from the same build alongside. If that returns 2560 floats, the trunk decision is
proven and everything above is engineering.

## 6. What would flip this decision

**One fact: if `llama-cpp-2`'s vendored llama.cpp turns out to predate PR #19435, or if
`embeddings_seq_ith` returns garbage / the wrong width on a `qwen35` GGUF** — i.e. if
llama.cpp's Qwen3.5 graph does not actually feed the pooling path — then llama-cpp-2 loses
and we go to §4 against a *hand-picked recent tarball*, because the prebuilt path lets us
pin any `bXXXXX` we like while a submodule crate pins what its maintainer chose.

The converter bug (#27019 / #27132) is **not** decision-flipping. It sits in Python, at
model-preparation time, outside the runtime and outside the product. Worst case we convert
our NLI checkpoint by applying the draft PR's two tensor-layout fixes locally, once,
offline, with `--no-mtp` — and ship a GGUF. No Python at runtime, as promised.
