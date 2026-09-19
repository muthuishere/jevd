# 0002 — `llama-cpp-2`, compiled from its vendored submodule, over direct FFI against upstream prebuilt libraries

Status: **accepted**, and it **reverses an instruction in the build brief**. Amends
`docs/design/01-inference-backend.md` §1 and R5.

## What we wanted

The brief was explicit, and it was the right instinct: *consume upstream prebuilt
llama.cpp release libraries and auto-resolve the right native for the platform — do not
make the user compile llama.cpp.* This mirrors the owner's JVM engine `mochallama`, which
stages a per-platform dylib closure (`libggml-base`, `libggml-cpu`, `libggml-metal`,
`libllama`, …) and dlopens it in dependency order, so a user installs a jar and nothing
else.

The upstream material genuinely exists. Release `b11054` carries **33 assets** —
`macos-arm64` (11 MB, Metal shaders embedded, no stray `.metallib`), `macos-x64`,
`ubuntu-{x64,arm64}`, `ubuntu-cuda-{12.8,13.3}`, `ubuntu-vulkan-*`, `ubuntu-rocm-*`,
`win-cuda-*`, `android-arm64` — at 1–3 releases per day, each dylib carrying its full
SONAME symlink chain and an `@loader_path` rpath, which is exactly what makes
stage-into-one-directory-and-dlopen work. Verified by download, `otool -L` and `otool -l`.

## Why we are not doing it

`llama-cpp-sys-2`'s `build.rs` is ~1459 lines of cmake + bindgen over a **pinned
submodule**, and it has **no escape hatch for linking an existing `libllama`**.
`system-ggml` covers ggml only; `dynamic-link` merely flips `BUILD_SHARED_LIBS` on the
build it is already doing. There is no supported path from this crate to a prebuilt
`libllama`.

The alternative — hand-written `extern "C"` over `llama.h` against the prebuilt assets —
was costed, not hand-waved. It fails on **ABI surface, not on concept**. `llama.h` at
b11054 is 1646 lines, and the two structs we must populate are
`llama_model_params` (13 fields, 6 trailing bools) and `llama_context_params`
(**32 fields**, including 5 enums and 2 function pointers). We would be mirroring those
layouts by hand, per release, against a project that ships 1–3 releases a day and has
already renamed and reordered these fields repeatedly. A layout drift does not fail to
link. It writes `n_gpu_layers` into `main_gpu`, and the failure surfaces as a slow run or
a wrong number. On top of that we would own the safety layer, Metal shader codesigning,
and version skew between the resolver and the mirror.

Owning a 32-field C struct mirror to avoid a build-time compiler is a bad trade, and it is
the same trade `mochallama` **also declined** — worth saying plainly, because the brief
cited it as precedent for the opposite. mochallama does not consume upstream prebuilts
either: its `natives.yml` is a *tier-1 workflow that compiles llama.cpp from a pinned tag
(`b9371`) in CI* across four platforms and stores the closure in a durable GitHub Release,
which tier 2 then downloads. The property the owner actually wanted — *the user never
compiles anything* — is delivered by **building once in CI and shipping the artefact**,
not by linking someone else's.

## Decision

**v0.1 builds on `llama-cpp-2` 0.1.156, compiled from its vendored submodule, behind our
own one-method `Backend` trait.** Every GPU backend is a cargo feature pass-through
(`metal`, `cuda`, `vulkan`), so CPU + Metal + CUDA come from one codebase.

Verified before committing to it, because the whole plan dies if it is false: the
llama.cpp vendored by `llama-cpp-sys-2` 0.1.156 **has Qwen3.5**. `src/llama-arch.cpp`
registers `LLM_ARCH_QWEN35` → `"qwen35"` and `LLM_ARCH_QWEN35MOE`; `src/models/models.h`
defines `llama_model_qwen35` (line 2153); `src/models/qwen35.cpp` exists. That is PR
\#19435, merged 2026-02-08 — seven months of bake. Issue \#27019 and draft PR \#27132
concern `convert_hf_to_gguf.py` **only**, and conversion is a publish-time step, so no
Python ships.

## What it costs, stated plainly

- **A C++ toolchain and cmake are build-time requirements.** For a user installing from
  crates.io, that is a real regression against "download a binary".
- **A per-backend build matrix.** `metal`, `cuda` and `vulkan` are separate compiles, so
  release engineering grows a dimension.
- **First build is slow** (minutes, not seconds) and is charged to CI on every matrix leg.

The mitigation is the same one mochallama uses, and it is a packaging job rather than a
code job: **compile once per platform in CI, ship binaries and release artefacts**, so the
cost lands on us and never on a user. Nothing in this ADR obliges anyone to `cargo install`.

## What contains the cost

The `Backend` trait is one method — `forward(batch) -> last hidden state`. Everything the
rest of the crate does (`predict`, `rerank`, `grade`, `latents`, the head, the label map,
the template, pooling) sits above it and knows no backend name. Replacing the llama.cpp
integration is one file, `backends/llamacpp.rs`, plus a cargo feature. That is the
insurance premium this ADR is paying for.

## What would let us revisit

- `llama-cpp-sys-2` gaining a supported "link against this `libllama`" mode.
- Upstream publishing a stable C ABI with an opaque-params builder (`llama_*_params_set_*`
  accessors), which would end the struct-mirror problem outright.
- Our own thin `extern "C"` bridge compiled once in CI — the mochallama shape — if the
  build-time toolchain requirement turns out to hurt more than expected.
