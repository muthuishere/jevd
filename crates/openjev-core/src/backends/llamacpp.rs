//! llama.cpp as the **trunk only**: tokens in, last-token hidden state out. The 3-label
//! classification head is applied in Rust (see [`crate::head`]).
//!
//! This deletes the whole class of risk around GGUF's `cls.output.weight`,
//! `pooling_type = RANK` and `classifier.output_labels` — a path built and tested for
//! yes/no rerankers on dense-attention models, not a 3-way head on a hybrid recurrent
//! trunk. We ask llama.cpp for the cheapest thing it can give us and own the rest.
//!
//! **Per-sequence state reset.** The 24 linear-attention layers of a Qwen3.5 hybrid carry
//! a gated-DeltaNet recurrent state. A leak across sequences is a silently wrong label,
//! not a crash. Measured behaviour (docs/adr/0001): llama.cpp already resets correctly —
//! on CPU the same text after a 1400-word distractor is *bit-identical* to the same text
//! alone. We do not rely on that alone: v0.1 forwards one sequence per context generation
//! and clears memory between them, so there is no state to leak by construction.

use crate::backend::{
    Backend, BackendFactory, BackendInfo, Caps, EncodedInput, Hidden, OpenRequest,
};
use crate::device::{Device, Dtype};
use crate::error::{JevError, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::sync::{Mutex, OnceLock};

pub const NAME: &str = "llamacpp";

/// Capability tokens this backend advertises, matched against a model entry's
/// `requires`. `qwen3_5-hybrid` is here because llama.cpp registers `LLM_ARCH_QWEN35`
/// and ships `src/models/qwen35.cpp` — verified against the source vendored by
/// llama-cpp-sys-2 0.1.156, not assumed.
const PROVIDES: &[&str] = &["qwen3_5-hybrid", "gguf", "quantised-trunk"];

/// llama.cpp and ggml both write to stderr by default, and **core never prints** — a
/// library that owns stderr is a library you cannot embed. Both log sinks are redirected
/// into `tracing`, so the diagnostics survive at `debug` instead of being thrown away
/// (`void_logs`) or corrupting a caller's output.
///
/// `llama_log_set` covers llama.cpp; `ggml_log_set` is a *separate* sink and is where the
/// Metal/CUDA device banner comes from. Setting only the first leaves ggml shouting.
unsafe extern "C" fn log_to_tracing(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: llama.cpp guarantees a NUL-terminated C string for the lifetime of the call.
    let msg = unsafe { std::ffi::CStr::from_ptr(text) };
    let msg = msg.to_string_lossy();
    let msg = msg.trim_end();
    if msg.is_empty() {
        return;
    }
    match level {
        llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR => tracing::error!(target: "llama.cpp", "{msg}"),
        llama_cpp_sys_2::GGML_LOG_LEVEL_WARN => tracing::warn!(target: "llama.cpp", "{msg}"),
        llama_cpp_sys_2::GGML_LOG_LEVEL_INFO => tracing::debug!(target: "llama.cpp", "{msg}"),
        _ => tracing::trace!(target: "llama.cpp", "{msg}"),
    }
}

/// `llama_backend_init` is process-global and must happen exactly once. A second init is
/// not merely wasteful — the ggml backend registry is global mutable state.
fn global_backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            // Installed before init so the device-probe banner is captured too.
            // SAFETY: both take a plain C callback and a user-data pointer we leave null.
            unsafe {
                llama_cpp_sys_2::llama_log_set(Some(log_to_tracing), std::ptr::null_mut());
                llama_cpp_sys_2::ggml_log_set(Some(log_to_tracing), std::ptr::null_mut());
            }
            let b = LlamaBackend::init().map_err(|e| e.to_string())?;
            unsafe {
                llama_cpp_sys_2::llama_log_set(Some(log_to_tracing), std::ptr::null_mut());
                llama_cpp_sys_2::ggml_log_set(Some(log_to_tracing), std::ptr::null_mut());
            }
            Ok(b)
        })
        .as_ref()
        .map_err(|e| JevError::Native(format!("llama_backend_init failed: {e}")))
}

/// Serialises model load + context creation across every backend instance in the
/// process. See `LlamaCppBackend::open`.
static INIT_LOCK: Mutex<()> = Mutex::new(());

pub struct Factory;

pub fn factory() -> &'static dyn BackendFactory {
    &Factory
}

/// Which GPU backends were actually compiled into this binary.
///
/// On macOS/aarch64 `llama-cpp-2` enables Metal by target dependency, so it is present
/// without our `metal` feature. Everything else is an explicit opt-in, and asking for one
/// that is absent must name the cargo feature — a silent CPU run at 40x the latency is
/// not what the user asked for.
pub fn compiled_devices() -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    if cfg!(feature = "metal") || cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        v.push(Device::Metal);
    }
    if cfg!(feature = "cuda") {
        v.push(Device::Cuda(0));
    }
    if cfg!(feature = "vulkan") {
        v.push(Device::Vulkan);
    }
    v
}

/// The cargo feature that would add a device, for the error message.
pub fn feature_for(device: Device) -> Option<&'static str> {
    match device {
        Device::Cpu => None,
        Device::Metal => Some("metal"),
        Device::Cuda(_) => Some("cuda"),
        Device::Vulkan => Some("vulkan"),
    }
}

impl BackendFactory for Factory {
    fn name(&self) -> &'static str {
        NAME
    }

    fn provides(&self) -> &'static [&'static str] {
        PROVIDES
    }

    fn supports_device(&self, device: Device) -> bool {
        compiled_devices().iter().any(|d| d.kind() == device.kind())
    }

    fn open(&self, req: &OpenRequest) -> Result<Box<dyn Backend>> {
        if !self.supports_device(req.device) {
            let hint = feature_for(req.device)
                .map(|f| format!(" (build with --features {f})"))
                .unwrap_or_default();
            return Err(JevError::Device(format!(
                "backend 'llamacpp' in this binary has no kernel for device '{}'{hint}",
                req.device
            )));
        }
        Ok(Box::new(LlamaCppBackend::open(req)?))
    }
}

/// Owns the model and a context borrowed from it.
///
/// `ctx` borrows `model`. The lifetime is erased to `'static` because the two are owned
/// together and `model` is behind a `Box` (stable address). Soundness rests on two
/// things, both enforced here: the `Box` is never moved out of or reallocated, and `ctx`
/// is declared *before* `model` so Rust drops it first. Reordering these fields is a
/// use-after-free.
struct LlamaCppBackend {
    inner: Mutex<Inner>,
    info: BackendInfo,
}

struct Inner {
    ctx: LlamaContext<'static>,
    #[allow(dead_code)]
    model: Box<LlamaModel>,
    /// Monotonic sequence id. A fresh id per forward, on top of the memory clear, so a
    /// stale recurrent state has no id to be found under.
    next_seq: i32,
    /// Per-sequence context limit. llama.cpp's `n_ctx` is the whole KV budget shared
    /// across `n_seq_max` sequences, so the limit one input must respect is that divided
    /// by the sequence count — not the number the operator typed.
    n_ctx: u32,
    max_seqs: usize,
}

// SAFETY: every access to `ctx` goes through the `Mutex`, which is the synchronisation
// llama.cpp's context requires (it is not internally synchronised for concurrent decode).
unsafe impl Send for Inner {}

impl LlamaCppBackend {
    fn open(req: &OpenRequest) -> Result<Self> {
        let backend = global_backend()?;

        // Model load and context creation are serialised process-wide.
        //
        // Measured, not assumed: five `LlamaCppBackend`s opened concurrently in one
        // process produce an **all-zero hidden state** from one of them — no error, no
        // panic, just a silently dead vector, which the head would happily turn into a
        // confident label. ggml's device/backend setup carries global state that
        // `llama_backend_init` alone does not make re-entrant. This lock costs nothing
        // at steady state (it is boot-only) and removes the entire failure class.
        let _boot = INIT_LOCK
            .lock()
            .map_err(|_| JevError::backend(NAME, anyhow::anyhow!("init lock poisoned")))?;

        let n_gpu_layers = if req.device.is_gpu() { u32::MAX } else { 0 };
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

        let model = Box::new(
            LlamaModel::load_from_file(backend, &req.weights, &model_params).map_err(|e| {
                JevError::backend(
                    NAME,
                    anyhow::anyhow!("loading {}: {e}", req.weights.display()),
                )
            })?,
        );

        let n_embd = usize::try_from(model.n_embd())
            .map_err(|_| JevError::backend(NAME, anyhow::anyhow!("negative n_embd")))?;
        if n_embd != req.spec.hidden_size {
            return Err(JevError::backend(
                NAME,
                anyhow::anyhow!(
                    "weights have hidden size {n_embd} but the registry says {} — the head \
                     would read the wrong width",
                    req.spec.hidden_size
                ),
            ));
        }

        let n_ctx = u32::try_from(req.context).unwrap_or(u32::MAX);
        let threads = req
            .n_threads
            .and_then(|t| i32::try_from(t).ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| i32::try_from(n.get()).unwrap_or(4))
                    .unwrap_or(4)
            });

        let max_seqs = req.max_seqs();
        // `context` is the TOTAL KV budget, shared across `max_seqs` sequences — which is
        // what llama.cpp's `n_ctx` means. Scaling it *up* with the sequence count instead
        // is how the first attempt at this asked for 131072 tokens of context and
        // segfaulted; memory here must stay flat as batching grows, and the per-sequence
        // limit falls out as the quotient.
        let per_seq = (n_ctx / u32::try_from(max_seqs).unwrap_or(1)).max(1);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            // n_batch/n_ubatch bound the tokens in one decode. Sized to the whole context
            // so a coalesced group is one graph rather than several — which is the entire
            // point of batching on a GPU this underfed.
            .with_n_batch(n_ctx)
            .with_n_ubatch(n_ctx)
            .with_n_seq_max(u32::try_from(max_seqs).unwrap_or(1))
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            .with_embeddings(true)
            // Last-token pooling. The head reads the last *non-pad* token, and on this
            // path nothing is padded, so "last" is exactly right. The integration test
            // proves which token it read rather than trusting the name.
            .with_pooling_type(LlamaPoolingType::Last);

        let ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| JevError::backend(NAME, anyhow::anyhow!("creating context: {e}")))?;

        // SAFETY: see the struct doc. `model` is boxed and outlives `ctx` by field order.
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };

        let info = BackendInfo {
            name: NAME,
            version: concat!("llama-cpp-2 ", env!("CARGO_PKG_VERSION")).to_string(),
            device: req.device,
            dtype: req.dtype,
            context: req.context,
            hidden_size: n_embd,
            // No IMAGES: v0.1 is text-only and says so, rather than emitting image
            // placeholder tokens the trunk would answer wrongly about.
            // BATCH only when the context was actually built for more than one sequence.
            // Declaring it otherwise would make the Engine hand us batches we would then
            // silently serialise, and the throughput table would stop meaning anything.
            caps: if max_seqs > 1 {
                Caps::LATENTS | Caps::BATCH
            } else {
                Caps::LATENTS
            },
        };

        Ok(Self {
            inner: Mutex::new(Inner {
                ctx,
                model,
                next_seq: 0,
                n_ctx: per_seq,
                max_seqs,
            }),
            info,
        })
    }
}

impl Backend for LlamaCppBackend {
    fn describe(&self) -> BackendInfo {
        self.info.clone()
    }

    fn forward(&self, batch: &[EncodedInput]) -> Result<Vec<Hidden>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JevError::backend(NAME, anyhow::anyhow!("context mutex poisoned")))?;

        let per_seq_limit = inner.n_ctx as usize;
        for input in batch {
            if input.tokens.len() > per_seq_limit {
                return Err(JevError::ContextOverflow {
                    tokens: input.tokens.len(),
                    limit: per_seq_limit,
                });
            }
        }

        let max_seqs = inner.max_seqs;
        let mut out = Vec::with_capacity(batch.len());

        // One decode per group of at most `max_seqs` sequences. At max_seqs = 1 this is
        // exactly the old one-sequence-per-decode path, byte for byte.
        for group in batch.chunks(max_seqs) {
            // Full memory clear before every group. This is what makes a leaked
            // gated-DeltaNet state unrepresentable rather than merely unlikely: no
            // sequence in this group can see anything from a previous group, and within
            // the group each sequence carries its own recurrent state under its own
            // seq_id. That llama.cpp really keeps those states separate is not taken on
            // faith — `tests/golden.rs` forwards a mixed-length batch and asserts every
            // pair gets the answer it gets alone.
            inner.ctx.clear_kv_cache();
            let seq_base = inner.next_seq;
            inner.next_seq = inner
                .next_seq
                .wrapping_add(i32::try_from(group.len()).unwrap_or(1))
                .max(0);

            let total_tokens: usize = group.iter().map(|i| i.tokens.len()).sum();
            // Second argument is seq-ids *per token*, not sequences per batch: every
            // token here belongs to exactly one sequence.
            let mut lbatch = LlamaBatch::new(total_tokens, 1);

            for (s, input) in group.iter().enumerate() {
                let seq_id = i32::try_from(s)
                    .map_err(|_| JevError::backend(NAME, anyhow::anyhow!("sequence overflow")))?;
                for (pos, &tok) in input.tokens.iter().enumerate() {
                    let pos = i32::try_from(pos).map_err(|_| {
                        JevError::backend(NAME, anyhow::anyhow!("position overflow"))
                    })?;
                    // `logits = true` only on the pooled position. With pooling = LAST
                    // llama.cpp needs the sequence's output flagged, and flagging every
                    // token would allocate an output buffer per token for nothing.
                    let last = pos as usize == input.pool_index;
                    lbatch
                        .add(LlamaToken(tok as i32), pos, &[seq_id], last)
                        .map_err(|e| JevError::backend(NAME, anyhow::anyhow!("batch add: {e}")))?;
                }
            }

            inner
                .ctx
                .decode(&mut lbatch)
                .map_err(|e| JevError::backend(NAME, anyhow::anyhow!("decode: {e}")))?;

            for s in 0..group.len() {
                let seq_id = i32::try_from(s)
                    .map_err(|_| JevError::backend(NAME, anyhow::anyhow!("sequence overflow")))?;
                let embd = inner.ctx.embeddings_seq_ith(seq_id).map_err(|e| {
                    JevError::backend(NAME, anyhow::anyhow!("embeddings seq {seq_id}: {e}"))
                })?;

                if embd.len() != self.info.hidden_size {
                    return Err(JevError::backend(
                        NAME,
                        anyhow::anyhow!(
                            "pooled state is {} wide, expected {}",
                            embd.len(),
                            self.info.hidden_size
                        ),
                    ));
                }
                // An all-zero or non-finite pooled state is the observed shape of ggml
                // corruption (see the init lock above): no error is raised, the vector is
                // just dead, and the head would turn it into a uniform-but-confident
                // label. Refusing here is the difference between a loud failure and a
                // wrong answer. Batching makes this check matter more, not less — a
                // sequence whose slot was never filled reads as exactly this.
                if !embd.iter().any(|v| *v != 0.0) {
                    return Err(JevError::backend(
                        NAME,
                        anyhow::anyhow!(
                            "pooled hidden state for sequence {seq_id} is all zeros — the \
                             forward pass produced nothing"
                        ),
                    ));
                }
                if !embd.iter().all(|v| v.is_finite()) {
                    return Err(JevError::backend(
                        NAME,
                        anyhow::anyhow!("pooled hidden state contains NaN or infinity"),
                    ));
                }
                out.push(Hidden(embd.to_vec()));
            }
            let _ = seq_base;
        }
        Ok(out)
    }
}

/// A label for the trunk dtype on this path. GGUF's trunk dtype is its quantisation, and
/// the registry carries it as data; this is only the fallback for an entry that omits it.
pub fn dtype_label(spec_label: Option<&str>) -> Dtype {
    match spec_label {
        Some("q4_k_m") => Dtype::Quant("q4_k_m"),
        Some("q5_k_m") => Dtype::Quant("q5_k_m"),
        Some("q8_0") => Dtype::Quant("q8_0"),
        Some("f16") => Dtype::F16,
        Some("bf16") => Dtype::Bf16,
        _ => Dtype::Quant("gguf"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_compiled_in() {
        assert!(compiled_devices().contains(&Device::Cpu));
    }

    #[test]
    fn metal_is_present_on_apple_silicon_without_an_explicit_feature() {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert!(Factory.supports_device(Device::Metal));
        }
    }

    #[test]
    fn an_uncompiled_device_names_the_cargo_feature() {
        if !cfg!(feature = "cuda") {
            assert_eq!(feature_for(Device::Cuda(0)), Some("cuda"));
            assert!(!Factory.supports_device(Device::Cuda(0)));
        }
    }

    #[test]
    fn advertises_the_capability_the_registry_requires() {
        assert!(Factory.provides().contains(&"qwen3_5-hybrid"));
    }

    #[test]
    fn v01_declares_no_batch_and_no_images() {
        // Both are load-bearing: !BATCH is why nothing pads on this path, and !IMAGES is
        // why an image request is a 501 instead of a confident wrong label.
        let caps = Caps::LATENTS;
        assert!(!caps.contains(Caps::BATCH));
        assert!(!caps.contains(Caps::IMAGES));
    }
}
