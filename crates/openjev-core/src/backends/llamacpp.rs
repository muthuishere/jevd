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
    n_ctx: u32,
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

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            // One sequence per generation: v0.1 declares !Caps::BATCH, so nothing pads and
            // nothing shares a context with anything else.
            .with_n_batch(n_ctx)
            .with_n_ubatch(n_ctx)
            .with_n_seq_max(1)
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
            // No BATCH: one sequence per forward, which is also why nothing pads.
            caps: Caps::LATENTS,
        };

        Ok(Self {
            inner: Mutex::new(Inner {
                ctx,
                model,
                next_seq: 0,
                n_ctx,
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

        let mut out = Vec::with_capacity(batch.len());
        for input in batch {
            if input.tokens.len() > inner.n_ctx as usize {
                return Err(JevError::ContextOverflow {
                    tokens: input.tokens.len(),
                    limit: inner.n_ctx as usize,
                });
            }
            // Full memory clear between sequences. This is what makes a leaked
            // gated-DeltaNet state unrepresentable rather than merely unlikely.
            inner.ctx.clear_kv_cache();
            let seq = inner.next_seq;
            inner.next_seq = inner.next_seq.wrapping_add(1).max(0);

            let mut lbatch = LlamaBatch::new(input.tokens.len(), 1);
            for (pos, &tok) in input.tokens.iter().enumerate() {
                let pos = i32::try_from(pos)
                    .map_err(|_| JevError::backend(NAME, anyhow::anyhow!("position overflow")))?;
                let last = pos as usize == input.pool_index;
                lbatch
                    .add(LlamaToken(tok as i32), pos, &[0], last)
                    .map_err(|e| JevError::backend(NAME, anyhow::anyhow!("batch add: {e}")))?;
            }

            inner
                .ctx
                .decode(&mut lbatch)
                .map_err(|e| JevError::backend(NAME, anyhow::anyhow!("decode: {e}")))?;

            let embd = inner
                .ctx
                .embeddings_seq_ith(0)
                .map_err(|e| JevError::backend(NAME, anyhow::anyhow!("embeddings: {e}")))?;

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
            // just dead, and the head would turn it into a uniform-but-confident label.
            // Refusing here is the difference between a loud failure and a wrong answer.
            if !embd.iter().any(|v| *v != 0.0) {
                return Err(JevError::backend(
                    NAME,
                    anyhow::anyhow!(
                        "pooled hidden state is all zeros — the forward pass produced nothing"
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
            let _ = seq;
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
