//! The backend contract.
//!
//! The whole contract is `forward(batch) -> last hidden state`. `predict`, `rerank`,
//! `grade` and `latents` are free functions in [`crate::ops`], built on `forward` plus the
//! classification head.
//!
//! Organised against: two backends disagreeing about what "entailment probability" means.
//! A backend that implements one method gets all four operations, and cannot implement
//! them inconsistently.

use crate::device::{Device, Dtype};
use crate::error::Result;
use crate::registry::ModelSpec;
use std::path::PathBuf;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Caps: u32 {
        /// Can return pooled hidden states to the caller (`/v1/latents`).
        const LATENTS = 1 << 0;
        /// Can forward more than one sequence per `forward` call with a real throughput win.
        const BATCH   = 1 << 1;
        /// Accepts image placeholder tokens. v0.1: nobody does.
        const IMAGES  = 1 << 2;
    }
}

/// One already-tokenised sequence. Backends never see text — one tokenizer for every
/// backend, so a backend swap cannot move the decision boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedInput {
    pub tokens: Vec<u32>,
    /// Index of the token whose hidden state the head reads. For unpadded input this is
    /// `tokens.len() - 1`; for a left-padded batch it is still the last index, which is
    /// the entire point of left padding.
    pub pool_index: usize,
}

impl EncodedInput {
    pub fn unpadded(tokens: Vec<u32>) -> Result<Self> {
        if tokens.is_empty() {
            return Err(crate::error::JevError::Tokenizer(
                "empty token sequence; the template produced no tokens".into(),
            ));
        }
        let pool_index = tokens.len() - 1;
        Ok(Self { tokens, pool_index })
    }
}

/// A pooled last-token hidden state, always f32 regardless of trunk dtype.
#[derive(Debug, Clone, PartialEq)]
pub struct Hidden(pub Vec<f32>);

impl Hidden {
    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: &'static str,
    /// Concrete runtime build, e.g. `b11054`. Printed in the startup banner so a bug
    /// report names the exact native.
    pub version: String,
    pub device: Device,
    pub dtype: Dtype,
    pub context: usize,
    pub hidden_size: usize,
    pub caps: Caps,
}

/// Everything a factory needs to open a backend. Resolved before any factory is consulted.
#[derive(Debug, Clone)]
pub struct OpenRequest {
    pub spec: ModelSpec,
    /// Local path to the trunk weights this backend declared in the registry.
    pub weights: PathBuf,
    pub device: Device,
    pub dtype: Dtype,
    pub context: usize,
    pub n_threads: Option<usize>,
}

pub trait Backend: Send + Sync {
    fn describe(&self) -> BackendInfo;

    fn capabilities(&self) -> Caps {
        self.describe().caps
    }

    /// Last-token hidden state per input, in input order. Implementations MUST reset any
    /// recurrent state between sequences — a leaked gated-DeltaNet state is a silently
    /// wrong label, not a crash. See `docs/adr/0001`.
    fn forward(&self, batch: &[EncodedInput]) -> Result<Vec<Hidden>>;
}

/// Registered by cargo feature. A backend that is not compiled in is *absent* from the
/// registry, which is exactly how "this binary lacks the backend" is detected.
pub trait BackendFactory: Send + Sync {
    fn name(&self) -> &'static str;

    /// Capability tokens this backend advertises, matched against a model's
    /// `backends.<name>.requires`. Strings, not an enum: a new architecture must be a
    /// registry edit, not a core release.
    fn provides(&self) -> &'static [&'static str] {
        &[]
    }

    fn supports_device(&self, device: Device) -> bool;

    fn open(&self, req: &OpenRequest) -> Result<Box<dyn Backend>>;
}

/// Every backend compiled into this binary.
pub fn factories() -> Vec<&'static dyn BackendFactory> {
    // Each arm is a cargo feature. An unbuilt backend is simply absent, which is how
    // "this binary lacks the backend" is detected without a second list to keep in sync.
    vec![
        #[cfg(feature = "backend-llamacpp")]
        crate::backends::llamacpp::factory(),
    ]
}

pub fn factory_named(name: &str) -> Option<&'static dyn BackendFactory> {
    factories().into_iter().find(|f| f.name() == name)
}
