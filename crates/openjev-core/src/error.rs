//! One error type for the whole crate.
//!
//! Organised against: a library that returns `anyhow::Error` and forces every caller to
//! string-match to decide whether it is a 501, a 503 or a 413. Backends may be sloppy
//! internally (`anyhow`) and must map at the trait boundary.

use std::path::PathBuf;

/// Why one backend candidate was rejected for a model. Carried in
/// [`JevError::BackendUnavailable`] so the CLI can print the whole ledger, not just the
/// first failure — "it didn't work" is not actionable, "candle lacks qwen3_5-hybrid" is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendRejection {
    pub backend: String,
    pub reason: String,
}

impl std::fmt::Display for BackendRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:<10} {}", self.backend, self.reason)
    }
}

pub type Result<T, E = JevError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JevError {
    #[error("unknown model '{id}' (known: {})", known.join(", "))]
    ModelNotFound { id: String, known: Vec<String> },

    /// The configured model needs a backend this binary does not have, or that cannot
    /// serve it. Never degrade past this — different weights giving confident wrong
    /// labels is the worst failure this system has.
    #[error("model '{model}' needs a backend this binary does not have\n{}",
            candidates.iter().map(|c| format!("  {c}")).collect::<Vec<_>>().join("\n"))]
    BackendUnavailable {
        model: String,
        candidates: Vec<BackendRejection>,
    },

    #[error("backend '{backend}' does not support {capability}")]
    NotSupported {
        backend: String,
        capability: &'static str,
    },

    #[error("input is {tokens} tokens, context limit is {limit}")]
    ContextOverflow { tokens: usize, limit: usize },

    #[error("device: {0}")]
    Device(String),

    #[error("config {path}: {message}", path = path.display())]
    Config { path: PathBuf, message: String },

    #[error("invalid model config: {0}")]
    Model(String),

    #[error("download {url}: {message}")]
    Download { url: String, message: String },

    /// Integrity failures delete the partial and fail. Never keep-and-warn: a corrupt
    /// 3 GB GGUF that loads is worse than no GGUF.
    #[error("integrity: {file} has sha256 {actual}, config pins {expected}", file = file.display())]
    Integrity {
        file: PathBuf,
        expected: String,
        actual: String,
    },

    #[error("offline: {what} is not in the cache; expected it at {path}", path = path.display())]
    Offline { what: String, path: PathBuf },

    #[error(
        "download of {what} ({bytes} bytes) needs consent; pass --yes or set OPENJEV_ASSUME_YES=1"
    )]
    ConsentRequired { what: String, bytes: u64 },

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("classification head: {0}")]
    Head(String),

    #[error("native library: {0}")]
    Native(String),

    #[error("backend '{backend}': {source}")]
    Backend {
        backend: &'static str,
        #[source]
        source: anyhow::Error,
    },

    #[error("io {path}: {source}", path = path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl JevError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        JevError::Io {
            path: path.into(),
            source,
        }
    }

    pub fn backend(backend: &'static str, source: impl Into<anyhow::Error>) -> Self {
        JevError::Backend {
            backend,
            source: source.into(),
        }
    }

    /// The exit code the CLI should use. 78 is `EX_CONFIG` from sysexits(3) — the
    /// "your configuration is wrong, retrying will not help" class.
    pub fn exit_code(&self) -> i32 {
        match self {
            JevError::ModelNotFound { .. }
            | JevError::BackendUnavailable { .. }
            | JevError::Config { .. }
            | JevError::Model(_) => 78,
            JevError::ConsentRequired { .. } => 77,
            JevError::NotSupported { .. } => 69,
            _ => 1,
        }
    }
}
