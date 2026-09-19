//! `openjev-core` — NLI inference, backend-agnostic.
//!
//! The shape of this crate is one decision: **the backend contract is
//! `forward(batch) -> last hidden state`, and nothing else.** `predict`, `rerank`,
//! `grade` and `latents` are free functions over that plus a `Linear(hidden -> labels)`
//! head applied in Rust. A backend that implements one method gets all four operations
//! and cannot implement them inconsistently.
//!
//! The second decision is that **a model is config, not code**: an embedded TOML registry
//! plus a user override. Arch, template, label map, tokenizer, head shape and per-backend
//! weights are data.
//!
//! Core never prints. Progress is a callback, warnings are returned, and the caller owns
//! stderr.

pub mod backend;
pub mod backends;
pub mod boot;
pub mod device;
pub mod error;
pub mod head;
pub mod hub;
pub mod ops;
pub mod registry;
pub mod tokenize;

pub use backend::{Backend, BackendInfo, Caps, EncodedInput, Hidden};
pub use boot::{BootOptions, BootReport, boot};
pub use device::{Device, DeviceRequest, Dtype};
pub use error::{JevError, Result};
pub use ops::{Grade, Prediction, Ranked, Session};
pub use registry::{ModelSpec, Registry};
