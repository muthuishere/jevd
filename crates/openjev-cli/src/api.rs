//! The wire types. One definition, used by the server to answer and by the one-shot
//! commands to talk to a server — so `openjev predict --server` cannot drift from what
//! `openjev serve` returns.
//!
//! Scores are a **map keyed by label**, never a bare array: an array's ordering becomes
//! an undocumented tribal fact the first time a label set changes.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const API_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub pairs: usize,
    #[serde(default)]
    pub tokens: usize,
    #[serde(default)]
    pub queue_ms: u64,
    #[serde(default)]
    pub compute_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Truncate {
    /// A silently truncated premise produces a confident, wrong, unfalsifiable answer.
    #[default]
    Error,
    Tail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pair {
    pub premise: String,
    pub hypothesis: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PredictRequest {
    pub pairs: Vec<Pair>,
    #[serde(default)]
    pub truncate: Truncate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub index: usize,
    pub label: String,
    pub scores: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictResponse {
    pub object: String,
    pub model: String,
    pub revision: String,
    pub results: Vec<PredictResult>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RerankRequest {
    pub question: String,
    pub options: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Echoing 200 option strings back doubles the payload for nothing — the caller sent
    /// them.
    #[serde(default)]
    pub return_documents: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub rank: usize,
    pub index: usize,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    pub object: String,
    pub model: String,
    pub revision: String,
    pub results: Vec<RerankResult>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GradeRequest {
    pub answer: String,
    pub reference: String,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

fn default_threshold() -> f32 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradeResponse {
    pub object: String,
    pub model: String,
    pub revision: String,
    pub label: String,
    pub scores: BTreeMap<String, f32>,
    pub pass: bool,
    pub threshold: f32,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LatentsRequest {
    pub texts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentsResponse {
    pub object: String,
    pub model: String,
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limits {
    pub max_body_bytes: u64,
    pub max_pairs: usize,
    pub max_options: usize,
    pub max_field_chars: usize,
    /// `/v1/systemone`: questions per request, and criteria per question.
    pub max_questions: usize,
    pub max_criteria: usize,
    pub max_queue: usize,
    pub max_batch: usize,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model: String,
    pub revision: String,
    pub device: String,
    pub dtype: String,
    pub backend: String,
    pub backend_version: String,
    pub context: usize,
    pub hidden_size: usize,
    pub labels: Vec<String>,
    pub entailment_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoResponse {
    pub object: String,
    pub api_version: u32,
    pub server_version: String,
    /// Feature-detect on this, never on `server_version`: `enable_latents = false` is a
    /// capability difference at an identical version, which version arithmetic cannot
    /// express.
    pub capabilities: Vec<String>,
    pub limits: Limits,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelInfo>,
    pub phase: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadyDetail {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_done: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadyResponse {
    pub ready: bool,
    pub phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ReadyDetail>,
    pub since: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorBody>,
}

/// The envelope, everywhere, including 500s. `code` is the contract and is additive
/// only; `message` is for humans and may change under a client's feet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

/// Every `error.code` the server can emit. Exhaustive on purpose: a code that is not
/// here is a code no client was told about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Code {
    InvalidRequest,
    UnsupportedMediaType,
    Unauthorized,
    NotFound,
    Unprocessable,
    /// `/v1/systemone`: a question the server understood the shape of and
    /// still cannot answer. Separate codes because "you sent a type I do not know" and
    /// "you sent a choice with one option" call for different fixes in the client, and a
    /// single `unprocessable` makes the caller parse `message` to tell them apart.
    InvalidQuestion,
    UnknownQuestionType,
    EmptyCriteria,
    TooManyQuestions,
    StateTooLong,
    PayloadTooLarge,
    QueueFull,
    Timeout,
    Canceled,
    ModelNotReady,
    DeviceError,
    ShuttingDown,
    Internal,
}

impl Code {
    pub fn as_str(self) -> &'static str {
        match self {
            Code::InvalidRequest => "invalid_request",
            Code::UnsupportedMediaType => "unsupported_media_type",
            Code::Unauthorized => "unauthorized",
            Code::NotFound => "not_found",
            Code::Unprocessable => "unprocessable",
            Code::InvalidQuestion => "invalid_question",
            Code::UnknownQuestionType => "unknown_question_type",
            Code::EmptyCriteria => "empty_criteria",
            Code::TooManyQuestions => "too_many_questions",
            Code::StateTooLong => "state_too_long",
            Code::PayloadTooLarge => "payload_too_large",
            Code::QueueFull => "queue_full",
            Code::Timeout => "timeout",
            Code::Canceled => "canceled",
            Code::ModelNotReady => "model_not_ready",
            Code::DeviceError => "device_error",
            Code::ShuttingDown => "shutting_down",
            Code::Internal => "internal",
        }
    }

    pub fn status(self) -> u16 {
        match self {
            Code::InvalidRequest => 400,
            Code::UnsupportedMediaType => 415,
            Code::Unauthorized => 401,
            Code::NotFound => 404,
            Code::Unprocessable
            | Code::InvalidQuestion
            | Code::UnknownQuestionType
            | Code::EmptyCriteria => 422,
            Code::TooManyQuestions | Code::StateTooLong => 413,
            Code::PayloadTooLarge => 413,
            Code::QueueFull => 429,
            Code::Timeout => 504,
            // Never sent — the client is gone. Logged with this code so the logs and the
            // API speak one vocabulary.
            Code::Canceled => 499,
            Code::ModelNotReady | Code::DeviceError | Code::ShuttingDown => 503,
            Code::Internal => 500,
        }
    }

    /// Only the codes where retrying is the right move carry one.
    pub fn retry_after(self) -> Option<u64> {
        match self {
            Code::QueueFull => Some(1),
            Code::ModelNotReady => Some(5),
            _ => None,
        }
    }
}

/// Lifecycle phases, in the order they happen. `/readyz`, `/v1/events` and the state
/// machine all name the same strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Starting,
    Resolving,
    Downloading,
    Loading,
    Ready,
    Draining,
    Failed,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Starting => "starting",
            Phase::Resolving => "resolving",
            Phase::Downloading => "downloading",
            Phase::Loading => "loading",
            Phase::Ready => "ready",
            Phase::Draining => "draining",
            Phase::Failed => "failed",
        }
    }

    pub fn is_ready(self) -> bool {
        matches!(self, Phase::Ready)
    }
}

pub fn scores_map(labels: &[String], probs: &[f32]) -> BTreeMap<String, f32> {
    labels.iter().cloned().zip(probs.iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_keeps_its_documented_status() {
        for (code, status) in [
            (Code::InvalidRequest, 400),
            (Code::UnsupportedMediaType, 415),
            (Code::Unauthorized, 401),
            (Code::NotFound, 404),
            (Code::Unprocessable, 422),
            (Code::InvalidQuestion, 422),
            (Code::UnknownQuestionType, 422),
            (Code::EmptyCriteria, 422),
            (Code::TooManyQuestions, 413),
            (Code::StateTooLong, 413),
            (Code::PayloadTooLarge, 413),
            (Code::QueueFull, 429),
            (Code::Timeout, 504),
            (Code::Canceled, 499),
            (Code::ModelNotReady, 503),
            (Code::DeviceError, 503),
            (Code::Internal, 500),
        ] {
            assert_eq!(code.status(), status, "{}", code.as_str());
        }
    }

    #[test]
    fn only_retryable_codes_carry_retry_after() {
        assert_eq!(Code::QueueFull.retry_after(), Some(1));
        assert_eq!(Code::ModelNotReady.retry_after(), Some(5));
        assert_eq!(Code::Unprocessable.retry_after(), None);
    }

    #[test]
    fn the_envelope_serialises_to_the_documented_shape() {
        let e = ErrorEnvelope {
            error: ErrorBody {
                code: Code::PayloadTooLarge.as_str().into(),
                message: "too big".into(),
                detail: Some(serde_json::json!({"limit_bytes": 1048576})),
                request_id: Some("01JBX".into()),
            },
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["error"]["code"], "payload_too_large");
        assert_eq!(v["error"]["detail"]["limit_bytes"], 1_048_576);
        assert_eq!(v["error"]["request_id"], "01JBX");
    }

    #[test]
    fn scores_are_keyed_by_label_never_positional() {
        let m = scores_map(
            &["contradiction".to_string(), "entailment".to_string()],
            &[0.1, 0.9],
        );
        assert_eq!(m["entailment"], 0.9);
    }

    #[test]
    fn truncate_defaults_to_error_when_the_caller_is_silent() {
        let r: PredictRequest = serde_json::from_str(r#"{"pairs":[]}"#).unwrap();
        assert_eq!(r.truncate, Truncate::Error);
    }
}
