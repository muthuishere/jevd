//! The inference worker: one thread, one model, a bounded queue and opportunistic
//! batching.
//!
//! `openjev-core`'s `Session` is synchronous, and ADR 0003 says concurrent context
//! creation corrupts silently. So there is exactly **one** worker thread, it owns the
//! `Session` outright, and the async side talks to it over a bounded channel. No
//! `spawn_blocking` pool, because a pool would be two contexts the first time it grew —
//! we do not pretend to a concurrency we do not have.
//!
//! Batching is opportunistic and never delayed: when the worker wakes it drains whatever
//! is *already* queued up to `max_batch` pairs and runs one forward pass. Throughput of
//! batching, zero added latency at low load. A batching *timer* would trade the common
//! case (one caller, idle box) for the rare one.

use crate::api::{ErrorBody, ModelInfo, Phase, ReadyDetail};
use crate::runtime::{LoadSpec, ProgressRenderer};
use openjev_core::hub::Progress;
use openjev_core::{Prediction, Registry, Session};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::{broadcast, mpsc, oneshot};

#[derive(Debug, Clone)]
pub struct PhaseState {
    pub phase: Phase,
    pub since: String,
    pub detail: Option<ReadyDetail>,
    pub error: Option<ErrorBody>,
    /// The exit code a preloading server should die with. Carried from the load failure
    /// so "no backend compiled in" (5) does not report itself as "model unavailable" (4).
    pub exit_code: i32,
}

impl Default for PhaseState {
    fn default() -> Self {
        Self {
            phase: Phase::Starting,
            since: crate::util::rfc3339_now(),
            detail: None,
            error: None,
            exit_code: crate::exit::FAILURE,
        }
    }
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum JobError {
    #[error("the model is not loaded (phase: {0})")]
    NotReady(String),
    #[error("queue is full")]
    QueueFull,
    #[error("server is shutting down")]
    ShuttingDown,
    #[error("{0}")]
    Device(String),
    #[error("{0}")]
    Unprocessable(String),
    #[error("{0}")]
    Internal(String),
}

#[derive(Debug)]
pub struct Outcome<T> {
    pub value: T,
    pub queue_ms: u64,
    pub compute_ms: u64,
    /// Tokens this job's own inputs encoded to — its share of the coalesced batch, never
    /// the batch's total. Zero only where the work is not pair work.
    pub tokens: usize,
}

enum Work {
    Pairs {
        pairs: Vec<(String, String)>,
        resp: oneshot::Sender<Result<Outcome<Vec<Prediction>>, JobError>>,
    },
    Latents {
        texts: Vec<String>,
        resp: oneshot::Sender<Result<Outcome<Vec<Vec<f32>>>, JobError>>,
    },
}

struct Job {
    work: Work,
    enqueued: Instant,
}

/// Everything the HTTP layer reads without touching the worker.
pub struct Shared {
    pub state: RwLock<PhaseState>,
    pub model: RwLock<Option<ModelInfo>>,
    pub events: broadcast::Sender<serde_json::Value>,
    pub queue_depth: AtomicUsize,
    pub started: Instant,
    pub requests_served: AtomicUsize,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        let (events, _) = broadcast::channel(256);
        Arc::new(Self {
            state: RwLock::new(PhaseState::default()),
            model: RwLock::new(None),
            events,
            queue_depth: AtomicUsize::new(0),
            started: Instant::now(),
            requests_served: AtomicUsize::new(0),
        })
    }

    pub fn phase(&self) -> Phase {
        self.state.read().map(|s| s.phase).unwrap_or(Phase::Failed)
    }

    pub fn snapshot(&self) -> PhaseState {
        self.state
            .read()
            .map(|s| s.clone())
            .unwrap_or_else(|_| PhaseState {
                phase: Phase::Failed,
                since: crate::util::rfc3339_now(),
                detail: None,
                error: Some(ErrorBody {
                    code: "internal".into(),
                    message: "phase lock poisoned".into(),
                    detail: None,
                    request_id: None,
                }),
                exit_code: crate::exit::FAILURE,
            })
    }

    pub fn set_phase(&self, phase: Phase, detail: Option<ReadyDetail>) {
        if let Ok(mut s) = self.state.write() {
            let changed = s.phase != phase;
            s.phase = phase;
            s.detail = detail.clone();
            if changed {
                s.since = crate::util::rfc3339_now();
                tracing::info!(phase = phase.as_str(), "phase");
            }
        }
        self.emit(
            "phase",
            serde_json::json!({ "phase": phase.as_str(), "detail": detail }),
        );
        metrics::gauge!("openjev_ready").set(if phase.is_ready() { 1.0 } else { 0.0 });
    }

    pub fn fail(&self, message: impl Into<String>) {
        self.fail_with(message, crate::exit::FAILURE);
    }

    pub fn fail_with(&self, message: impl Into<String>, exit_code: i32) {
        let message = message.into();
        if let Ok(mut s) = self.state.write() {
            s.phase = Phase::Failed;
            s.since = crate::util::rfc3339_now();
            s.error = Some(ErrorBody {
                code: "internal".into(),
                message: message.clone(),
                detail: None,
                request_id: None,
            });
            s.exit_code = exit_code;
        }
        self.emit(
            "phase",
            serde_json::json!({ "phase": "failed", "error": message }),
        );
    }

    /// Fire-and-forget: no subscribers is the normal case, not an error.
    pub fn emit(&self, event: &str, data: serde_json::Value) {
        let _ = self
            .events
            .send(serde_json::json!({ "event": event, "data": data }));
    }
}

#[derive(Clone)]
pub struct Engine {
    tx: mpsc::Sender<Job>,
    pub shared: Arc<Shared>,
}

impl Engine {
    /// Spawns the worker thread. Loading happens *inside* it, so a 90-second model load
    /// never blocks the runtime that is already serving `/healthz` and `/readyz`.
    pub fn spawn(shared: Arc<Shared>, max_queue: usize, max_batch: usize, loader: Loader) -> Self {
        let (tx, rx) = mpsc::channel(max_queue);
        let s = shared.clone();
        std::thread::Builder::new()
            .name("openjev-infer".into())
            .spawn(move || worker(rx, s, max_batch, loader))
            .expect("spawn inference worker");
        Engine { tx, shared }
    }

    pub fn queue_depth(&self) -> usize {
        self.shared.queue_depth.load(Ordering::Relaxed)
    }

    pub async fn predict(
        &self,
        pairs: Vec<(String, String)>,
    ) -> Result<Outcome<Vec<Prediction>>, JobError> {
        let (resp, rx) = oneshot::channel();
        self.submit(Job {
            work: Work::Pairs { pairs, resp },
            enqueued: Instant::now(),
        })?;
        rx.await.map_err(|_| JobError::ShuttingDown)?
    }

    pub async fn latents(&self, texts: Vec<String>) -> Result<Outcome<Vec<Vec<f32>>>, JobError> {
        let (resp, rx) = oneshot::channel();
        self.submit(Job {
            work: Work::Latents { texts, resp },
            enqueued: Instant::now(),
        })?;
        rx.await.map_err(|_| JobError::ShuttingDown)?
    }

    /// `try_send`, never `send`. Waiting for queue space *is* the unbounded queue we are
    /// organised against — every caller waits 40 s, times out, retries, and the queue
    /// never drains.
    fn submit(&self, job: Job) -> Result<(), JobError> {
        let phase = self.shared.phase();
        if !phase.is_ready() {
            return Err(match phase {
                Phase::Draining => JobError::ShuttingDown,
                p => JobError::NotReady(p.as_str().to_string()),
            });
        }
        match self.tx.try_send(job) {
            Ok(()) => {
                let d = self.shared.queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
                metrics::gauge!("openjev_queue_depth").set(d as f64);
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(_)) => Err(JobError::QueueFull),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(JobError::ShuttingDown),
        }
    }
}

/// Deferred load, run on the worker thread.
pub type LoadFailure = (String, i32);
pub type Loader = Box<dyn FnOnce(&mut ProgressRenderer) -> Result<Session, LoadFailure> + Send>;

/// Builds the loader the server uses: core's `boot`, with progress fanned out to
/// `/readyz` and `/v1/events`.
pub fn boot_loader(load: LoadSpec, shared: Arc<Shared>) -> Loader {
    Box::new(move |renderer: &mut ProgressRenderer| {
        let fail = |e: crate::exit::CliError| (e.to_string(), e.exit_code());
        let reg = Registry::load(None).map_err(|e| fail(e.into()))?;
        let spec = crate::runtime::resolve_spec(&reg, &load).map_err(fail)?;
        if spec.revision_is_floating() {
            // Loud, not hidden: a floating revision means two machines can serve
            // different weights under one version string.
            tracing::warn!(
                model = %spec.id,
                revision = %spec.revision,
                "model revision is not a commit sha — the weights this serves can change under you"
            );
            shared.emit(
                "warning",
                serde_json::json!({
                    "kind": "floating_revision",
                    "model": spec.id,
                    "revision": spec.revision
                }),
            );
        }
        let s2 = shared.clone();
        renderer.observer = Some(Box::new(move |file: &str, p: Progress| match p {
            Progress::Start { total } => {
                s2.set_phase(
                    Phase::Downloading,
                    Some(ReadyDetail {
                        file: Some(file.to_string()),
                        bytes_done: Some(0),
                        bytes_total: total,
                        eta_seconds: None,
                        message: None,
                    }),
                );
            }
            Progress::Advance { done, total } => {
                s2.set_phase(
                    Phase::Downloading,
                    Some(ReadyDetail {
                        file: Some(file.to_string()),
                        bytes_done: Some(done),
                        bytes_total: total,
                        eta_seconds: None,
                        message: None,
                    }),
                );
            }
            Progress::Done { bytes } => {
                s2.emit(
                    "download_done",
                    serde_json::json!({ "file": file, "bytes": bytes }),
                );
            }
        }));
        shared.set_phase(Phase::Resolving, None);
        let t = Instant::now();
        let (session, report) =
            crate::runtime::load_session(&reg, &load, &spec, renderer).map_err(fail)?;
        for w in &report.warnings {
            tracing::warn!("{w}");
        }
        for d in &report.demotions {
            tracing::warn!("device: {d}");
        }
        metrics::histogram!("openjev_model_load_seconds").record(t.elapsed().as_secs_f64());
        Ok(session)
    })
}

pub fn model_info(session: &Session) -> ModelInfo {
    let i = session.backend.describe();
    ModelInfo {
        model: session.spec.id.clone(),
        revision: session.spec.revision.clone(),
        device: i.device.to_string(),
        dtype: i.dtype.to_string(),
        backend: i.name.to_string(),
        backend_version: i.version.clone(),
        context: i.context,
        hidden_size: i.hidden_size,
        labels: session.spec.labels.clone(),
        entailment_label: session.spec.entailment_label.clone(),
    }
}

fn worker(mut rx: mpsc::Receiver<Job>, shared: Arc<Shared>, max_batch: usize, loader: Loader) {
    let mut renderer = ProgressRenderer::new(crate::util::stderr_is_tty());
    let session = match loader(&mut renderer) {
        Ok(s) => s,
        Err((e, code)) => {
            tracing::error!("model load failed: {e}");
            shared.fail_with(e, code);
            // Drain and refuse: a caller blocked forever on a server that will never
            // load is worse than a 503 that says why.
            rx.close();
            while let Some(job) = rx.blocking_recv() {
                fail_job(job, JobError::NotReady("failed".into()));
            }
            return;
        }
    };
    if let Some(line) = renderer.summary() {
        eprintln!("{line}");
    }
    shared
        .model
        .write()
        .map(|mut m| *m = Some(model_info(&session)))
        .ok();
    tracing::info!("{}", session.banner());
    shared.set_phase(Phase::Ready, None);

    while let Some(first) = rx.blocking_recv() {
        let mut batch = vec![first];
        let mut pairs_in_batch = batch_pairs(&batch[0]);
        // Drain what is already there. `try_recv` never waits, which is the difference
        // between opportunistic batching and a latency tax.
        while pairs_in_batch < max_batch {
            match rx.try_recv() {
                Ok(job) => {
                    pairs_in_batch += batch_pairs(&job);
                    batch.push(job);
                }
                Err(_) => break,
            }
        }
        shared.queue_depth.fetch_sub(batch.len(), Ordering::Relaxed);
        metrics::gauge!("openjev_queue_depth")
            .set(shared.queue_depth.load(Ordering::Relaxed) as f64);
        metrics::histogram!("openjev_batch_size").record(pairs_in_batch as f64);
        run_batch(&session, batch);
    }
}

fn batch_pairs(job: &Job) -> usize {
    match &job.work {
        Work::Pairs { pairs, .. } => pairs.len(),
        Work::Latents { texts, .. } => texts.len(),
    }
}

/// Pair jobs are coalesced into one forward pass; latents jobs run alone because the
/// backend may not support them at all and the failure must land on the right caller.
fn run_batch(session: &Session, batch: Vec<Job>) {
    let mut pair_jobs = Vec::new();
    let mut flat: Vec<(String, String)> = Vec::new();
    for job in batch {
        match job.work {
            Work::Pairs { pairs, resp } => {
                // A client that hung up while queued costs us nothing to drop, and that
                // is the main win under load.
                if resp.is_closed() {
                    metrics::counter!("openjev_requests_total", "endpoint" => "canceled", "code" => crate::api::Code::Canceled.status().to_string()).increment(1);
                    continue;
                }
                let range = flat.len()..flat.len() + pairs.len();
                flat.extend(pairs);
                pair_jobs.push((range, resp, job.enqueued));
            }
            Work::Latents { texts, resp } => {
                let t = Instant::now();
                let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                let out = session
                    .latents(&refs)
                    .map_err(map_err)
                    .map(|value| Outcome {
                        value,
                        queue_ms: job.enqueued.elapsed().as_millis() as u64
                            - t.elapsed().as_millis() as u64,
                        compute_ms: t.elapsed().as_millis() as u64,
                        // Latents do not go through the pair encoder, so there is no
                        // honest count to report here rather than a guessed one.
                        tokens: 0,
                    });
                let _ = resp.send(out);
            }
        }
    }
    if flat.is_empty() {
        return;
    }
    let t = Instant::now();
    let refs: Vec<(&str, &str)> = flat.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    // A second tokenisation pass, deliberately. It is microseconds against a ~50 ms
    // forward, and the alternative — threading a count out of `predict` — would put the
    // number on a path where a backend could report one thing and the encoder another.
    let token_counts = session.count_pair_tokens(&refs).unwrap_or_default();
    let result = session.predict(&refs);
    let compute_ms = t.elapsed().as_millis() as u64;
    metrics::histogram!("openjev_inference_duration_seconds").record(t.elapsed().as_secs_f64());
    metrics::counter!("openjev_pairs_total").increment(flat.len() as u64);
    match result {
        Ok(preds) => {
            for (range, resp, enqueued) in pair_jobs {
                let tokens = token_counts
                    .get(range.clone())
                    .map_or(0, |t| t.iter().sum());
                let slice = preds[range].to_vec();
                let queue_ms = enqueued
                    .elapsed()
                    .as_millis()
                    .saturating_sub(compute_ms as u128) as u64;
                let _ = resp.send(Ok(Outcome {
                    value: slice,
                    queue_ms,
                    compute_ms,
                    tokens,
                }));
            }
        }
        Err(e) => {
            // One bad pair fails the whole coalesced pass; the alternative is re-running
            // each request alone, which turns a transient device fault into an N-times
            // slower outage.
            let err = map_err(e);
            for (_, resp, _) in pair_jobs {
                let _ = resp.send(Err(err.clone()));
            }
        }
    }
}

fn fail_job(job: Job, err: JobError) {
    match job.work {
        Work::Pairs { resp, .. } => {
            let _ = resp.send(Err(err));
        }
        Work::Latents { resp, .. } => {
            let _ = resp.send(Err(err));
        }
    }
}

fn map_err(e: openjev_core::JevError) -> JobError {
    use openjev_core::JevError as J;
    match &e {
        J::ContextOverflow { .. } => JobError::Unprocessable(e.to_string()),
        J::NotSupported { .. } => JobError::Unprocessable(e.to_string()),
        J::Device(_) | J::Backend { .. } | J::Native(_) => JobError::Device(e.to_string()),
        _ => JobError::Internal(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_server_is_starting_and_not_ready() {
        let s = Shared::new();
        assert_eq!(s.phase(), Phase::Starting);
        assert!(!s.phase().is_ready());
    }

    #[test]
    fn phase_transitions_stamp_a_new_since_and_survive_a_reader() {
        let s = Shared::new();
        let first = s.snapshot().since;
        s.set_phase(Phase::Downloading, None);
        let snap = s.snapshot();
        assert_eq!(snap.phase, Phase::Downloading);
        assert!(!snap.since.is_empty());
        assert_eq!(first.len(), snap.since.len());
    }

    #[test]
    fn a_failed_load_is_terminal_and_carries_the_reason() {
        let s = Shared::new();
        s.fail("no backend compiled in");
        let snap = s.snapshot();
        assert_eq!(snap.phase, Phase::Failed);
        assert!(snap.error.expect("error").message.contains("no backend"));
    }

    #[tokio::test]
    async fn work_submitted_before_ready_is_refused_rather_than_queued() {
        let shared = Shared::new();
        let engine = Engine::spawn(
            shared.clone(),
            4,
            32,
            Box::new(|_| {
                Err((
                    "no weights in a unit test".to_string(),
                    crate::exit::FAILURE,
                ))
            }),
        );
        let err = engine
            .predict(vec![("a".into(), "b".into())])
            .await
            .unwrap_err();
        assert!(
            matches!(err, JobError::NotReady(_)),
            "queueing against an unloaded model is how you get a caller waiting forever"
        );
    }

    #[test]
    fn events_do_not_fail_when_nobody_is_listening() {
        let s = Shared::new();
        s.emit("phase", serde_json::json!({"phase":"ready"}));
    }
}
