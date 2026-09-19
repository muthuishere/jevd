//! The HTTP server (design 02 §3, §4).
//!
//! axum, because admission control, body limits, timeouts and tracing are tower layers
//! we compose rather than framework hooks we learn. The concurrency story lives in
//! `engine.rs`; this module is the contract: one error envelope, `/healthz` split from
//! `/readyz`, and a state file written after bind.

use crate::api::*;
use crate::config::Layered;
use crate::engine::{Engine, JobError, Shared, boot_loader};
use crate::exit::{CliError, CliResult};
use crate::runtime::LoadSpec;
use crate::util;
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

pub use json_extract::ApiJson;
use metrics_exporter_prometheus::PrometheusHandle;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
pub enum Auth {
    /// Loopback only. A token on 127.0.0.1 protects against nothing a local attacker
    /// cannot already do, and it makes the headline command one line instead of three.
    None,
    Bearer {
        hash: String,
    },
}

impl Auth {
    pub fn mode(&self) -> &'static str {
        match self {
            Auth::None => "none",
            Auth::Bearer { .. } => "bearer",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub limits: Limits,
    pub shutdown_grace_secs: u64,
    pub cors_origins: Vec<String>,
    pub metrics: bool,
    pub enable_latents: bool,
    pub auth: Auth,
    pub state_file: PathBuf,
    pub print_ready_json: bool,
    pub fail_if_running: bool,
    pub preload: bool,
    pub load: LoadSpec,
}

/// Flags over env over file over defaults, then the rules that can refuse a startup.
pub fn build_config(args: &crate::cli::ServeArgs, cfg: &mut Layered) -> CliResult<ServerConfig> {
    use toml::Value as V;
    cfg.set_flag("server.host", args.host.clone().map(V::String));
    cfg.set_flag("server.port", args.port.map(|p| V::Integer(p as i64)));
    cfg.set_flag(
        "server.max_queue",
        args.max_queue.map(|n| V::Integer(n as i64)),
    );
    cfg.set_flag(
        "server.max_batch",
        args.max_batch.map(|n| V::Integer(n as i64)),
    );
    cfg.set_flag(
        "server.request_timeout_secs",
        args.request_timeout.map(|n| V::Integer(n as i64)),
    );
    cfg.set_flag(
        "server.shutdown_grace_secs",
        args.shutdown_grace.map(|n| V::Integer(n as i64)),
    );
    if !args.cors_origin.is_empty() {
        cfg.set_flag(
            "server.cors_origins",
            Some(V::Array(
                args.cors_origin.iter().cloned().map(V::String).collect(),
            )),
        );
    }
    cfg.set_flag("model.id", args.model.clone().map(V::String));
    cfg.set_flag("model.revision", args.revision.clone().map(V::String));
    cfg.set_flag(
        "model.cache_dir",
        args.cache_dir
            .clone()
            .map(|p| V::String(p.display().to_string())),
    );
    cfg.set_flag(
        "device.kind",
        args.device
            .map(|d| V::String(format!("{d:?}").to_lowercase())),
    );
    cfg.set_flag("auth.token", args.token.clone().map(V::String));
    cfg.set_flag(
        "auth.token_file",
        args.token_file
            .clone()
            .map(|p| V::String(p.display().to_string())),
    );

    let host = cfg.string("server.host");
    let port = u16::try_from(cfg.int("server.port")).map_err(|_| {
        CliError::config(format!(
            "server.port {} is not a port",
            cfg.int("server.port")
        ))
    })?;

    let token = match (&args.token, cfg.opt_string("auth.token_file")) {
        (Some(t), _) => Some(t.clone()),
        (None, Some(f)) => Some(read_token_file(&PathBuf::from(f))?),
        (None, None) => cfg.opt_string("auth.token"),
    };

    let loopback = util::is_loopback(&host);
    let auth = match (&token, args.no_auth, loopback) {
        (Some(t), _, _) => Auth::Bearer {
            hash: util::sha256_hex(t),
        },
        (None, true, _) => Auth::None,
        (None, false, true) => Auth::None,
        // Refused, not warned: the warning is the thing nobody reads before this ends up
        // on a public IP.
        (None, false, false) => {
            let fresh = util::generate_token();
            return Err(CliError::config(format!(
                "refusing to serve {host} without authentication.\n\
                 A non-loopback bind needs --token, --token-file or auth.token.\n\
                 Here is one to use:\n\n    --token {fresh}\n\n\
                 Override deliberately with --no-auth if this address is already protected."
            )));
        }
    };

    let cors_origins = cfg.strings("server.cors_origins");
    if matches!(auth, Auth::Bearer { .. }) && cors_origins.iter().any(|o| o == "*") {
        // A wildcard origin plus a bearer token is a token-exfiltration invitation.
        return Err(CliError::config(
            "server.cors_origins contains \"*\" while auth is enabled; list origins explicitly"
                .to_string(),
        ));
    }

    let mut load = LoadSpec::from_config(cfg);
    load.offline = args.offline;

    Ok(ServerConfig {
        host,
        port,
        limits: Limits {
            max_body_bytes: cfg.int("server.max_body_bytes").max(0) as u64,
            max_pairs: cfg.usize("server.max_pairs"),
            max_options: cfg.usize("server.max_options"),
            max_field_chars: cfg.usize("server.max_field_chars"),
            max_queue: cfg.usize("server.max_queue"),
            max_batch: cfg.usize("server.max_batch"),
            request_timeout_secs: cfg.int("server.request_timeout_secs").max(0) as u64,
        },
        shutdown_grace_secs: cfg.int("server.shutdown_grace_secs").max(0) as u64,
        cors_origins,
        metrics: cfg.bool("server.metrics"),
        enable_latents: cfg.bool("server.enable_latents"),
        auth,
        state_file: args
            .state_file
            .clone()
            .unwrap_or_else(crate::state::default_state_file),
        print_ready_json: args.print_ready_json,
        fail_if_running: args.fail_if_running,
        preload: args.preload(),
        load,
    })
}

fn read_token_file(path: &PathBuf) -> CliResult<String> {
    let t = std::fs::read_to_string(path)
        .map_err(|e| CliError::config(format!("{}: {e}", path.display())))?;
    let t = t.trim().to_string();
    if t.is_empty() {
        return Err(CliError::config(format!("{} is empty", path.display())));
    }
    Ok(t)
}

// ---------------------------------------------------------------- error envelope

#[derive(Debug, Clone)]
pub struct ApiError {
    pub code: Code,
    pub message: String,
    pub detail: Option<serde_json::Value>,
    pub request_id: Option<String>,
}

impl ApiError {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            detail: None,
            request_id: None,
        }
    }
    pub fn with_detail(mut self, d: serde_json::Value) -> Self {
        self.detail = Some(d);
        self
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.code.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = ErrorEnvelope {
            error: ErrorBody {
                code: self.code.as_str().to_string(),
                message: self.message,
                detail: self.detail,
                request_id: self.request_id,
            },
        };
        let mut resp = (status, Json(body)).into_response();
        if let Some(secs) = self.code.retry_after() {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        resp
    }
}

impl From<JobError> for ApiError {
    fn from(e: JobError) -> Self {
        match e {
            JobError::NotReady(p) => ApiError::new(
                Code::ModelNotReady,
                format!("model is not ready (phase: {p})"),
            ),
            JobError::QueueFull => ApiError::new(Code::QueueFull, "inference queue is full"),
            JobError::ShuttingDown => ApiError::new(Code::ShuttingDown, "server is shutting down"),
            JobError::Device(m) => ApiError::new(Code::DeviceError, m),
            JobError::Unprocessable(m) => ApiError::new(Code::Unprocessable, m),
            JobError::Internal(m) => ApiError::new(Code::Internal, m),
        }
    }
}

// ---------------------------------------------------------------- app state

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<ServerConfig>,
    pub engine: Engine,
    pub shared: Arc<Shared>,
    pub metrics: Option<PrometheusHandle>,
}

impl AppState {
    fn model_or_not_ready(&self) -> Result<ModelInfo, ApiError> {
        self.shared
            .model
            .read()
            .ok()
            .and_then(|m| m.clone())
            .ok_or_else(|| {
                ApiError::new(
                    Code::ModelNotReady,
                    format!(
                        "model is not loaded (phase: {})",
                        self.shared.phase().as_str()
                    ),
                )
            })
    }
}

// ---------------------------------------------------------------- router

pub fn router(state: AppState) -> Router {
    use tower_http::limit::RequestBodyLimitLayer;

    let public = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz));

    let metrics_route = Router::new().route("/metrics", get(metrics_handler));

    let protected = Router::new()
        .route("/v1/info", get(info))
        .route("/v1/model", get(model))
        .route("/v1/events", get(events))
        .route("/v1/predict", post(predict))
        .route("/v1/rerank", post(rerank))
        .route("/v1/grade", post(grade))
        .route("/v1/latents", post(latents))
        .route("/openapi.json", get(openapi))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_auth,
        ));

    // /metrics is unauthenticated on a loopback bind and token-gated otherwise: a probe
    // that needs a secret is a probe that will be misconfigured, but an open metrics
    // endpoint on a public IP is a free map of your traffic.
    let metrics_route = if matches!(state.cfg.auth, Auth::Bearer { .. }) {
        metrics_route.layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_auth,
        ))
    } else {
        metrics_route
    };

    let mut app = public.merge(protected);
    if state.cfg.metrics {
        app = app.merge(metrics_route);
    }

    app.fallback(not_found)
        .layer(axum::middleware::from_fn(envelope_layer))
        .layer(RequestBodyLimitLayer::new(
            state.cfg.limits.max_body_bytes as usize,
        ))
        .layer(cors_layer(&state.cfg))
        .with_state(state)
}

fn cors_layer(cfg: &ServerConfig) -> tower_http::cors::CorsLayer {
    use tower_http::cors::CorsLayer;
    if cfg.cors_origins.is_empty() {
        // Off by default. An allowlist is a decision; a wildcard is an accident.
        return CorsLayer::new();
    }
    let origins: Vec<HeaderValue> = cfg
        .cors_origins
        .iter()
        .filter_map(|o| HeaderValue::from_str(o).ok())
        .collect();
    CorsLayer::new()
        .allow_origin(origins)
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
}

/// One request id, one set of protocol headers, one access log line, one metric — for
/// every response including the ones tower produced without us.
async fn envelope_layer(mut req: Request, next: Next) -> Response {
    let id = util::request_id();
    let endpoint = req.uri().path().to_string();
    let started = Instant::now();
    req.extensions_mut().insert(RequestId(id.clone()));
    let mut resp = next.run(req).await;

    let status = resp.status();
    if status == StatusCode::PAYLOAD_TOO_LARGE && !is_our_envelope(&resp) {
        // tower's body limit rejects before any handler runs, so the envelope has to be
        // put back on here or the one documented error shape has a hole in it.
        let mut replaced = ApiError {
            code: Code::PayloadTooLarge,
            message: "request body exceeds the configured limit".into(),
            detail: None,
            request_id: Some(id.clone()),
        }
        .into_response();
        std::mem::swap(&mut resp, &mut replaced);
    }

    // Every error envelope carries the same id as the header. Doing it here means no
    // handler can forget, and the id in a user's screenshot always matches the logs.
    if status.is_client_error() || status.is_server_error() {
        resp = inject_request_id(resp, &id).await;
    }

    let h = resp.headers_mut();
    h.insert("x-openjev-api", HeaderValue::from_static("1"));
    if let Ok(v) = HeaderValue::from_str(&id) {
        h.insert("x-request-id", v);
    }
    h.insert("x-frame-options", HeaderValue::from_static("DENY"));
    h.insert(
        "content-security-policy",
        HeaderValue::from_static("frame-ancestors 'none'"),
    );

    let code = status.as_u16().to_string();
    metrics::counter!("openjev_requests_total", "endpoint" => endpoint.clone(), "code" => code)
        .increment(1);
    metrics::histogram!("openjev_request_duration_seconds", "endpoint" => endpoint.clone())
        .record(started.elapsed().as_secs_f64());
    // Counts and timings, never content.
    tracing::info!(
        request_id = %id,
        endpoint = %endpoint,
        status = status.as_u16(),
        duration_ms = started.elapsed().as_millis() as u64,
        "request"
    );
    resp
}

async fn inject_request_id(resp: Response, id: &str) -> Response {
    if !is_our_envelope(&resp) {
        return resp;
    }
    let (parts, body) = resp.into_parts();
    // Error bodies are small by construction; buffering one is not the 4 GB read the
    // body limit exists to prevent.
    let Ok(bytes) = axum::body::to_bytes(body, 64 * 1024).await else {
        return Response::from_parts(parts, axum::body::Body::empty());
    };
    let Ok(mut v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Response::from_parts(parts, axum::body::Body::from(bytes));
    };
    if v.get("error").is_some() && v["error"].get("request_id").is_none() {
        v["error"]["request_id"] = serde_json::Value::String(id.to_string());
    }
    let body = serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec());
    let mut parts = parts;
    parts.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(parts, axum::body::Body::from(body))
}

fn is_our_envelope(resp: &Response) -> bool {
    resp.headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"))
}

#[derive(Clone)]
pub struct RequestId(pub String);

async fn require_auth(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let Auth::Bearer { hash } = &state.cfg.auth else {
        return next.run(req).await;
    };
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if util::token_matches(presented, hash) {
        next.run(req).await
    } else {
        tracing::warn!("unauthorized request");
        ApiError::new(Code::Unauthorized, "missing or invalid bearer token").into_response()
    }
}

async fn not_found() -> ApiError {
    // A 404 is a better error than a wrong number: there is no OpenAI-shaped shim here
    // on purpose (design 02 §3.10).
    ApiError::new(Code::NotFound, "no such endpoint")
}

// ---------------------------------------------------------------- handlers

/// "Do not restart me" — 200 while downloading, loading and draining. A supervisor that
/// conflates this with /readyz kills the process 40 s into a 90 s load, forever.
async fn healthz(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "ok": true,
        "phase": state.shared.phase().as_str(),
        "uptime_seconds": state.shared.started.elapsed().as_secs(),
        "queue_depth": state.engine.queue_depth(),
    }))
}

/// "Send me traffic" — 503 until weights are resident, with the live phase and byte
/// counts that make a multi-GB wait legible instead of mysterious.
async fn readyz(State(state): State<AppState>) -> Response {
    let snap = state.shared.snapshot();
    let ready = snap.phase.is_ready();
    let body = ReadyResponse {
        ready,
        phase: snap.phase.as_str().to_string(),
        detail: snap.detail.clone(),
        since: snap.since.clone(),
        error: snap.error.clone(),
    };
    if ready {
        (StatusCode::OK, Json(body)).into_response()
    } else {
        let mut resp = (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
        resp
    }
}

async fn info(State(state): State<AppState>) -> Json<InfoResponse> {
    let mut capabilities = vec![
        "predict".to_string(),
        "rerank".to_string(),
        "grade".to_string(),
        "events".to_string(),
    ];
    if state.cfg.metrics {
        capabilities.push("metrics".into());
    }
    if state.cfg.enable_latents {
        capabilities.push("latents".into());
    }
    Json(InfoResponse {
        object: "info".into(),
        api_version: API_VERSION,
        server_version: SERVER_VERSION.into(),
        capabilities,
        limits: state.cfg.limits.clone(),
        model: state.shared.model.read().ok().and_then(|m| m.clone()),
        phase: state.shared.phase().as_str().to_string(),
    })
}

async fn model(State(state): State<AppState>) -> Result<Json<ModelInfo>, ApiError> {
    Ok(Json(state.model_or_not_ready()?))
}

async fn events(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    use tokio_stream::StreamExt;
    let rx = state.shared.events.subscribe();
    let initial = {
        let snap = state.shared.snapshot();
        serde_json::json!({ "event": "phase", "data": { "phase": snap.phase.as_str(), "detail": snap.detail }})
    };
    let stream = tokio_stream::once(initial)
        .chain(tokio_stream::wrappers::BroadcastStream::new(rx).filter_map(|r| r.ok()))
        .map(|v| {
            let name = v
                .get("event")
                .and_then(|e| e.as_str())
                .unwrap_or("message")
                .to_string();
            let data = v.get("data").cloned().unwrap_or(serde_json::Value::Null);
            Ok(Event::default().event(name).data(data.to_string()))
        });
    // A 15 s comment keeps proxies from reaping a connection that is, by design, silent
    // for minutes at a time.
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

async fn metrics_handler(State(state): State<AppState>) -> Response {
    match &state.metrics {
        Some(h) => (
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            h.render(),
        )
            .into_response(),
        None => ApiError::new(Code::NotFound, "metrics are disabled").into_response(),
    }
}

async fn openapi(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(crate::openapi::document(&state.cfg))
}

fn check_field(name: &str, text: &str, max: usize) -> Result<(), ApiError> {
    if text.chars().count() > max {
        return Err(ApiError::new(
            Code::Unprocessable,
            format!("{name} is longer than the {max}-character limit"),
        )
        .with_detail(serde_json::json!({"field": name, "limit_chars": max})));
    }
    Ok(())
}

/// Core owns tokenisation and exposes no truncating encode, so `truncate: "tail"` cannot
/// be honoured honestly. Refusing is the only option that does not risk a confident
/// answer computed from half a premise.
fn check_truncate(t: Truncate) -> Result<(), ApiError> {
    match t {
        Truncate::Error => Ok(()),
        Truncate::Tail => Err(ApiError::new(
            Code::Unprocessable,
            "truncate: \"tail\" is not supported by this build; send shorter input or split it",
        )),
    }
}

async fn with_deadline<F, T>(state: &AppState, fut: F) -> Result<T, ApiError>
where
    F: std::future::Future<Output = Result<T, ApiError>>,
{
    let secs = state.cfg.limits.request_timeout_secs;
    if secs == 0 {
        return fut.await;
    }
    match tokio::time::timeout(Duration::from_secs(secs), fut).await {
        Ok(v) => v,
        Err(_) => Err(ApiError::new(
            Code::Timeout,
            format!("request exceeded the {secs}s server deadline"),
        )),
    }
}

async fn predict(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<PredictRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    check_truncate(req.truncate)?;
    if req.pairs.is_empty() {
        return Err(ApiError::new(Code::Unprocessable, "pairs is empty"));
    }
    if req.pairs.len() > state.cfg.limits.max_pairs {
        return Err(ApiError::new(
            Code::PayloadTooLarge,
            format!(
                "{} pairs, limit is {}",
                req.pairs.len(),
                state.cfg.limits.max_pairs
            ),
        )
        .with_detail(serde_json::json!({"limit_pairs": state.cfg.limits.max_pairs})));
    }
    for (i, p) in req.pairs.iter().enumerate() {
        check_field(
            &format!("pairs[{i}].premise"),
            &p.premise,
            state.cfg.limits.max_field_chars,
        )?;
        check_field(
            &format!("pairs[{i}].hypothesis"),
            &p.hypothesis,
            state.cfg.limits.max_field_chars,
        )?;
    }
    let info = state.model_or_not_ready()?;
    let pairs: Vec<(String, String)> = req
        .pairs
        .iter()
        .map(|p| (p.premise.clone(), p.hypothesis.clone()))
        .collect();
    let n = pairs.len();
    let out = with_deadline(&state, async { Ok(state.engine.predict(pairs).await?) }).await?;
    state.shared.requests_served.fetch_add(1, Ordering::Relaxed);

    // Request order is a promise. Batching happens behind it and must never show.
    let results = out
        .value
        .iter()
        .enumerate()
        .map(|(index, p)| PredictResult {
            id: req.pairs[index].id.clone(),
            index,
            label: p.label.clone(),
            scores: scores_map(&info.labels, &p.probs),
        })
        .collect();
    Ok(Json(PredictResponse {
        object: "predict".into(),
        model: info.model,
        revision: info.revision,
        results,
        usage: Usage {
            pairs: n,
            tokens: 0,
            queue_ms: out.queue_ms,
            compute_ms: out.compute_ms,
        },
    }))
}

async fn rerank(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<RerankRequest>,
) -> Result<Json<RerankResponse>, ApiError> {
    if req.options.is_empty() {
        return Err(ApiError::new(Code::Unprocessable, "options is empty"));
    }
    if req.options.len() > state.cfg.limits.max_options {
        return Err(ApiError::new(
            Code::PayloadTooLarge,
            format!(
                "{} options, limit is {}",
                req.options.len(),
                state.cfg.limits.max_options
            ),
        )
        .with_detail(serde_json::json!({"limit_options": state.cfg.limits.max_options})));
    }
    check_field("question", &req.question, state.cfg.limits.max_field_chars)?;
    for (i, o) in req.options.iter().enumerate() {
        check_field(
            &format!("options[{i}]"),
            o,
            state.cfg.limits.max_field_chars,
        )?;
    }
    let info = state.model_or_not_ready()?;
    // The pair order (question, option) is core's `Session::rerank`: the question is the
    // premise, the option is the hypothesis. Duplicated here rather than routed through
    // core so that batching still coalesces reranks with predicts — which means this line
    // is a second place to be wrong, and it *was* wrong (both sides scored the pair
    // backwards until the golden fixture caught it). See `docs/adr/0012`.
    let pairs: Vec<(String, String)> = req
        .options
        .iter()
        .map(|o| (req.question.clone(), o.clone()))
        .collect();
    let n = pairs.len();
    let out = with_deadline(&state, async { Ok(state.engine.predict(pairs).await?) }).await?;
    state.shared.requests_served.fetch_add(1, Ordering::Relaxed);

    // `unwrap_or(0)` here would rank every option by P(contradiction) the moment the
    // entailment label stopped matching — a confident, silent reversal of the ranking.
    // Core refuses that at boot; so does this.
    let ent = info
        .labels
        .iter()
        .position(|l| *l == info.entailment_label)
        .ok_or_else(|| {
            ApiError::new(
                Code::Internal,
                format!(
                    "entailment label '{}' is not among the model's labels {:?}",
                    info.entailment_label, info.labels
                ),
            )
        })?;
    let mut ranked: Vec<(usize, f32)> = out
        .value
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.probs.get(ent).copied().unwrap_or(0.0)))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top = req.top_k.unwrap_or(ranked.len()).min(ranked.len());
    let results = ranked[..top]
        .iter()
        .enumerate()
        .map(|(rank, (index, score))| RerankResult {
            rank,
            index: *index,
            score: *score,
            text: req.return_documents.then(|| req.options[*index].clone()),
        })
        .collect();
    Ok(Json(RerankResponse {
        object: "rerank".into(),
        model: info.model,
        revision: info.revision,
        results,
        usage: Usage {
            pairs: n,
            tokens: 0,
            queue_ms: out.queue_ms,
            compute_ms: out.compute_ms,
        },
    }))
}

async fn grade(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<GradeRequest>,
) -> Result<Json<GradeResponse>, ApiError> {
    check_field("answer", &req.answer, state.cfg.limits.max_field_chars)?;
    check_field(
        "reference",
        &req.reference,
        state.cfg.limits.max_field_chars,
    )?;
    let info = state.model_or_not_ready()?;
    // premise = reference, hypothesis = answer: we are asking whether the ground truth
    // entails what was said, not the reverse.
    let pairs = vec![(req.reference.clone(), req.answer.clone())];
    let out = with_deadline(&state, async { Ok(state.engine.predict(pairs).await?) }).await?;
    state.shared.requests_served.fetch_add(1, Ordering::Relaxed);
    let p = out
        .value
        .first()
        .ok_or_else(|| ApiError::new(Code::Internal, "grade produced no prediction"))?;
    let scores = scores_map(&info.labels, &p.probs);
    let entail = scores.get(&info.entailment_label).copied().unwrap_or(0.0);
    Ok(Json(GradeResponse {
        object: "grade".into(),
        model: info.model,
        revision: info.revision,
        label: p.label.clone(),
        scores,
        pass: entail >= req.threshold,
        threshold: req.threshold,
        usage: Usage {
            pairs: 1,
            tokens: 0,
            queue_ms: out.queue_ms,
            compute_ms: out.compute_ms,
        },
    }))
}

async fn latents(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<LatentsRequest>,
) -> Result<Json<LatentsResponse>, ApiError> {
    if !state.cfg.enable_latents {
        // Behind a flag it is a research tool; on by default it is a compatibility
        // obligation nobody agreed to.
        return Err(ApiError::new(
            Code::NotFound,
            "latents are disabled; set server.enable_latents = true to enable them",
        ));
    }
    if req.texts.is_empty() {
        return Err(ApiError::new(Code::Unprocessable, "texts is empty"));
    }
    let info = state.model_or_not_ready()?;
    let n = req.texts.len();
    let out = with_deadline(&state, async { Ok(state.engine.latents(req.texts).await?) }).await?;
    let dim = out.value.first().map(Vec::len).unwrap_or(0);
    Ok(Json(LatentsResponse {
        object: "latents".into(),
        model: info.model,
        dim,
        vectors: out.value,
        usage: Usage {
            pairs: n,
            tokens: 0,
            queue_ms: out.queue_ms,
            compute_ms: out.compute_ms,
        },
    }))
}

// ---------------------------------------------------------------- lifecycle

pub async fn run(cfg: ServerConfig, assume_yes: bool) -> CliResult<i32> {
    let cfg = Arc::new(cfg);

    // 3. The lock, before anything expensive. A repeated start is idempotent — exactly
    // what a KeepAlive unit plus an impatient human produces.
    let pid_file = crate::state::pid_file_for(&cfg.state_file);
    let _lock = match crate::state::acquire(&pid_file)? {
        crate::state::Lock::Acquired(l) => l,
        crate::state::Lock::Held => {
            let existing = crate::state::read_state(&cfg.state_file);
            let where_ = existing
                .as_ref()
                .map(|s| format!("{} (pid {})", s.url, s.pid))
                .unwrap_or_else(|| "an unknown address".into());
            eprintln!("openjev is already running on {where_}");
            return Ok(if cfg.fail_if_running {
                crate::exit::FAILURE
            } else {
                crate::exit::OK
            });
        }
    };

    // 4. Preflight before the first byte: ENOSPC at 94 % of a 7 GB download is entirely
    // preventable here, and consent must be asked before anything is fetched.
    let reg = crate::runtime::registry()?;
    let spec = crate::runtime::resolve_spec(&reg, &cfg.load)?;
    let plan = crate::runtime::plan(&spec, cfg.load.cache_dir.as_ref())?;
    if let Some((need, free)) = plan.space_shortfall() {
        return Err(CliError::Jev(openjev_core::JevError::Download {
            url: spec.repo.clone(),
            message: format!(
                "needs {} free in {} (10% headroom), only {} available",
                util::human_bytes(need),
                plan.cache_root.display(),
                util::human_bytes(free)
            ),
        }));
    }
    print_banner(&cfg, &plan);
    crate::runtime::ensure_consent(
        &plan,
        &crate::runtime::ConsentOptions {
            assume_yes,
            interactive: util::stdin_is_tty() && util::stderr_is_tty(),
        },
    )?;

    let shared = Shared::new();
    let metrics = if cfg.metrics {
        metrics_exporter_prometheus::PrometheusBuilder::new()
            .install_recorder()
            .inspect(|_| {
                metrics::gauge!("openjev_build_info", "version" => SERVER_VERSION).set(1.0)
            })
            .ok()
    } else {
        None
    };

    let engine = Engine::spawn(
        shared.clone(),
        cfg.limits.max_queue,
        cfg.limits.max_batch,
        boot_loader(cfg.load.clone(), shared.clone()),
    );

    // 5. Bind. 6. Write the state file *after* bind, so bound_addr is real and --port 0
    // is discoverable.
    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse().map_err(|e| {
        CliError::config(format!("{}:{} is not an address: {e}", cfg.host, cfg.port))
    })?;
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        CliError::other(format!(
            "cannot bind {addr}: {e}. Another process owns the port — try --port 0."
        ))
    })?;
    let bound = listener.local_addr()?;
    let url = format!("http://{bound}");

    tracing::info!(
        version = SERVER_VERSION,
        api_version = API_VERSION,
        "openjev"
    );
    tracing::info!(model = %spec.id, revision = %spec.revision, "model");
    tracing::info!(cache = %plan.cache_root.display(), "cache");
    tracing::info!(bind = %bound, auth = cfg.auth.mode(), "listening");
    tracing::info!(
        queue = cfg.limits.max_queue,
        batch = cfg.limits.max_batch,
        body = cfg.limits.max_body_bytes,
        timeout_secs = cfg.limits.request_timeout_secs,
        "limits"
    );
    if !util::is_loopback(&cfg.host) {
        tracing::warn!("serving a non-loopback address");
    }
    if matches!(cfg.auth, Auth::None) && !util::is_loopback(&cfg.host) {
        tracing::warn!("--no-auth: this server is unauthenticated on a routable address");
    }
    if !cfg.cors_origins.is_empty() {
        tracing::warn!(origins = ?cfg.cors_origins, "CORS is enabled");
    }

    crate::state::write_state(
        &cfg.state_file,
        &crate::state::ServerState {
            schema: crate::state::SCHEMA,
            pid: std::process::id(),
            url: url.clone(),
            bound_addr: bound.to_string(),
            api_version: API_VERSION,
            server_version: SERVER_VERSION.into(),
            model: spec.id.clone(),
            revision: spec.revision.clone(),
            device: cfg
                .load
                .device
                .map(|d| format!("{d:?}").to_lowercase())
                .unwrap_or_else(|| "auto".into()),
            auth: cfg.auth.mode().into(),
            started_at: util::rfc3339_now(),
        },
    )?;

    let state = AppState {
        cfg: cfg.clone(),
        engine,
        shared: shared.clone(),
        metrics,
    };
    let app = router(state.clone());

    // The ready announcement lives in its own task so a 90 s load never blocks the
    // listener that is already answering /healthz and /readyz.
    let reloader = tokio::spawn(watch_sighup(cfg.clone()));
    let announce = tokio::spawn(announce_when_ready(
        cfg.clone(),
        shared.clone(),
        url.clone(),
    ));

    // --preload (default): a load failure is a startup failure, so a supervisor restarts
    // rather than parking on a server that will never be ready. --no-preload keeps it up
    // and honest at 503.
    let failed = if cfg.preload {
        Some(tokio::spawn(watch_for_failed_load(
            shared.clone(),
            cfg.state_file.clone(),
        )))
    } else {
        None
    };

    let shutdown_shared = shared.clone();
    let grace = cfg.shutdown_grace_secs;
    let (drained_tx, mut drained_rx) = tokio::sync::watch::channel(false);
    let serve_fut = axum::serve(listener, app).with_graceful_shutdown(async move {
        let code = wait_for_shutdown().await;
        tracing::info!(signal = code, "draining");
        // /readyz goes 503 first so a load balancer stops sending before we stop
        // accepting.
        shutdown_shared.set_phase(Phase::Draining, None);
        let _ = drained_tx.send(true);
        tokio::time::sleep(Duration::from_millis(50)).await;
    });
    // IntoFuture, not Future: pin the future it produces.
    let serve_fut = std::future::IntoFuture::into_future(serve_fut);
    tokio::pin!(serve_fut);
    // The grace window is a bound, not a hope: a "graceful" shutdown you cannot get out
    // of is the failure this is organised against.
    let served = tokio::select! {
        r = &mut serve_fut => r,
        _ = async {
            let _ = drained_rx.wait_for(|d| *d).await;
            tokio::time::sleep(Duration::from_secs(grace)).await;
        } => {
            tracing::warn!(grace_secs = grace, "grace window expired; dropping in-flight work");
            Ok(())
        }
    };

    announce.abort();
    reloader.abort();
    if let Some(h) = failed {
        h.abort();
    }
    crate::state::remove_state(&cfg.state_file);
    served.map_err(|e| CliError::other(format!("server: {e}")))?;
    Ok(crate::exit::OK)
}

/// SIGTERM/SIGINT begins the drain; a **second** signal exits 130 immediately, because a
/// graceful shutdown you cannot get out of is not graceful.
async fn wait_for_shutdown() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
        let mut int = signal(SignalKind::interrupt()).expect("SIGINT handler");
        let first = tokio::select! {
            _ = term.recv() => "SIGTERM",
            _ = int.recv() => "SIGINT",
        };
        tokio::spawn(async move {
            let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
            let mut int = signal(SignalKind::interrupt()).expect("SIGINT handler");
            tokio::select! {
                _ = term.recv() => {},
                _ = int.recv() => {},
            }
            eprintln!("second signal — exiting now");
            std::process::exit(crate::exit::INTERRUPTED);
        });
        first
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        "CTRL-C"
    }
}

/// SIGHUP reloads what cannot affect loaded weights. Anything else is **refused by
/// name** rather than silently ignored — a silently-ignored reload is how you spend an
/// afternoon wondering why the device did not change.
#[cfg(unix)]
async fn watch_sighup(cfg: Arc<ServerConfig>) {
    use tokio::signal::unix::{SignalKind, signal};
    let Ok(mut hup) = signal(SignalKind::hangup()) else {
        return;
    };
    const FROZEN: &[&str] = &[
        "server.host",
        "server.port",
        "server.enable_latents",
        "model.id",
        "model.revision",
        "device.kind",
    ];
    while hup.recv().await.is_some() {
        let env: std::collections::BTreeMap<String, String> = std::env::vars().collect();
        let fresh = match crate::config::Layered::load(None, &env) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("SIGHUP: config is invalid, keeping the running one: {e}");
                continue;
            }
        };
        match crate::logging::set_level(&fresh.string("log.level")) {
            Ok(()) => {
                tracing::info!(level = %fresh.string("log.level"), "SIGHUP: log level reloaded")
            }
            Err(e) => tracing::error!("SIGHUP: log level not reloaded: {e}"),
        }
        for key in FROZEN {
            let now = fresh.string(key);
            let running = match *key {
                "server.host" => cfg.host.clone(),
                "server.port" => cfg.port.to_string(),
                "server.enable_latents" => cfg.enable_latents.to_string(),
                "model.id" => cfg.load.model.clone().unwrap_or_default(),
                "model.revision" => cfg.load.revision.clone().unwrap_or_default(),
                _ => cfg
                    .load
                    .device
                    .map(|d| format!("{d:?}").to_lowercase())
                    .unwrap_or_default(),
            };
            if !now.is_empty() && now != running {
                tracing::warn!(
                    key,
                    running = %running,
                    requested = %now,
                    "SIGHUP: this key needs a restart and was NOT applied"
                );
            }
        }
    }
}

#[cfg(not(unix))]
async fn watch_sighup(_cfg: Arc<ServerConfig>) {
    std::future::pending::<()>().await
}

/// A model that will never load is a startup failure, not a permanent 503.
async fn watch_for_failed_load(shared: Arc<Shared>, state_file: PathBuf) {
    loop {
        match shared.phase() {
            Phase::Failed => {
                let snap = shared.snapshot();
                let msg = snap
                    .error
                    .map(|e| e.message)
                    .unwrap_or_else(|| "model load failed".into());
                eprintln!("error: {msg}");
                // Exiting skips every destructor, so the discovery contract is honoured
                // by hand here: a state file outliving its process is a client dialling a
                // dead port.
                crate::state::remove_state(&state_file);
                let _ = std::fs::remove_file(crate::state::pid_file_for(&state_file));
                std::process::exit(snap.exit_code);
            }
            Phase::Ready => return,
            _ => tokio::time::sleep(Duration::from_millis(200)).await,
        }
    }
}

async fn announce_when_ready(cfg: Arc<ServerConfig>, shared: Arc<Shared>, url: String) {
    loop {
        match shared.phase() {
            Phase::Ready => break,
            Phase::Failed => return,
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
    let model = shared.model.read().ok().and_then(|m| m.clone());
    if cfg.print_ready_json {
        // Exactly one line of JSON on stdout, ever. A client blocks on this line.
        let line = serde_json::json!({
            "url": url,
            "pid": std::process::id(),
            "api_version": API_VERSION,
            "model": model.as_ref().map(|m| m.model.clone()),
            "revision": model.as_ref().map(|m| m.revision.clone()),
        });
        println!("{line}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
    let auth_note = match cfg.auth {
        Auth::None => "loopback only, no auth required",
        Auth::Bearer { .. } => "bearer token required",
    };
    eprintln!();
    eprintln!("  listening on {url}   ({auth_note})");
    eprintln!(
        "  ready  ·  try:  curl -s {url}/v1/predict -H 'content-type: application/json' \
         -d '{{\"pairs\":[{{\"premise\":\"it rained\",\"hypothesis\":\"the ground is wet\"}}]}}'"
    );
    eprintln!();
    eprintln!("  press Ctrl-C to stop");
}

/// §2.7's transcript header: what will be used, where, and what it will cost — printed
/// **before** the wait starts, never after.
fn print_banner(cfg: &ServerConfig, plan: &crate::runtime::LoadPlan) {
    let cfg_path = crate::config::default_config_path();
    let found = if cfg_path.is_file() {
        String::new()
    } else {
        "    (not found — using defaults)".to_string()
    };
    eprintln!(
        "openjev {SERVER_VERSION}  ·  api v{API_VERSION}\n\n  model    {}@{}\n  device   {}\n  cache    {}{}\n  config   {}{}",
        plan.spec.id,
        plan.spec.revision,
        cfg.load
            .device
            .map(|d| format!("{d:?}").to_lowercase())
            .unwrap_or_else(|| "auto".into()),
        plan.cache_root.display(),
        plan.free_bytes
            .map(|f| format!("          ({} free)", util::human_bytes(f)))
            .unwrap_or_default(),
        cfg_path.display(),
        found
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    fn serve_args(args: &[&str]) -> crate::cli::ServeArgs {
        let cli = Cli::try_parse_from(args).expect("parse");
        match cli.command {
            Command::Serve(s) => *s,
            _ => panic!("not serve"),
        }
    }

    fn empty_config() -> Layered {
        Layered::load(None, &std::collections::BTreeMap::new()).expect("defaults")
    }

    #[test]
    fn a_non_loopback_bind_without_a_token_refuses_to_start_and_hands_you_one() {
        let args = serve_args(&["openjev", "serve", "--host", "0.0.0.0"]);
        let err = build_config(&args, &mut empty_config()).unwrap_err();
        assert_eq!(err.exit_code(), crate::exit::CONFIG);
        let msg = err.to_string();
        assert!(msg.contains("refusing to serve"), "{msg}");
        assert!(
            msg.contains("--token "),
            "it must hand over a usable token: {msg}"
        );
    }

    #[test]
    fn a_non_loopback_bind_with_a_token_is_fine_and_the_token_is_only_kept_hashed() {
        let args = serve_args(&["openjev", "serve", "--host", "0.0.0.0", "--token", "sekrit"]);
        let cfg = build_config(&args, &mut empty_config()).unwrap();
        match &cfg.auth {
            Auth::Bearer { hash } => {
                assert_ne!(hash, "sekrit");
                assert!(util::token_matches("sekrit", hash));
            }
            _ => panic!("expected bearer"),
        }
    }

    #[test]
    fn no_auth_is_an_explicit_override_not_a_default() {
        let args = serve_args(&["openjev", "serve", "--host", "0.0.0.0", "--no-auth"]);
        let cfg = build_config(&args, &mut empty_config()).unwrap();
        assert!(matches!(cfg.auth, Auth::None));
    }

    #[test]
    fn loopback_needs_no_token_at_all() {
        let args = serve_args(&["openjev", "serve"]);
        let cfg = build_config(&args, &mut empty_config()).unwrap();
        assert!(matches!(cfg.auth, Auth::None));
        assert_eq!(cfg.port, crate::cli::DEFAULT_PORT);
    }

    #[test]
    fn a_wildcard_cors_origin_is_refused_when_a_token_is_in_play() {
        let args = serve_args(&[
            "openjev",
            "serve",
            "--host",
            "0.0.0.0",
            "--token",
            "t",
            "--cors-origin",
            "*",
        ]);
        let err = build_config(&args, &mut empty_config()).unwrap_err();
        assert!(err.to_string().contains("cors_origins"), "{err}");
    }

    #[test]
    fn flags_beat_the_config_file_for_the_server_section() {
        let mut cfg = empty_config();
        cfg.merge_toml(
            "[server]\nport = 1234\nmax_queue = 7\n",
            std::path::Path::new("t"),
        )
        .unwrap();
        let args = serve_args(&["openjev", "serve", "--port", "4321"]);
        let out = build_config(&args, &mut cfg).unwrap();
        assert_eq!(out.port, 4321, "the flag wins");
        assert_eq!(
            out.limits.max_queue, 7,
            "the file wins where no flag was typed"
        );
    }

    #[test]
    fn a_token_file_is_read_and_trimmed() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("tok");
        std::fs::write(&f, "  filetoken\n").unwrap();
        let args = serve_args(&[
            "openjev",
            "serve",
            "--host",
            "0.0.0.0",
            "--token-file",
            f.to_str().unwrap(),
        ]);
        let cfg = build_config(&args, &mut empty_config()).unwrap();
        match &cfg.auth {
            Auth::Bearer { hash } => assert!(util::token_matches("filetoken", hash)),
            _ => panic!("expected bearer"),
        }
    }
}

/// A JSON extractor that speaks our envelope.
///
/// axum's own `Json` rejection is a bare text body with no `error.code`, which would put
/// a hole in "one shape, everywhere" at exactly the moment a client is most confused —
/// when it sent something malformed.
mod json_extract {
    use super::{ApiError, Code};
    use axum::extract::{FromRequest, Request, rejection::JsonRejection};

    pub struct ApiJson<T>(pub T);

    impl<S, T> FromRequest<S> for ApiJson<T>
    where
        S: Send + Sync,
        axum::Json<T>: FromRequest<S, Rejection = JsonRejection>,
    {
        type Rejection = ApiError;

        async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
            match axum::Json::<T>::from_request(req, state).await {
                Ok(axum::Json(v)) => Ok(ApiJson(v)),
                Err(JsonRejection::MissingJsonContentType(_)) => Err(ApiError::new(
                    Code::UnsupportedMediaType,
                    "send application/json",
                )),
                Err(JsonRejection::BytesRejection(_)) => Err(ApiError::new(
                    Code::PayloadTooLarge,
                    "request body exceeds the configured limit",
                )),
                Err(e) => Err(ApiError::new(Code::InvalidRequest, e.body_text())),
            }
        }
    }
}
