//! The HTTP contract, exercised through the real router.
//!
//! These tests run without weights on purpose: every property the Go plugin depends on —
//! the readyz phase split, the error envelope, admission control, auth, the route set —
//! is observable before a single parameter is loaded, and a test that needs a 4 GB
//! download is a test nobody runs.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use openjev_cli::api::{ModelInfo, Phase};
use openjev_cli::cli::{Cli, Command};
use openjev_cli::config::Layered;
use openjev_cli::engine::{Engine, Shared};
use openjev_cli::openapi::ROUTES;
use openjev_cli::server::{AppState, ServerConfig, build_config, router};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tower::ServiceExt;

fn config(args: &[&str]) -> ServerConfig {
    let cli = <Cli as clap::Parser>::try_parse_from(args).expect("parse");
    let Command::Serve(s) = cli.command else {
        panic!("not serve")
    };
    let mut layered = Layered::load(None, &BTreeMap::new()).expect("defaults");
    build_config(&s, &mut layered).expect("config")
}

/// A server whose model never loads: the phase stays where we put it, so every
/// not-ready behaviour is reachable.
fn app_with(mut cfg: ServerConfig, phase: Phase, model: bool) -> (axum::Router, Arc<Shared>) {
    // The worker in these tests never finishes loading, so anything that reaches it must
    // hit the deadline quickly rather than park the test for a minute.
    cfg.limits.request_timeout_secs = 1;
    let shared = Shared::new();
    let engine = Engine::spawn(
        shared.clone(),
        cfg.limits.max_queue,
        cfg.limits.max_batch,
        Box::new(|_| {
            // Never returns: the worker parks and the queue is the only thing moving.
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }),
    );
    if model {
        *shared.model.write().unwrap() = Some(ModelInfo {
            model: "test-model".into(),
            revision: "main".into(),
            device: "cpu".into(),
            dtype: "f32".into(),
            backend: "test".into(),
            backend_version: "0".into(),
            context: 8192,
            hidden_size: 2560,
            labels: vec![
                "contradiction".into(),
                "entailment".into(),
                "neutral".into(),
            ],
            entailment_label: "entailment".into(),
        });
    }
    shared.set_phase(phase, None);
    let metrics = cfg.metrics.then(metrics_handle).flatten();
    let state = AppState {
        cfg: Arc::new(cfg),
        engine,
        shared: shared.clone(),
        metrics,
    };
    (router(state), shared)
}

/// The Prometheus recorder is process-global and installs once; later calls reuse the
/// handle so every test in this binary still has a working /metrics.
fn metrics_handle() -> Option<metrics_exporter_prometheus::PrometheusHandle> {
    use std::sync::OnceLock;
    static H: OnceLock<Option<metrics_exporter_prometheus::PrometheusHandle>> = OnceLock::new();
    H.get_or_init(|| {
        metrics_exporter_prometheus::PrometheusBuilder::new()
            .install_recorder()
            .ok()
    })
    .clone()
}

async fn send(
    app: &axum::Router,
    req: Request<Body>,
) -> (StatusCode, Value, axum::http::HeaderMap) {
    let resp = app.clone().oneshot(req).await.expect("response");
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json, headers)
}

fn get(path: &str) -> Request<Body> {
    Request::builder()
        .uri(path)
        .body(Body::empty())
        .expect("request")
}

fn post(path: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request")
}

#[tokio::test]
async fn healthz_is_200_while_the_model_is_still_downloading() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Downloading, false);
    let (status, body, _) = send(&app, get("/healthz")).await;
    // A supervisor that restarts on this would kill every multi-GB first run, forever.
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["phase"], "downloading");
}

#[tokio::test]
async fn readyz_is_503_with_the_phase_and_a_retry_after_until_the_model_is_resident() {
    let (app, shared) = app_with(config(&["openjev", "serve"]), Phase::Downloading, false);
    let (status, body, headers) = send(&app, get("/readyz")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["ready"], false);
    assert_eq!(body["phase"], "downloading");
    assert!(body["since"].as_str().is_some_and(|s| s.ends_with('Z')));
    assert_eq!(headers["retry-after"], "5");

    // Downloading and loading are different answers to "why not yet", which is the
    // entire reason the phase is on the wire.
    shared.set_phase(Phase::Loading, None);
    let (_, body, _) = send(&app, get("/readyz")).await;
    assert_eq!(body["phase"], "loading");

    shared.set_phase(Phase::Ready, None);
    let (status, body, _) = send(&app, get("/readyz")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["ready"], true);
}

#[tokio::test]
async fn readyz_carries_live_byte_counts_during_a_download() {
    let (app, shared) = app_with(config(&["openjev", "serve"]), Phase::Starting, false);
    shared.set_phase(
        Phase::Downloading,
        Some(openjev_cli::api::ReadyDetail {
            file: Some("model-00001-of-00002.safetensors".into()),
            bytes_done: Some(4_402_341_888),
            bytes_total: Some(7_935_819_776),
            eta_seconds: Some(40),
            message: None,
        }),
    );
    let (_, body, _) = send(&app, get("/readyz")).await;
    assert_eq!(body["detail"]["bytes_done"], 4_402_341_888u64);
    assert_eq!(body["detail"]["bytes_total"], 7_935_819_776u64);
    assert_eq!(body["detail"]["eta_seconds"], 40);
}

#[tokio::test]
async fn a_failed_load_is_a_terminal_503_that_says_why() {
    let (app, shared) = app_with(config(&["openjev", "serve"]), Phase::Starting, false);
    shared.fail("no backend compiled in");
    let (status, body, _) = send(&app, get("/readyz")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["phase"], "failed");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("no backend"))
    );
}

#[tokio::test]
async fn work_sent_before_the_model_is_ready_gets_model_not_ready_and_a_retry_after() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Loading, false);
    let (status, body, headers) = send(
        &app,
        post(
            "/v1/predict",
            serde_json::json!({"pairs":[{"premise":"a","hypothesis":"b"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "model_not_ready");
    assert_eq!(headers["retry-after"], "5");
}

#[tokio::test]
async fn every_error_carries_the_envelope_and_the_request_id_from_the_header() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let (status, body, headers) = send(&app, get("/nope")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");
    assert!(body["error"]["message"].is_string());
    assert_eq!(
        body["error"]["request_id"].as_str().unwrap(),
        headers["x-request-id"].to_str().unwrap(),
        "the id in the body must be the id in the header, or a bug report cannot be traced"
    );
    assert_eq!(headers["x-openjev-api"], "1");
}

#[tokio::test]
async fn a_body_that_is_not_json_is_a_415_and_malformed_json_is_a_400() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/predict")
        .body(Body::from("pairs=1"))
        .unwrap();
    let (status, body, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_eq!(body["error"]["code"], "unsupported_media_type");

    let req = Request::builder()
        .method("POST")
        .uri("/v1/predict")
        .header("content-type", "application/json")
        .body(Body::from("{ not json"))
        .unwrap();
    let (status, body, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn limits_are_enforced_before_any_inference_and_named_in_the_detail() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);

    let pairs: Vec<Value> = (0..300)
        .map(|_| serde_json::json!({"premise":"a","hypothesis":"b"}))
        .collect();
    let (status, body, _) = send(
        &app,
        post("/v1/predict", serde_json::json!({"pairs": pairs})),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body["error"]["code"], "payload_too_large");
    assert_eq!(body["error"]["detail"]["limit_pairs"], 256);

    let (status, body, _) = send(&app, post("/v1/predict", serde_json::json!({"pairs": []}))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], "unprocessable");

    let long = "x".repeat(40_000);
    let (status, body, _) = send(
        &app,
        post(
            "/v1/predict",
            serde_json::json!({"pairs":[{"premise":long,"hypothesis":"b"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["detail"]["limit_chars"], 32_768);

    let (status, body, _) = send(
        &app,
        post(
            "/v1/rerank",
            serde_json::json!({"question":"q","options":[]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], "unprocessable");
}

#[tokio::test]
async fn tail_truncation_is_refused_rather_than_faked() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let (status, body, _) = send(
        &app,
        post(
            "/v1/predict",
            serde_json::json!({"pairs":[{"premise":"a","hypothesis":"b"}],"truncate":"tail"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], "unprocessable");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_full_queue_is_429_with_a_retry_after_not_an_unbounded_wait() {
    // max_queue 2 and a worker that never finishes loading: admission is the only thing
    // that can answer, which is exactly the death-spiral case.
    let (app, _) = app_with(
        config(&["openjev", "serve", "--max-queue", "2"]),
        Phase::Ready,
        true,
    );
    let body = serde_json::json!({"pairs":[{"premise":"a","hypothesis":"b"}]});
    // Concurrently, because the queue only fills while earlier requests are still in it.
    let mut set = tokio::task::JoinSet::new();
    for _ in 0..8 {
        let app = app.clone();
        let body = body.clone();
        set.spawn(async move { send(&app, post("/v1/predict", body)).await });
    }
    let mut saw_429 = false;
    while let Some(res) = set.join_next().await {
        let (status, json, headers) = res.expect("task");
        if status == StatusCode::TOO_MANY_REQUESTS {
            assert_eq!(json["error"]["code"], "queue_full");
            assert_eq!(headers["retry-after"], "1");
            saw_429 = true;
        }
    }
    assert!(saw_429, "a bounded queue must reject, not absorb");
}

#[tokio::test]
async fn latents_are_absent_until_they_are_switched_on() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let (status, body, _) = send(
        &app,
        post("/v1/latents", serde_json::json!({"texts":["a"]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");

    let (_, info, _) = send(&app, get("/v1/info")).await;
    let caps: Vec<String> = serde_json::from_value(info["capabilities"].clone()).unwrap();
    assert!(
        !caps.contains(&"latents".to_string()),
        "capabilities must be feature-detectable: an off endpoint is not advertised"
    );
}

#[tokio::test]
async fn info_publishes_the_limits_a_client_must_chunk_to() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let (status, body, _) = send(&app, get("/v1/info")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["api_version"], 1);
    assert_eq!(body["limits"]["max_pairs"], 256);
    assert_eq!(body["limits"]["max_options"], 512);
    assert_eq!(body["limits"]["max_queue"], 64);
    assert_eq!(body["model"]["labels"][1], "entailment");
    assert_eq!(body["phase"], "ready");
}

#[tokio::test]
async fn a_secured_server_answers_probes_unauthenticated_and_nothing_else() {
    let cfg = config(&["openjev", "serve", "--host", "0.0.0.0", "--token", "sekrit"]);
    let (app, _) = app_with(cfg, Phase::Ready, true);

    for path in ["/healthz", "/readyz"] {
        let (status, _, _) = send(&app, get(path)).await;
        assert_ne!(
            status,
            StatusCode::UNAUTHORIZED,
            "{path} must never need a secret: a probe that needs one gets misconfigured"
        );
    }

    let (status, body, _) = send(&app, get("/v1/info")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "unauthorized");

    let req = Request::builder()
        .uri("/v1/info")
        .header("authorization", "Bearer sekrit")
        .body(Body::empty())
        .unwrap();
    let (status, _, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::OK);

    let req = Request::builder()
        .uri("/v1/info")
        .header("authorization", "Bearer wrong")
        .body(Body::empty())
        .unwrap();
    let (status, _, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn every_documented_route_is_actually_routed() {
    // The anti-drift check for /openapi.json: a documented path that 404s is a generated
    // client that fails on its first call.
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    for (path, method) in ROUTES {
        let req = if *method == "get" {
            get(path)
        } else {
            post(path, serde_json::json!({}))
        };
        // /v1/events is an SSE stream that never ends by design, so "it did not answer
        // 404 within a second" is the strongest honest assertion for it.
        match tokio::time::timeout(std::time::Duration::from_secs(2), send(&app, req)).await {
            Ok((status, _, _)) => assert_ne!(
                status,
                StatusCode::NOT_FOUND,
                "{method} {path} is documented but not routed",
            ),
            Err(_) => assert_eq!(*path, "/v1/events", "{method} {path} hung"),
        }
    }
}

#[tokio::test]
async fn openapi_is_served_and_describes_this_servers_limits() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let (status, doc, _) = send(&app, get("/openapi.json")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(doc["openapi"], "3.1.0");
    assert_eq!(
        doc["components"]["schemas"]["PredictRequest"]["properties"]["pairs"]["maxItems"],
        256
    );
}

#[tokio::test]
async fn metrics_are_open_on_loopback_and_absent_when_switched_off() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let (status, _, _) = send(&app, get("/metrics")).await;
    assert_eq!(status, StatusCode::OK, "metrics need no token on loopback");

    let mut cfg = config(&["openjev", "serve"]);
    cfg.metrics = false;
    let (app, _) = app_with(cfg, Phase::Ready, true);
    let (status, body, _) = send(&app, get("/metrics")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");
}

// ---------------------------------------------------------------- /v1/systemone
//
// Every rejection below is reached before the worker, so it is observable with no
// weights — which is the point: a client integrating against this server can see the
// whole error contract on a laptop.

/// The owner's exact example, as a value, so the request under test and the one in the
/// README cannot drift apart silently.
fn owners_example() -> Value {
    serde_json::json!({
        "model": "openjev",
        "state": "My card was charged twice. Please help ASAP.",
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": "Does this message convey urgency?",
                "criteria": { "true": "Explicitly time-sensitive", "false": "No urgency expressed" }
            },
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments and refunds",
                    "technical": "Bugs and integrations",
                    "sales": "Pricing and new accounts"
                }
            }
        }
    })
}

#[tokio::test]
async fn systemone_is_routed_and_reaches_the_worker_on_a_valid_request() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let (status, body, _) = send(&app, post("/v1/systemone", owners_example())).await;
    // The model in this harness never loads, so the honest end of the road is the
    // deadline — not a 404 and not a validation error. Anything else means the owner's
    // example does not even get as far as the engine.
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert_eq!(body["error"]["code"], "timeout");
}

#[tokio::test]
async fn systemone_is_refused_before_the_model_is_resident() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Loading, false);
    let (status, body, headers) = send(&app, post("/v1/systemone", owners_example())).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "model_not_ready");
    assert_eq!(headers["retry-after"], "5");
}

#[tokio::test]
async fn systemone_needs_the_same_bearer_token_as_every_other_protected_route() {
    let cfg = config(&["openjev", "serve", "--token", "s3cret"]);
    let (app, _) = app_with(cfg, Phase::Ready, true);
    let (status, body, _) = send(&app, post("/v1/systemone", owners_example())).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "unauthorized");

    // And the right token gets through to the same deadline as the unauthenticated
    // loopback case: one auth path, not a second one bolted onto the new endpoint.
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .header("authorization", "Bearer s3cret")
        .body(Body::from(owners_example().to_string()))
        .expect("request");
    let (status, _, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
}

#[tokio::test]
async fn a_malformed_question_is_a_clean_4xx_with_its_own_code() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let cases: &[(Value, StatusCode, &str)] = &[
        (
            serde_json::json!({"state":"s","questions":{"q":{"type":"vibes","instructions":"?"}}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "unknown_question_type",
        ),
        (
            // One option is not a choice: answering it would be confident and meaningless.
            serde_json::json!({"state":"s","questions":{"q":{"type":"choice","criteria":{"only":"just the one"}}}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_question",
        ),
        (
            serde_json::json!({"state":"s","questions":{"q":{"type":"choice","criteria":{"a":"","b":""}}}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "empty_criteria",
        ),
        (
            serde_json::json!({"state":"s","questions":{"q":{"type":"choice"}}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "empty_criteria",
        ),
        (
            serde_json::json!({"state":"","questions":{"q":{"type":"noul","instructions":"?"}}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "unprocessable",
        ),
        (
            serde_json::json!({"state":"s","questions":{}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "unprocessable",
        ),
        (
            serde_json::json!({"state":"x".repeat(32_769),"questions":{"q":{"type":"noul","instructions":"?"}}}),
            StatusCode::PAYLOAD_TOO_LARGE,
            "state_too_long",
        ),
    ];
    for (body, status, code) in cases {
        let (got, answer, _) = send(&app, post("/v1/systemone", body.clone())).await;
        assert_eq!(got, *status, "{code}: {answer}");
        assert_eq!(answer["error"]["code"], *code, "{answer}");
        // The one envelope, on the new endpoint too.
        assert!(answer["error"]["message"].is_string());
        assert!(answer["error"]["request_id"].is_string());
    }
}

#[tokio::test]
async fn too_many_questions_is_refused_by_count_before_anything_is_encoded() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let mut questions = serde_json::Map::new();
    for i in 0..33 {
        questions.insert(
            format!("q{i}"),
            serde_json::json!({"type":"noul","instructions":"?"}),
        );
    }
    let (status, body, _) = send(
        &app,
        post(
            "/v1/systemone",
            serde_json::json!({"state":"s","questions":questions}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body["error"]["code"], "too_many_questions");
}

#[tokio::test]
async fn info_advertises_systemone_and_its_limits() {
    let (app, _) = app_with(config(&["openjev", "serve"]), Phase::Ready, true);
    let (_, body, _) = send(&app, get("/v1/info")).await;
    let caps: Vec<String> = serde_json::from_value(body["capabilities"].clone()).expect("caps");
    // Feature detection, not version arithmetic: a client asks whether this server has
    // the endpoint before it posts to it.
    assert!(caps.contains(&"systemone".to_string()), "{caps:?}");
    assert_eq!(body["limits"]["max_questions"], 32);
    assert_eq!(body["limits"]["max_criteria"], 255);
}
