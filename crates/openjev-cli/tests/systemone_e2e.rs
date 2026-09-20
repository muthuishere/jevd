//! The owner's example, through the whole server, with no weights.
//!
//! `systemone_live.rs` runs the same request against the real checkpoint and is skipped
//! unless someone has one. This file closes the gap: it builds a **real**
//! `openjev_core::Session` — real registry entry, real template, real tokenizer, real
//! linear head — over a stub backend whose hidden states are chosen, and runs the request
//! through the real router, the real auth layer, the real worker and the real answer
//! arithmetic.
//!
//! So the numbers are ours and the *shape* is the server's. That is the right split: the
//! shape is the compatibility contract and can be pinned exactly here, on every machine,
//! in milliseconds; what the checkpoint believes about a double charge is a fact about the
//! checkpoint and belongs in a test that loads it.
//!
//! The stub is deliberately not a mock of `predict`. Mocking there would skip the
//! template, the pooling rule and the label lookup — which is where this codebase's two
//! worst bugs both lived.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use openjev_cli::api::Phase;
use openjev_cli::cli::{Cli, Command};
use openjev_cli::config::Layered;
use openjev_cli::engine::{Engine, Shared};
use openjev_cli::server::{AppState, ServerConfig, build_config, router};
use openjev_core::backend::{Backend, BackendInfo, Caps, EncodedInput, Hidden};
use openjev_core::device::{Device, Dtype};
use openjev_core::head::Head;
use openjev_core::registry::{HeadSpec, Registry};
use openjev_core::tokenize::Encoder;
use openjev_core::{Result as JevResult, Session};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tower::ServiceExt;

/// Words that make a hypothesis true, by token id. Everything else is not entailed.
///
/// `refunds` appears only in the billing criterion and `sensitive` only in the urgent
/// `true` criterion, so the stub encodes one opinion — "this customer has a refund
/// problem and it is time-sensitive" — and every other criterion disagrees with it.
const ENTAILED_WORDS: [&str; 2] = ["refunds", "sensitive"];

struct Stub {
    entailed: Vec<u32>,
}

impl Backend for Stub {
    fn describe(&self) -> BackendInfo {
        BackendInfo {
            name: "stub",
            version: "0".into(),
            device: Device::Cpu,
            dtype: Dtype::F32,
            context: 8192,
            hidden_size: 3,
            caps: Caps::empty(),
        }
    }
    fn forward(&self, batch: &[EncodedInput]) -> JevResult<Vec<Hidden>> {
        Ok(batch
            .iter()
            .map(|i| {
                let hit = i.tokens.iter().any(|t| self.entailed.contains(t));
                // Identity head over 3 labels: [contradiction, entailment, neutral].
                Hidden(if hit {
                    vec![0.0, 6.0, 0.0]
                } else {
                    vec![0.0, 0.0, 6.0]
                })
            })
            .collect())
    }
}

/// Identity 3x3, so the hidden state *is* the logit vector and the test's intent is
/// legible from the numbers above.
fn identity_head() -> Head {
    let mut raw = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            let v: f32 = if r == c { 1.0 } else { 0.0 };
            raw.extend_from_slice(&v.to_le_bytes());
        }
    }
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 3], &raw)
        .expect("view");
    let bytes =
        safetensors::serialize(vec![("score.weight".to_string(), view)], None).expect("ser");
    Head::from_bytes(
        &HeadSpec::Linear {
            in_features: 3,
            out_features: 3,
            bias: false,
            tensor: "score.weight".into(),
            repo: "r".into(),
            file: "f".into(),
            revision: None,
            sha256: String::new(),
        },
        &bytes,
    )
    .expect("head")
}

/// A word-level tokenizer over the vocabulary this example actually uses. Every unknown
/// word maps to `unk`, which is not in `ENTAILED_WORDS`, so an unlisted word can never
/// accidentally make a hypothesis true.
fn tokenizer() -> tokenizers::Tokenizer {
    let mut vocab = String::from("\"unk\": 0");
    for (i, w) in ENTAILED_WORDS.iter().enumerate() {
        vocab.push_str(&format!(", \"{w}\": {}", i + 1));
    }
    format!(
        r#"{{
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [], "normalizer": {{"type": "Lowercase"}},
        "pre_tokenizer": {{"type": "Whitespace"}},
        "post_processor": null, "decoder": null,
        "model": {{"type": "WordLevel", "unk_token": "unk", "vocab": {{{vocab}}}}}
    }}"#
    )
    .parse()
    .expect("tokenizer json")
}

fn session() -> Session {
    let mut reg = Registry::default();
    reg.merge_str(
        r#"
[models.stub]
repo = "local/stub"
arch = "stub"
template = "Premise: {premise} Hypothesis: {hypothesis}"
labels = ["contradiction", "entailment", "neutral"]
entailment_label = "entailment"
context = 8192
hidden_size = 3
tokenizer = { repo = "local/stub", file = "tokenizer.json", pad_token_id = 0 }
head = { kind = "linear", in_features = 3, out_features = 3, tensor = "score.weight", repo = "r", file = "f" }
[models.stub.backends.stub]
weights = { repo = "r", file = "w" }
"#,
        std::path::Path::new("<test>"),
    )
    .expect("registry");
    let spec = reg.get("stub").expect("spec").clone();
    let tok = tokenizer();
    let encoder = Encoder::new(&spec, tok.clone()).expect("encoder");
    let entailed = ENTAILED_WORDS
        .iter()
        .map(|w| {
            tok.encode(*w, false)
                .expect("encode")
                .get_ids()
                .first()
                .copied()
                .expect("one id")
        })
        .collect();
    Session::new(spec, encoder, identity_head(), Box::new(Stub { entailed })).expect("session")
}

async fn ready_app() -> axum::Router {
    let cli = <Cli as clap::Parser>::try_parse_from(["openjev", "serve"]).expect("parse");
    let Command::Serve(s) = cli.command else {
        panic!("not serve")
    };
    let mut layered = Layered::load(None, &BTreeMap::new()).expect("defaults");
    let cfg: ServerConfig = build_config(&s, &mut layered).expect("config");

    let shared = Shared::new();
    let engine = Engine::spawn(
        shared.clone(),
        cfg.limits.max_queue,
        cfg.limits.max_batch,
        Box::new(|_| Ok(session())),
    );
    for _ in 0..200 {
        if shared.phase() == Phase::Ready {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(shared.phase(), Phase::Ready, "the stub session must load");
    router(AppState {
        cfg: Arc::new(cfg),
        engine,
        shared,
        metrics: None,
    })
}

async fn systemone(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let resp = app.clone().oneshot(req).await.expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("body");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// Byte for byte the body in the README.
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
async fn the_owners_example_returns_the_documented_shape() {
    let app = ready_app().await;
    let (status, body) = systemone(&app, owners_example()).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    assert_eq!(body["model"], "openjev", "model is echoed, not replaced");
    assert_eq!(
        body["provider"], "openjev",
        "the one deliberate divergence: we never claim to be TypeSafe"
    );
    let id = body["id"].as_str().expect("id");
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(parts.len(), 4, "{id}");
    assert_eq!(&id[..8], "gen-dec-");
    assert!(parts[2].parse::<u64>().is_ok(), "{id}");

    // The usage gap STATUS called out: a real count, produced by the encoder that ran.
    let tokens = body["usage"]["input_tokens"]
        .as_u64()
        .expect("input_tokens");
    assert!(tokens > 0, "input_tokens is still a placeholder zero");
    assert_eq!(body["usage"]["output_tokens"], 0);
    assert_eq!(body["usage"]["cost"], 0.0);

    let urgent = &body["answers"]["urgent"];
    assert_eq!(urgent["type"], "noul");
    let noul = urgent["noul"].as_f64().expect("noul");
    assert!(noul > 0.9, "the stub entails the urgent case: {noul}");
    assert!(urgent.get("confidence").is_none(), "a noul is the number");
    assert!(urgent.get("probabilities").is_none());

    let team = &body["answers"]["team"];
    assert_eq!(team["type"], "choice");
    assert_eq!(team["choice"], "billing");
    let probabilities = team["probabilities"].as_object().expect("probabilities");
    let mut names: Vec<&String> = probabilities.keys().collect();
    names.sort();
    assert_eq!(names, ["billing", "sales", "technical"]);
    let sum: f64 = probabilities
        .values()
        .map(|v| v.as_f64().expect("number"))
        .sum();
    assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {sum}");
    let confidence = team["confidence"].as_f64().expect("confidence");
    assert_eq!(confidence, probabilities["billing"].as_f64().expect("f64"));
    assert!(confidence > 0.9, "{confidence}");
}

/// Reordering the questions, and adding an unrelated one, must not move an answer. This
/// is the property that makes a batched endpoint trustworthy: nothing about how the pairs
/// were packed may reach the numbers.
#[tokio::test]
async fn neither_question_order_nor_extra_questions_change_an_answer() {
    let app = ready_app().await;
    let (_, first) = systemone(&app, owners_example()).await;

    let mut reordered = owners_example();
    let q = reordered["questions"].as_object_mut().expect("questions");
    let urgent = q.remove("urgent").expect("urgent");
    q.insert("urgent".into(), urgent);
    q.insert(
        "aaa_unrelated".into(),
        serde_json::json!({ "type": "score", "instructions": "How long is this?",
                            "criteria": ["short", "medium", "long"] }),
    );
    let (status, second) = systemone(&app, reordered).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(first["answers"]["team"], second["answers"]["team"]);
    assert_eq!(first["answers"]["urgent"], second["answers"]["urgent"]);

    // And the score answer carries everything its spec requires.
    let score = &second["answers"]["aaa_unrelated"];
    assert_eq!(score["type"], "score");
    let s = score["score"].as_f64().expect("score");
    assert!((0.0..=2.0).contains(&s), "score {s} outside 0..levels-1");
    // The legend echoes the rubric as written, not the hypothesis the model was shown.
    assert_eq!(score["legend"]["0"], "short");
    assert_eq!(score["legend"]["2"], "long");
    assert!(score["confidence"].is_number());
}

/// Nothing in the state is relevant to the question. Recorded, not guessed: with no
/// criterion entailed the distribution is uniform and the confidence says so, rather than
/// the server picking a winner and sounding sure about it.
#[tokio::test]
async fn an_irrelevant_question_answers_uniformly_rather_than_confidently() {
    let app = ready_app().await;
    let (status, body) = systemone(
        &app,
        serde_json::json!({
            "state": "The mitochondrion is the powerhouse of the cell.",
            "questions": { "team": { "type": "choice", "instructions": "Which team?",
                "criteria": { "alpha": "one thing", "beta": "another thing" } } }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let team = &body["answers"]["team"];
    assert!((team["confidence"].as_f64().expect("confidence") - 0.5).abs() < 1e-5);
    assert!((team["probabilities"]["alpha"].as_f64().expect("f64") - 0.5).abs() < 1e-5);
}

/// Mutually contradictory criteria: both sides of a noul described so that both are
/// entailed. Recorded: the two-sided contest reports 0.5, where scoring the `true` side
/// alone would have reported ~1.0 and sounded certain.
#[tokio::test]
async fn mutually_contradictory_criteria_come_out_undecided() {
    let app = ready_app().await;
    let (status, body) = systemone(
        &app,
        serde_json::json!({
            "state": "My card was charged twice.",
            "questions": { "q": { "type": "noul", "instructions": "Is it so?",
                // Both contain a word the stub entails, so both sides are "true".
                "criteria": { "true": "refunds apply", "false": "refunds do not apply" } } }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let noul = body["answers"]["q"]["noul"].as_f64().expect("noul");
    assert!((noul - 0.5).abs() < 1e-5, "noul {noul}");
}
