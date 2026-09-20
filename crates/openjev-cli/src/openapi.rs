//! `/openapi.json`, built from the same limits the server enforces.
//!
//! Generated rather than hand-maintained so clients can be generated too — the whole
//! reason there is no OpenAI-shaped shim (design 02 §3.10) is that a generated client is
//! better than a mistranslated one. Drift is caught by `tests/http.rs`, which asserts
//! every path listed here is actually routed.

use crate::api::API_VERSION;
use crate::server::{Auth, SERVER_VERSION, ServerConfig};
use serde_json::{Value, json};

/// Every path the server serves, and the methods on it. One table, used by the document
/// and by the drift test.
pub const ROUTES: &[(&str, &str)] = &[
    ("/healthz", "get"),
    ("/readyz", "get"),
    ("/v1/info", "get"),
    ("/v1/model", "get"),
    ("/v1/events", "get"),
    ("/v1/predict", "post"),
    ("/v1/rerank", "post"),
    ("/v1/grade", "post"),
    ("/v1/systemone", "post"),
    ("/v1/latents", "post"),
    ("/metrics", "get"),
    ("/openapi.json", "get"),
];

fn op(summary: &str, body: Option<&str>, ok: &str) -> Value {
    let mut o = json!({
        "summary": summary,
        "responses": {
            "200": { "description": "ok", "content": { "application/json": { "schema": { "$ref": format!("#/components/schemas/{ok}") } } } },
            "default": { "description": "error envelope", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorEnvelope" } } } }
        }
    });
    if let Some(b) = body {
        o["requestBody"] = json!({
            "required": true,
            "content": { "application/json": { "schema": { "$ref": format!("#/components/schemas/{b}") } } }
        });
    }
    o
}

pub fn document(cfg: &ServerConfig) -> Value {
    let secured = matches!(cfg.auth, Auth::Bearer { .. });
    let mut paths = json!({
        "/healthz": { "get": op("liveness: 200 whenever the process can answer", None, "Health") },
        "/readyz":  { "get": op("readiness: 200 only when the model can serve", None, "ReadyResponse") },
        "/v1/info": { "get": op("version, capabilities and limits", None, "InfoResponse") },
        "/v1/model": { "get": op("the loaded model", None, "ModelInfo") },
        "/v1/events": { "get": op("SSE lifecycle stream", None, "Event") },
        "/v1/predict": { "post": op("pairs to label probabilities", Some("PredictRequest"), "PredictResponse") },
        "/v1/rerank": { "post": op("question plus options to a ranking", Some("RerankRequest"), "RerankResponse") },
        "/v1/grade": { "post": op("answer against reference", Some("GradeRequest"), "GradeResponse") },
        "/v1/systemone": { "post": op("TypeSafe System One: one state, typed questions, typed answers", Some("SystemOneRequest"), "SystemOneResponse") },
        "/metrics": { "get": op("Prometheus text exposition", None, "Metrics") },
        "/openapi.json": { "get": op("this document", None, "OpenAPI") },
    });
    paths["/v1/latents"] = json!({
        "post": op("pooled hidden states (off unless server.enable_latents)", Some("LatentsRequest"), "LatentsResponse")
    });

    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "openjev",
            "version": SERVER_VERSION,
            "description": format!(
                "NLI cross-encoder. api_version {API_VERSION}. Clients gate on api_version and \
                 feature-detect on /v1/info capabilities, never on server_version."
            )
        },
        "paths": paths,
        "security": if secured { json!([{ "bearerAuth": [] }]) } else { json!([]) },
        "components": {
            "securitySchemes": { "bearerAuth": { "type": "http", "scheme": "bearer" } },
            "schemas": schemas(cfg)
        }
    })
}

fn schemas(cfg: &ServerConfig) -> Value {
    let scores = json!({ "type": "object", "additionalProperties": { "type": "number" } });
    json!({
        "Health": { "type": "object", "properties": { "ok": {"type":"boolean"}, "phase": {"type":"string"} } },
        "ReadyResponse": { "type": "object", "properties": {
            "ready": {"type":"boolean"},
            "phase": {"type":"string","enum":["starting","resolving","downloading","loading","ready","draining","failed"]},
            "detail": {"type":"object"}, "since": {"type":"string"} } },
        "Pair": { "type": "object", "required": ["premise","hypothesis"], "properties": {
            "premise": {"type":"string","maxLength": cfg.limits.max_field_chars},
            "hypothesis": {"type":"string","maxLength": cfg.limits.max_field_chars},
            "id": {"type":"string"} } },
        "PredictRequest": { "type": "object", "required": ["pairs"], "properties": {
            "pairs": {"type":"array","maxItems": cfg.limits.max_pairs,"items":{"$ref":"#/components/schemas/Pair"}},
            "truncate": {"type":"string","enum":["error","tail"],"default":"error"} } },
        "PredictResponse": { "type": "object", "properties": {
            "object": {"type":"string"}, "model": {"type":"string"}, "revision": {"type":"string"},
            "results": {"type":"array","items":{"type":"object","properties":{
                "id":{"type":"string"},"index":{"type":"integer"},"label":{"type":"string"},"scores": scores }}},
            "usage": {"$ref":"#/components/schemas/Usage"} } },
        "RerankRequest": { "type": "object", "required": ["question","options"], "properties": {
            "question": {"type":"string"},
            "options": {"type":"array","maxItems": cfg.limits.max_options,"items":{"type":"string"}},
            "top_k": {"type":"integer"}, "return_documents": {"type":"boolean","default":false} } },
        "RerankResponse": { "type": "object", "properties": {
            "object": {"type":"string"},
            "results": {"type":"array","items":{"type":"object","properties":{
                "rank":{"type":"integer"},"index":{"type":"integer"},"score":{"type":"number"},"text":{"type":"string"}}}},
            "usage": {"$ref":"#/components/schemas/Usage"} } },
        "GradeRequest": { "type": "object", "required": ["answer","reference"], "properties": {
            "answer": {"type":"string"}, "reference": {"type":"string"}, "threshold": {"type":"number","default":0.5} } },
        "GradeResponse": { "type": "object", "properties": {
            "object": {"type":"string"}, "label": {"type":"string"}, "scores": scores,
            "pass": {"type":"boolean"}, "threshold": {"type":"number"},
            "usage": {"$ref":"#/components/schemas/Usage"} } },
        "Question": { "type": "object", "required": ["type"], "properties": {
            "type": {"type":"string","enum":["noul","boolean","choice","score"]},
            "instructions": {"description":"string, or any JSON object/array"},
            "criteria": {"description": format!(
                "noul/boolean: an object of true/false descriptions; choice: key -> description; \
                 score: an ordered array of levels, lowest first. At most {} entries.",
                cfg.limits.max_criteria)} } },
        "SystemOneRequest": { "type": "object", "required": ["state","questions"], "properties": {
            "model": {"type":"string"},
            "state": {"description":"string, or any JSON object/array; the premise"},
            "questions": {"type":"object","maxProperties": cfg.limits.max_questions,
                          "additionalProperties": {"$ref":"#/components/schemas/Question"}} } },
        "SystemOneResponse": { "type": "object", "required": ["model","answers","usage","id","provider"], "properties": {
            "model": {"type":"string"},
            "answers": {"type":"object","additionalProperties": {"type":"object","properties": {
                "type": {"type":"string","enum":["noul","boolean","choice","score"]},
                "noul": {"type":"number","minimum":0,"maximum":1},
                "probability": {"type":"number","minimum":0,"maximum":1},
                "choice": {"type":"string"},
                "score": {"type":"number"},
                "legend": {"type":"object","additionalProperties":{"type":"string"}},
                "probabilities": {"type":"object","additionalProperties":{"type":"number"}},
                "confidence": {"type":"number","minimum":0,"maximum":1} }}},
            "usage": {"type":"object","properties": {
                "input_tokens": {"type":"integer"},
                "output_tokens": {"type":"integer","const":0,"description":"a cross-encoder emits no tokens"},
                "cost": {"type":"number","const":0,"description":"local inference is free"} }},
            "id": {"type":"string","pattern":"^gen-dec-[0-9]+-[A-Za-z0-9]{20}$"},
            "provider": {"type":"string","const":"openjev","description":"openjev, never TypeSafe — see docs/adr/0017"} } },
        "LatentsRequest": { "type": "object", "required": ["texts"], "properties": {
            "texts": {"type":"array","items":{"type":"string"}} } },
        "LatentsResponse": { "type": "object", "properties": {
            "object": {"type":"string"}, "dim": {"type":"integer"},
            "vectors": {"type":"array","items":{"type":"array","items":{"type":"number"}}} } },
        "ModelInfo": { "type": "object", "properties": {
            "model": {"type":"string"}, "revision": {"type":"string"}, "device": {"type":"string"},
            "dtype": {"type":"string"}, "backend": {"type":"string"}, "context": {"type":"integer"},
            "labels": {"type":"array","items":{"type":"string"}}, "entailment_label": {"type":"string"} } },
        "InfoResponse": { "type": "object", "properties": {
            "api_version": {"type":"integer"}, "server_version": {"type":"string"},
            "capabilities": {"type":"array","items":{"type":"string"}},
            "limits": {"type":"object"}, "model": {"$ref":"#/components/schemas/ModelInfo"} } },
        "Usage": { "type": "object", "properties": {
            "pairs": {"type":"integer"}, "tokens": {"type":"integer"},
            "queue_ms": {"type":"integer"}, "compute_ms": {"type":"integer"} } },
        "Event": { "type": "object", "properties": { "event": {"type":"string"}, "data": {"type":"object"} } },
        "Metrics": { "type": "string" },
        "OpenAPI": { "type": "object" },
        "ErrorEnvelope": { "type": "object", "required": ["error"], "properties": { "error": {
            "type": "object", "required": ["code","message"], "properties": {
                "code": {"type":"string","enum":[
                    "invalid_request","unsupported_media_type","unauthorized","not_found",
                    "unprocessable","invalid_question","unknown_question_type","empty_criteria",
                    "too_many_questions","state_too_long",
                    "payload_too_large","queue_full","timeout","canceled",
                    "model_not_ready","device_error","shutting_down","internal"]},
                "message": {"type":"string"}, "detail": {"type":"object"}, "request_id": {"type":"string"} } } } }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ServerConfig {
        let args = <crate::cli::Cli as clap::Parser>::try_parse_from(["openjev", "serve"]).unwrap();
        let crate::cli::Command::Serve(s) = args.command else {
            panic!()
        };
        let mut layered =
            crate::config::Layered::load(None, &std::collections::BTreeMap::new()).unwrap();
        crate::server::build_config(&s, &mut layered).unwrap()
    }

    #[test]
    fn the_document_covers_every_route_in_the_table() {
        let doc = document(&cfg());
        for (path, method) in ROUTES {
            assert!(
                doc["paths"][path][method].is_object(),
                "{method} {path} is routed but undocumented"
            );
        }
    }

    #[test]
    fn the_documented_limits_are_the_enforced_limits() {
        let c = cfg();
        let doc = document(&c);
        assert_eq!(
            doc["components"]["schemas"]["PredictRequest"]["properties"]["pairs"]["maxItems"],
            c.limits.max_pairs
        );
        assert_eq!(
            doc["components"]["schemas"]["SystemOneRequest"]["properties"]["questions"]["maxProperties"],
            c.limits.max_questions
        );
    }

    #[test]
    fn every_error_code_the_server_can_emit_is_in_the_schema() {
        let doc = document(&cfg());
        let listed = doc["components"]["schemas"]["ErrorEnvelope"]["properties"]["error"]
            ["properties"]["code"]["enum"]
            .clone();
        let listed: Vec<String> = serde_json::from_value(listed).unwrap();
        for c in [
            crate::api::Code::InvalidQuestion,
            crate::api::Code::UnknownQuestionType,
            crate::api::Code::EmptyCriteria,
            crate::api::Code::TooManyQuestions,
            crate::api::Code::StateTooLong,
            crate::api::Code::InvalidRequest,
            crate::api::Code::Unauthorized,
            crate::api::Code::QueueFull,
            crate::api::Code::ModelNotReady,
            crate::api::Code::ShuttingDown,
            crate::api::Code::Internal,
        ] {
            assert!(listed.contains(&c.as_str().to_string()), "{}", c.as_str());
        }
    }
}
