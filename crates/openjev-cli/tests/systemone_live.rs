//! The owner's example, end to end, against a **running** server with real weights.
//!
//! ```text
//! openjev serve &
//! task systemone                    # or: OPENJEV_SERVER_URL=http://127.0.0.1:21131 cargo test ...
//! ```
//!
//! Gated on `OPENJEV_SERVER_URL` and silently green without it, exactly like `golden`:
//! a test that needs a 4.5 GB download is a test nobody runs, and `task check` must stay
//! fast. **Absence of the fixture is absence of evidence** — CI that means to enforce
//! this contract has to set the variable.
//!
//! What is asserted splits in two, deliberately:
//!   * the **shape** — every field the TypeSafe clients read, its type and its range.
//!     This is the compatibility contract and it is asserted exactly.
//!   * the **recorded expectation** — that a customer shouting about a double charge is
//!     routed to `billing` and reads as urgent. That is a claim about the checkpoint, not
//!     about this code, so it is asserted loosely (the argmax and the side of 0.5) and
//!     the numbers are printed rather than pinned.

use serde_json::Value;

fn server() -> Option<String> {
    std::env::var("OPENJEV_SERVER_URL")
        .ok()
        .map(|u| u.trim_end_matches('/').to_string())
}

fn post(url: &str, body: &Value) -> Value {
    let mut req =
        ureq::post(format!("{url}/v1/systemone")).header("content-type", "application/json");
    if let Ok(token) = std::env::var("OPENJEV_TOKEN") {
        req = req.header("authorization", &format!("Bearer {token}"));
    }
    let mut resp = req.send_json(body).expect("POST /v1/systemone");
    assert_eq!(resp.status(), 200, "{:?}", resp.body_mut().read_to_string());
    resp.body_mut().read_json::<Value>().expect("json")
}

/// Byte for byte the body in the README and in `docs/adr/0017`.
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

#[test]
fn the_owners_example_returns_the_documented_shape() {
    let Some(url) = server() else {
        eprintln!("OPENJEV_SERVER_URL unset — skipping (this proves nothing)");
        return;
    };
    let body = post(&url, &owners_example());
    eprintln!("{}", serde_json::to_string_pretty(&body).expect("pretty"));

    assert_eq!(body["model"], "openjev", "model must be echoed");
    assert_eq!(body["provider"], "openjev", "we never claim to be TypeSafe");
    let id = body["id"].as_str().expect("id");
    assert!(id.starts_with("gen-dec-"), "{id}");

    // usage: a real input count, an honest zero for the two that a local cross-encoder
    // cannot report.
    let tokens = body["usage"]["input_tokens"]
        .as_u64()
        .expect("input_tokens");
    assert!(tokens > 0, "input_tokens must be counted, not defaulted");
    assert_eq!(body["usage"]["output_tokens"], 0);
    assert_eq!(body["usage"]["cost"], 0.0);

    // answers: exactly the keys asked, nothing else.
    let answers = body["answers"].as_object().expect("answers");
    let mut keys: Vec<&String> = answers.keys().collect();
    keys.sort();
    assert_eq!(keys, ["team", "urgent"]);

    let urgent = &answers["urgent"];
    assert_eq!(urgent["type"], "noul");
    let noul = urgent["noul"].as_f64().expect("noul");
    assert!((0.0..=1.0).contains(&noul), "noul {noul}");
    // A noul carries the probability and nothing else — no confidence, no distribution.
    assert!(urgent.get("confidence").is_none());

    let team = &answers["team"];
    assert_eq!(team["type"], "choice");
    let probabilities = team["probabilities"].as_object().expect("probabilities");
    let mut names: Vec<&String> = probabilities.keys().collect();
    names.sort();
    assert_eq!(names, ["billing", "sales", "technical"]);
    let sum: f64 = probabilities
        .values()
        .map(|v| v.as_f64().expect("number"))
        .sum();
    assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {sum}");
    let choice = team["choice"].as_str().expect("choice");
    let confidence = team["confidence"].as_f64().expect("confidence");
    assert_eq!(
        confidence,
        probabilities[choice].as_f64().expect("number"),
        "confidence must be the winner's own probability"
    );

    // The recorded expectation. Loose on purpose: this is the checkpoint's judgement.
    assert_eq!(choice, "billing", "a double charge is a billing matter");
    assert!(noul > 0.5, "\"ASAP\" should read as urgent, got {noul}");
}

#[test]
fn a_bad_question_is_a_clean_4xx_from_the_real_server_too() {
    let Some(url) = server() else { return };
    let body = serde_json::json!({
        "state": "anything",
        "questions": { "q": { "type": "choice", "criteria": { "only": "one option" } } }
    });
    let mut resp = ureq::post(format!("{url}/v1/systemone"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("content-type", "application/json")
        .send_json(&body)
        .expect("send");
    assert_eq!(resp.status(), 422);
    let e: Value = resp.body_mut().read_json().expect("json");
    assert_eq!(e["error"]["code"], "invalid_question");
}
