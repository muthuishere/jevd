//! `POST /v1/systemone` — the TypeSafe System One wire shape, served by an NLI
//! cross-encoder.
//!
//! This module is the whole protocol *except* the forward pass: parsing, validation,
//! hypothesis construction and the arithmetic that turns `P(entailment)` into an answer.
//! It is pure and synchronous on purpose — every rule below is a unit test away, and
//! none of them needs weights.
//!
//! **Compatibility is the feature.** Existing clients (`jev-model-router`,
//! `fast-jev-compaction`) point at this by changing a base URL and nothing else, so the
//! field names here are copied from what those clients read, not invented. The one
//! deliberate divergence is `provider`, which reports `openjev` — see `docs/adr/0017`.
//!
//! The premise is always the `state`. A question becomes one hypothesis per criterion,
//! and the answer is built from `P(entailment)` of each — which is the same quantity
//! `rerank` and `grade` already rank by, so there is one notion of "how true" in the
//! whole server.

use crate::api::{Code, Limits};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// What the caller sent. `model` is optional and echoed back: the Vercel Gateway variant
/// of this protocol carries the model in a header instead of the body, and refusing that
/// body would break a client for no gain.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemOneRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// **Not a string.** `docs.typesafe.ai/api.md` types `state` as string, object or
    /// array, and `fast-jev-compaction` sends an object
    /// (`{context, goal, history:[...]}`). Taking `String` here would 400 a shipped
    /// client on its first request.
    #[serde(default)]
    pub state: Value,
    /// A `BTreeMap`, so the evaluation order is the key order and is the same for two
    /// requests that differ only in how the JSON object was written. Question order
    /// cannot change an answer because there is no order to carry.
    #[serde(default)]
    pub questions: BTreeMap<String, Question>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Question {
    /// Kept as a string rather than an enum: an unknown type must come back as
    /// `unknown_question_type` naming what was sent, not as a serde parse failure that
    /// says `unknown variant` about a field the caller cannot see.
    #[serde(rename = "type")]
    pub kind: String,
    /// Also any JSON: the "structured instructions" of
    /// `docs.typesafe.ai/primitives/advanced.md` are an object whose fields the question
    /// refers to by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Criteria>,
}

/// `choice` and `noul` key their criteria; `score` lists them, because a rubric is
/// ordered and an object's order is not a thing JSON promises.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Criteria {
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

/// How a JSON value becomes text for the model.
///
/// A string is itself; anything else is compact JSON. Not pretty-printed and not a
/// bespoke rendering: the model sees the caller's structure verbatim, and two callers
/// who sent the same object get the same tokens. `null` renders empty, which is how a
/// `"false": null` criterion means "no description", not the literal word null.
pub fn render(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------- the plan

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A single probability in [0,1]. TypeSafe's own name for a yes/no question.
    Noul,
    /// The same question in the Vercel Gateway's schema, answered as `probability`.
    Boolean,
    Choice,
    Score,
}

/// One question, reduced to the pairs it needs and the keys those pairs answer for.
#[derive(Debug, Clone)]
pub struct Planned {
    pub key: String,
    pub kind: Kind,
    /// The option key per hypothesis. For `noul`/`boolean` these are `true` and, when the
    /// caller supplied it, `false`. For `score` they are the rubric index as a string.
    pub options: Vec<String>,
    pub hypotheses: Vec<String>,
    /// `score` only: the rubric as the caller wrote it, echoed back as the answer's
    /// `legend`.
    pub legend: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub items: Vec<Planned>,
}

impl Plan {
    pub fn hypotheses(&self) -> usize {
        self.items.iter().map(|i| i.hypotheses.len()).sum()
    }
}

/// A validation failure, as the code the client contracts on plus the sentence a human
/// reads. Turned into the server's one error envelope by the handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invalid {
    pub code: Code,
    pub message: String,
}

fn bad(code: Code, message: impl Into<String>) -> Invalid {
    Invalid {
        code,
        message: message.into(),
    }
}

/// The hypothesis a criterion becomes.
///
/// One template for every type, because two templates would be two calibrations: the
/// instructions frame the question and the criterion is the claim being tested against
/// the state. Both are trimmed; core trims again at render time (ADR 0011) and agreeing
/// with it here keeps the length we count equal to the length we run.
fn hypothesis(instructions: Option<&str>, criterion: Option<&str>) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if let Some(i) = instructions.map(str::trim).filter(|s| !s.is_empty()) {
        parts.push(i);
    }
    if let Some(c) = criterion.map(str::trim).filter(|s| !s.is_empty()) {
        parts.push(c);
    }
    parts.join(" ")
}

/// The premise plus the pairs to run it against.
pub fn plan(req: &SystemOneRequest, limits: &Limits) -> Result<(String, Plan), Invalid> {
    let premise = render(&req.state);
    if premise.trim().is_empty() {
        return Err(bad(Code::Unprocessable, "state is empty"));
    }
    let chars = premise.chars().count();
    if chars > limits.max_field_chars {
        return Err(bad(
            Code::StateTooLong,
            format!(
                "state is {chars} characters, limit is {}",
                limits.max_field_chars
            ),
        ));
    }
    if req.questions.is_empty() {
        return Err(bad(Code::Unprocessable, "questions is empty"));
    }
    if req.questions.len() > limits.max_questions {
        return Err(bad(
            Code::TooManyQuestions,
            format!(
                "{} questions, limit is {}",
                req.questions.len(),
                limits.max_questions
            ),
        ));
    }
    let mut items = Vec::with_capacity(req.questions.len());
    for (key, q) in &req.questions {
        items.push(plan_one(key, q, limits)?);
    }
    Ok((premise, Plan { items }))
}

fn plan_one(key: &str, q: &Question, limits: &Limits) -> Result<Planned, Invalid> {
    let kind = match q.kind.as_str() {
        "noul" => Kind::Noul,
        "boolean" => Kind::Boolean,
        "choice" => Kind::Choice,
        "score" => Kind::Score,
        other => {
            return Err(bad(
                Code::UnknownQuestionType,
                format!(
                    "questions.{key}.type '{other}' is not one of noul, boolean, choice, score"
                ),
            ));
        }
    };
    let instructions = q.instructions.as_ref().map(render);
    let instructions = instructions.as_deref();
    let (options, criteria): (Vec<String>, Vec<Option<String>>) = match (kind, &q.criteria) {
        // A yes/no question may carry no criteria at all — `jev-model-router` asks its
        // `risky` question with instructions alone, and refusing that breaks a shipped
        // client.
        (Kind::Noul | Kind::Boolean, None) => (vec!["true".into()], vec![None]),
        (Kind::Noul | Kind::Boolean, Some(c)) => {
            let m = as_map(key, c)?;
            if m.is_empty() {
                return Err(bad(
                    Code::EmptyCriteria,
                    format!("questions.{key}.criteria is empty"),
                ));
            }
            for k in m.keys() {
                if k != "true" && k != "false" {
                    return Err(bad(
                        Code::InvalidQuestion,
                        format!(
                            "questions.{key}.criteria key '{k}' — a {} takes only 'true' and 'false'",
                            q.kind
                        ),
                    ));
                }
            }
            let t = m.get("true").ok_or_else(|| {
                bad(
                    Code::InvalidQuestion,
                    format!("questions.{key}.criteria has no 'true' case"),
                )
            })?;
            // `null` is a documented criterion value and means "no description", so it
            // falls back to the instructions rather than sending the model the word
            // "null". A described `true` with an undescribed `false` would be a rigged
            // contest, so the contest only happens when both sides are described.
            let t = render(t);
            let f = m.get("false").map(render);
            match (t.trim().is_empty(), f.as_deref().map(str::trim)) {
                (false, Some(f)) if !f.is_empty() => (
                    vec!["true".into(), "false".into()],
                    vec![Some(t.clone()), Some(f.to_string())],
                ),
                (false, _) => (vec!["true".into()], vec![Some(t.clone())]),
                (true, _) => (vec!["true".into()], vec![None]),
            }
        }
        (Kind::Choice, None) => {
            return Err(bad(
                Code::EmptyCriteria,
                format!("questions.{key}.criteria is required for a choice"),
            ));
        }
        (Kind::Choice, Some(c)) => {
            let m = as_map(key, c)?;
            if m.is_empty() {
                return Err(bad(
                    Code::EmptyCriteria,
                    format!("questions.{key}.criteria is empty"),
                ));
            }
            // One option is not a choice. Answering it would report the only key with
            // confidence 1 whatever the state says — a confident answer that measured
            // nothing, which is worse than a 4xx.
            if m.len() < 2 {
                return Err(bad(
                    Code::InvalidQuestion,
                    format!(
                        "questions.{key} is a choice with {} criterion; a choice needs at least two",
                        m.len()
                    ),
                ));
            }
            check_count(key, m.len(), limits)?;
            let mut opts = Vec::new();
            let mut crit = Vec::new();
            for (k, v) in &m {
                opts.push(k.clone());
                crit.push(Some(check_criterion(key, k, &render(v))?));
            }
            (opts, crit)
        }
        (Kind::Score, None) => {
            return Err(bad(
                Code::EmptyCriteria,
                format!("questions.{key}.criteria is required for a score: it is the rubric"),
            ));
        }
        (Kind::Score, Some(c)) => {
            let rubric = as_rubric(key, c)?;
            if rubric.is_empty() {
                return Err(bad(
                    Code::EmptyCriteria,
                    format!("questions.{key}.criteria is empty"),
                ));
            }
            if rubric.len() < 2 {
                return Err(bad(
                    Code::InvalidQuestion,
                    format!(
                        "questions.{key} is a score with one rubric step; a scale needs at least two"
                    ),
                ));
            }
            check_count(key, rubric.len(), limits)?;
            let mut opts = Vec::new();
            let mut crit = Vec::new();
            for (i, v) in rubric.iter().enumerate() {
                let name = i.to_string();
                crit.push(Some(check_criterion(key, &name, &render(v))?));
                opts.push(name);
            }
            (opts, crit)
        }
    };

    let hypotheses: Vec<String> = criteria
        .iter()
        .map(|c| hypothesis(instructions, c.as_deref()))
        .collect();
    for (h, o) in hypotheses.iter().zip(&options) {
        if h.trim().is_empty() {
            return Err(bad(
                Code::InvalidQuestion,
                format!("questions.{key} has neither instructions nor a usable criterion"),
            ));
        }
        if h.chars().count() > limits.max_field_chars {
            return Err(bad(
                Code::Unprocessable,
                format!(
                    "questions.{key}.criteria.{o} plus its instructions is longer than the {}-character limit",
                    limits.max_field_chars
                ),
            ));
        }
    }
    let legend = if kind == Kind::Score {
        options
            .iter()
            .cloned()
            .zip(criteria.iter().map(|c| c.clone().unwrap_or_default()))
            .collect()
    } else {
        BTreeMap::new()
    };
    Ok(Planned {
        key: key.to_string(),
        kind,
        options,
        hypotheses,
        legend,
    })
}

fn check_count(key: &str, n: usize, limits: &Limits) -> Result<(), Invalid> {
    if n > limits.max_criteria {
        return Err(bad(
            Code::PayloadTooLarge,
            format!(
                "questions.{key} has {n} criteria, limit is {}",
                limits.max_criteria
            ),
        ));
    }
    Ok(())
}

fn check_criterion(key: &str, name: &str, text: &str) -> Result<String, Invalid> {
    if text.trim().is_empty() {
        return Err(bad(
            Code::EmptyCriteria,
            format!("questions.{key}.criteria.{name} is an empty description"),
        ));
    }
    Ok(text.to_string())
}

fn as_map(key: &str, c: &Criteria) -> Result<BTreeMap<String, Value>, Invalid> {
    match c {
        Criteria::Map(m) => Ok(m.clone()),
        Criteria::List(_) => Err(bad(
            Code::InvalidQuestion,
            format!("questions.{key}.criteria must be an object of key -> description"),
        )),
    }
}

/// A rubric, lowest step first. An array is the shape `jev-model-router` sends. An object
/// is accepted only when every key is an integer, because then the order is recoverable;
/// any other object would make the score depend on key spelling.
fn as_rubric(key: &str, c: &Criteria) -> Result<Vec<Value>, Invalid> {
    match c {
        Criteria::List(v) => Ok(v.clone()),
        Criteria::Map(m) => {
            let mut numbered: Vec<(i64, Value)> = Vec::with_capacity(m.len());
            for (k, v) in m {
                let n = k.parse::<i64>().map_err(|_| {
                    bad(
                        Code::InvalidQuestion,
                        format!(
                            "questions.{key}.criteria must be an array of rubric steps, or an object keyed by integers; '{k}' is neither"
                        ),
                    )
                })?;
                numbered.push((n, v.clone()));
            }
            numbered.sort_by_key(|(n, _)| *n);
            Ok(numbered.into_iter().map(|(_, v)| v).collect())
        }
    }
}

// ---------------------------------------------------------------- the answers

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f32,
    },
    Boolean {
        probability: f32,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        /// Required by `docs.typesafe.ai/primitives/score.md`: the rubric echoed back,
        /// index -> description, so a reader of the answer can tell what 1.05 means
        /// without holding the request.
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
}

/// `P(entailment)` per hypothesis, in `plan.items` order, flattened.
///
/// Turning them into answers is arithmetic and nothing else — which is why the label
/// index this comes from is resolved by name at the call site and never guessed here.
pub fn answers(plan: &Plan, entailment: &[f32]) -> BTreeMap<String, Answer> {
    let mut out = BTreeMap::new();
    let mut at = 0usize;
    for item in &plan.items {
        let n = item.hypotheses.len();
        let scores = &entailment[at..(at + n).min(entailment.len())];
        at += n;
        out.insert(item.key.clone(), answer_one(item, scores));
    }
    out
}

fn answer_one(item: &Planned, scores: &[f32]) -> Answer {
    match item.kind {
        Kind::Noul | Kind::Boolean => {
            let p = binary_probability(scores);
            if item.kind == Kind::Noul {
                Answer::Noul { noul: p }
            } else {
                Answer::Boolean { probability: p }
            }
        }
        Kind::Choice => {
            let probabilities = normalise(&item.options, scores);
            let (choice, confidence) = argmax(&item.options, &probabilities);
            Answer::Choice {
                choice,
                probabilities,
                confidence,
            }
        }
        Kind::Score => {
            let probabilities = normalise(&item.options, scores);
            // The expected step, not the argmax. `jev-model-router` rounds what it gets
            // (`effortLevel` does `Math.round`), so a fractional score is the shape the
            // client already handles, and it keeps the information that "between 1 and 2"
            // is not the same answer as "solidly 2".
            let score = item
                .options
                .iter()
                .enumerate()
                .map(|(i, k)| i as f32 * probabilities.get(k).copied().unwrap_or(0.0))
                .sum();
            let (_, confidence) = argmax(&item.options, &probabilities);
            Answer::Score {
                score,
                legend: item.legend.clone(),
                probabilities,
                confidence,
            }
        }
    }
}

/// `P(true)`.
///
/// With only a `true` criterion there is nothing to contest, so the honest number is the
/// raw `P(entailment)` of that one hypothesis. With both cases described, the question
/// really is a two-way contest and normalising cancels the model's global entailment
/// bias, which a single hypothesis cannot.
fn binary_probability(scores: &[f32]) -> f32 {
    match scores {
        [t] => clamp01(*t),
        [t, f] => {
            let (t, f) = (clamp01(*t), clamp01(*f));
            let sum = t + f;
            if sum > 0.0 { t / sum } else { 0.5 }
        }
        _ => 0.0,
    }
}

/// L1, not softmax. `P(entailment)` is already a probability per option; a softmax over
/// probabilities would apply a temperature nobody chose and flatten a decisive answer.
/// An all-zero vector goes uniform rather than dividing by zero — a uniform distribution
/// is the honest statement that nothing was entailed.
fn normalise(keys: &[String], scores: &[f32]) -> BTreeMap<String, f32> {
    let vals: Vec<f32> = keys
        .iter()
        .enumerate()
        .map(|(i, _)| clamp01(scores.get(i).copied().unwrap_or(0.0)))
        .collect();
    let sum: f32 = vals.iter().sum();
    let n = keys.len().max(1) as f32;
    keys.iter()
        .cloned()
        .zip(
            vals.iter()
                .map(|v| if sum > 0.0 { v / sum } else { 1.0 / n }),
        )
        .collect()
}

/// The winner and its probability. Ties resolve to the first key in `keys` order, which
/// is sorted, so two identical requests never disagree.
fn argmax(keys: &[String], probabilities: &BTreeMap<String, f32>) -> (String, f32) {
    keys.iter()
        .map(|k| (k.clone(), probabilities.get(k).copied().unwrap_or(0.0)))
        .fold((String::new(), f32::NEG_INFINITY), |best, cur| {
            if cur.1 > best.1 { cur } else { best }
        })
}

fn clamp01(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

// ---------------------------------------------------------------- the envelope

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneUsage {
    /// Real, from the same encoder the forward pass uses.
    pub input_tokens: usize,
    /// Always 0. A cross-encoder emits no tokens — it emits one distribution over labels
    /// per pair. Any other number here would be a fabrication dressed as a measurement.
    pub output_tokens: usize,
    /// Always 0. This runs on your machine.
    pub cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: SystemOneUsage,
    pub id: String,
    /// `openjev`, never `TypeSafe`. See `docs/adr/0017`.
    pub provider: String,
}

pub const PROVIDER: &str = "openjev";

/// `gen-dec-<unix>-<20 chars>`, the shape TypeSafe emits, because clients log and dedupe
/// on it and a differently shaped id is a differently shaped log line.
pub fn generation_id() -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    use rand::Rng;
    let mut bytes = [0u8; 20];
    rand::rng().fill(&mut bytes);
    let suffix: String = bytes
        .iter()
        .map(|b| ALPHABET[*b as usize % ALPHABET.len()] as char)
        .collect();
    format!("gen-dec-{secs}-{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        Limits {
            max_body_bytes: 1_048_576,
            max_pairs: 256,
            max_options: 512,
            max_field_chars: 32_768,
            max_queue: 64,
            max_batch: 32,
            max_questions: 32,
            max_criteria: 255,
            request_timeout_secs: 60,
        }
    }

    fn req(json: serde_json::Value) -> SystemOneRequest {
        serde_json::from_value(json).expect("parse")
    }

    fn owners_example() -> SystemOneRequest {
        req(serde_json::json!({
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
        }))
    }

    #[test]
    fn the_owners_example_plans_five_hypotheses_against_one_state() {
        let (_, p) = plan(&owners_example(), &limits()).expect("plan");
        assert_eq!(p.items.len(), 2);
        assert_eq!(p.hypotheses(), 5);
        let team = p.items.iter().find(|i| i.key == "team").expect("team");
        // Sorted, so the pair order is a function of the request's content and not of
        // how its JSON happened to be serialised.
        assert_eq!(team.options, ["billing", "sales", "technical"]);
        assert_eq!(
            team.hypotheses[0],
            "Which team should handle this? Payments and refunds"
        );
    }

    #[test]
    fn a_noul_without_criteria_is_allowed_because_a_shipped_client_sends_one() {
        // jev-model-router's `risky` question: instructions only, no criteria.
        let r = req(serde_json::json!({
            "state": "deploy to prod",
            "questions": { "risky": { "type": "noul", "instructions": "Would this change production?" } }
        }));
        let (_, p) = plan(&r, &limits()).expect("plan");
        assert_eq!(p.items[0].options, ["true"]);
        assert_eq!(p.items[0].hypotheses, ["Would this change production?"]);
    }

    #[test]
    fn a_score_rubric_is_an_array_and_scores_to_its_indices() {
        let r = req(serde_json::json!({
            "state": "rename a variable",
            "questions": { "effort": {
                "type": "score",
                "instructions": "How much reasoning?",
                "criteria": ["almost none", "some", "a lot", "as much as possible"]
            }}
        }));
        let (_, p) = plan(&r, &limits()).expect("plan");
        assert_eq!(p.items[0].options, ["0", "1", "2", "3"]);
        let a = answers(&p, &[0.9, 0.1, 0.0, 0.0]);
        let Answer::Score { score, .. } = &a["effort"] else {
            panic!("not a score")
        };
        assert!(*score < 0.5, "score {score}");
    }

    #[test]
    fn a_score_rubric_keyed_by_integers_keeps_numeric_order_not_string_order() {
        // "10" sorts before "2" as a string. If that ordering reached the score, the
        // rubric would silently be scrambled for any scale longer than ten steps.
        let mut m = BTreeMap::new();
        for i in 0..11 {
            m.insert(i.to_string(), Value::from(format!("step {i}")));
        }
        let rubric = as_rubric("k", &Criteria::Map(m)).expect("rubric");
        assert_eq!(render(&rubric[2]), "step 2");
        assert_eq!(render(&rubric[10]), "step 10");
    }

    #[test]
    fn a_choice_with_one_criterion_is_refused_rather_than_answered_confidently() {
        let r = req(serde_json::json!({
            "state": "s",
            "questions": { "team": { "type": "choice", "criteria": { "billing": "money" } } }
        }));
        let e = plan(&r, &limits()).expect_err("must refuse");
        assert_eq!(e.code, Code::InvalidQuestion);
        assert_eq!(e.code.status(), 422);
    }

    #[test]
    fn an_empty_criterion_description_is_refused() {
        for criteria in [
            serde_json::json!({ "a": "", "b": "fine" }),
            serde_json::json!({ "a": "   ", "b": "fine" }),
        ] {
            let r = req(serde_json::json!({
                "state": "s",
                "questions": { "q": { "type": "choice", "criteria": criteria } }
            }));
            assert_eq!(
                plan(&r, &limits()).expect_err("must refuse").code,
                Code::EmptyCriteria
            );
        }
    }

    #[test]
    fn empty_criteria_and_missing_criteria_are_both_refused_for_a_choice() {
        for criteria in [Some(serde_json::json!({})), None] {
            let mut q = serde_json::json!({ "type": "choice" });
            if let Some(c) = criteria {
                q["criteria"] = c;
            }
            let r = req(serde_json::json!({ "state": "s", "questions": { "q": q } }));
            assert_eq!(
                plan(&r, &limits()).expect_err("must refuse").code,
                Code::EmptyCriteria
            );
        }
    }

    #[test]
    fn an_unknown_type_names_itself_in_the_error() {
        let r = req(serde_json::json!({
            "state": "s",
            "questions": { "q": { "type": "vibes", "instructions": "?" } }
        }));
        let e = plan(&r, &limits()).expect_err("must refuse");
        assert_eq!(e.code, Code::UnknownQuestionType);
        assert!(e.message.contains("vibes"), "{}", e.message);
    }

    #[test]
    fn the_limits_are_enforced_with_their_own_codes() {
        let l = limits();
        let mut questions = serde_json::Map::new();
        for i in 0..(l.max_questions + 1) {
            questions.insert(
                format!("q{i}"),
                serde_json::json!({ "type": "noul", "instructions": "?" }),
            );
        }
        let r = req(serde_json::json!({ "state": "s", "questions": questions }));
        assert_eq!(
            plan(&r, &l).expect_err("must refuse").code,
            Code::TooManyQuestions
        );

        let long = "x".repeat(l.max_field_chars + 1);
        let r = req(serde_json::json!({
            "state": long,
            "questions": { "q": { "type": "noul", "instructions": "?" } }
        }));
        assert_eq!(
            plan(&r, &l).expect_err("must refuse").code,
            Code::StateTooLong
        );
    }

    #[test]
    fn an_empty_state_or_no_questions_is_refused() {
        let r = req(serde_json::json!({
            "state": "  ",
            "questions": { "q": { "type": "noul", "instructions": "?" } }
        }));
        assert_eq!(
            plan(&r, &limits()).expect_err("must refuse").code,
            Code::Unprocessable
        );
        let r = req(serde_json::json!({ "state": "s", "questions": {} }));
        assert_eq!(
            plan(&r, &limits()).expect_err("must refuse").code,
            Code::Unprocessable
        );
    }

    #[test]
    fn a_question_with_nothing_to_ask_is_refused() {
        let r = req(serde_json::json!({
            "state": "s",
            "questions": { "q": { "type": "noul" } }
        }));
        assert_eq!(
            plan(&r, &limits()).expect_err("must refuse").code,
            Code::InvalidQuestion
        );
    }

    #[test]
    fn answer_keys_are_exactly_the_question_keys() {
        let r = owners_example();
        let (_, p) = plan(&r, &limits()).expect("plan");
        // Sorted key order: team's three hypotheses, then urgent's two.
        let a = answers(&p, &[0.99, 0.01, 0.01, 0.98, 0.02]);
        let asked: Vec<&String> = r.questions.keys().collect();
        let answered: Vec<&String> = a.keys().collect();
        assert_eq!(asked, answered);
    }

    #[test]
    fn choice_probabilities_are_non_negative_and_sum_to_one() {
        let (_, p) = plan(&owners_example(), &limits()).expect("plan");
        for scores in [
            vec![0.9, 0.05, 0.05, 0.98, 0.02],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![f32::NAN, 2.0, -1.0, 0.5, 0.5],
        ] {
            let a = answers(&p, &scores);
            let Answer::Choice {
                probabilities,
                confidence,
                choice,
            } = &a["team"]
            else {
                panic!("not a choice")
            };
            let sum: f32 = probabilities.values().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum {sum} for {scores:?}");
            assert!(probabilities.values().all(|v| *v >= 0.0));
            // Confidence is not a second opinion: it is the winner's own probability.
            assert_eq!(*confidence, probabilities[choice]);
            assert_eq!(
                *confidence,
                probabilities.values().copied().fold(f32::MIN, f32::max)
            );
        }
    }

    #[test]
    fn a_noul_is_in_zero_one_whatever_the_model_returns() {
        let (_, p) = plan(&owners_example(), &limits()).expect("plan");
        for scores in [
            vec![0.0, 0.0, 0.0, 0.98, 0.02],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, f32::NAN, -3.0],
            vec![0.0, 0.0, 0.0, 7.0, 7.0],
        ] {
            let Answer::Noul { noul } = answers(&p, &scores)["urgent"] else {
                panic!("not a noul")
            };
            assert!((0.0..=1.0).contains(&noul), "noul {noul} for {scores:?}");
        }
    }

    #[test]
    fn a_two_sided_noul_contests_the_cases_and_a_one_sided_one_does_not() {
        // Both criteria entail strongly: the contest says "undecided", the raw score
        // would have said "0.9, confident". That difference is the whole reason the two
        // paths exist.
        assert!((binary_probability(&[0.9, 0.9]) - 0.5).abs() < 1e-6);
        assert!((binary_probability(&[0.9]) - 0.9).abs() < 1e-6);
        assert!((binary_probability(&[0.8, 0.2]) - 0.8).abs() < 1e-6);
        // Nothing entailed at all is undecided, not zero.
        assert_eq!(binary_probability(&[0.0, 0.0]), 0.5);
    }

    #[test]
    fn a_boolean_answers_with_probability_the_way_the_gateway_client_reads_it() {
        let r = req(serde_json::json!({
            "state": "s",
            "questions": { "risky": { "type": "boolean", "instructions": "Would this change production?" } }
        }));
        let (_, p) = plan(&r, &limits()).expect("plan");
        let a = answers(&p, &[0.42]);
        let v = serde_json::to_value(&a["risky"]).expect("json");
        assert_eq!(v["type"], "boolean");
        assert!((v["probability"].as_f64().expect("f64") - 0.42).abs() < 1e-6);
    }

    #[test]
    fn question_order_cannot_change_an_answer() {
        let a = owners_example();
        let b: SystemOneRequest = serde_json::from_str(
            &serde_json::to_string(&a)
                .expect("ser")
                // Same two questions, written the other way round.
                .replace("\"urgent\"", "\"zzz_urgent\""),
        )
        .expect("parse");
        let (_, pa) = plan(&a, &limits()).expect("plan");
        let (_, pb) = plan(&b, &limits()).expect("plan");
        // `team` is planned identically whether it sorts first or second.
        let ta = pa.items.iter().find(|i| i.key == "team").expect("a");
        let tb = pb.items.iter().find(|i| i.key == "team").expect("b");
        assert_eq!(ta.hypotheses, tb.hypotheses);
        assert_eq!(ta.options, tb.options);
    }

    #[test]
    fn the_answer_serialises_to_the_documented_shape() {
        let (_, p) = plan(&owners_example(), &limits()).expect("plan");
        let v = serde_json::to_value(SystemOneResponse {
            model: "openjev".into(),
            answers: answers(&p, &[0.99, 0.001, 0.001, 0.98, 0.02]),
            usage: SystemOneUsage {
                input_tokens: 376,
                output_tokens: 0,
                cost: 0.0,
            },
            id: generation_id(),
            provider: PROVIDER.into(),
        })
        .expect("json");
        assert_eq!(v["answers"]["urgent"]["type"], "noul");
        assert!(v["answers"]["urgent"]["noul"].is_number());
        assert_eq!(v["answers"]["team"]["type"], "choice");
        assert_eq!(v["answers"]["team"]["choice"], "billing");
        assert!(v["answers"]["team"]["probabilities"]["sales"].is_number());
        assert!(v["answers"]["team"]["confidence"].is_number());
        assert_eq!(v["usage"]["output_tokens"], 0);
        assert_eq!(v["usage"]["cost"], 0.0);
        assert_eq!(v["provider"], "openjev");
    }

    #[test]
    fn a_state_that_is_an_object_becomes_the_premise_verbatim() {
        // fast-jev-compaction sends `{context, goal, history}`. Refusing it, or rendering
        // it in some prettier way of our own, is the difference between that client
        // working against this server and not.
        let r = req(serde_json::json!({
            "state": { "goal": "ship", "history": [{ "i": 0, "role": "user" }] },
            "questions": { "q": { "type": "noul", "instructions": "Is it done?" } }
        }));
        let (premise, _) = plan(&r, &limits()).expect("plan");
        assert_eq!(
            premise,
            r#"{"goal":"ship","history":[{"i":0,"role":"user"}]}"#
        );
    }

    #[test]
    fn structured_instructions_reach_the_hypothesis_as_json() {
        let r = req(serde_json::json!({
            "state": "s",
            "questions": { "dupe": {
                "type": "noul",
                "instructions": { "other": { "name": "John" }, "question": "Same person as `other`?" }
            }}
        }));
        let (_, p) = plan(&r, &limits()).expect("plan");
        assert!(
            p.items[0].hypotheses[0].contains("Same person as"),
            "{:?}",
            p.items[0].hypotheses
        );
        assert!(p.items[0].hypotheses[0].starts_with('{'));
    }

    #[test]
    fn a_null_criterion_means_no_description_not_the_word_null() {
        let r = req(serde_json::json!({
            "state": "s",
            "questions": { "q": {
                "type": "noul",
                "instructions": "Urgent?",
                "criteria": { "true": "time-sensitive", "false": null }
            }}
        }));
        let (_, p) = plan(&r, &limits()).expect("plan");
        // A described `true` against an undescribed `false` is a rigged contest, so it
        // stays one-sided rather than scoring "Urgent?" against "Urgent? null".
        assert_eq!(p.items[0].options, ["true"]);
        assert!(!p.items[0].hypotheses[0].contains("null"));
    }

    #[test]
    fn a_score_answer_carries_the_legend_the_spec_requires() {
        let r = req(serde_json::json!({
            "state": "s",
            "questions": { "tone": { "type": "score", "instructions": "How angry?",
                                     "criteria": ["Calm", "Frustrated", "Very angry"] }}
        }));
        let (_, p) = plan(&r, &limits()).expect("plan");
        let a = answers(&p, &[0.0, 0.95, 0.05]);
        let Answer::Score {
            score,
            legend,
            probabilities,
            confidence,
        } = &a["tone"]
        else {
            panic!("not a score")
        };
        assert_eq!(legend["0"], "Calm");
        assert_eq!(legend["2"], "Very angry");
        // The probability-weighted mean of the level indices, in [0, levels-1].
        assert!((0.0..=2.0).contains(score));
        assert!((score - 1.05).abs() < 1e-5, "score {score}");
        assert!((probabilities.values().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((*confidence - 0.95).abs() < 1e-5);
    }

    #[test]
    fn the_generation_id_has_the_shape_clients_log_and_dedupe_on() {
        let id = generation_id();
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 4, "{id}");
        assert_eq!(&id[..8], "gen-dec-");
        assert!(parts[2].parse::<u64>().is_ok(), "{id}");
        assert_eq!(parts[3].len(), 20, "{id}");
        assert_ne!(id, generation_id(), "ids must not repeat within a second");
    }
}
