//! **The contract.** Our Rust `predict` against probabilities produced by the reference
//! implementation (`transformers` + `Qwen3_5ForSequenceClassification`, driven exactly as
//! `modeling_openjev.py` drives it).
//!
//! `tests/golden/nli.json` is ground truth. It is not a snapshot of our own output — that
//! would only pin our bugs in place — it is a recording of the model this project claims
//! to serve. Everything here is graded against it.
//!
//! The fixture also carries the reference's own spread (bf16 vs fp32 on identical
//! weights, and batched vs unbatched). That spread is the **tolerance floor**: a Rust
//! result that sits closer to the bf16 reference than fp32 does is as correct as the
//! reference is, and demanding better than that would be demanding better than the model.
//!
//! ```text
//! OPENJEV_TEST_GGUF=.../openjev-4b-nli-v2-Q8_0.gguf \
//! OPENJEV_TEST_TOKENIZER=.../tokenizer.json \
//! OPENJEV_TEST_HEAD=.../score.safetensors \
//!   cargo test -p openjev-core --features real-weights,backend-llamacpp \
//!   --test golden -- --nocapture
//! ```
//!
//! Every test silently passes when the env is unset, so a laptop with no weights still
//! gets a green `task check`. That is deliberate and it is also the risk: absence of the
//! fixture is absence of evidence, so CI that means to enforce this must set the env.

#![cfg(all(feature = "real-weights", feature = "backend-llamacpp"))]

use openjev_core::backend::OpenRequest;
use openjev_core::backends::llamacpp;
use openjev_core::device::{Device, Dtype};
use openjev_core::head::Head;
use openjev_core::ops::Session;
use openjev_core::registry::Registry;
use openjev_core::tokenize::Encoder;
use std::path::PathBuf;

const LABELS: [&str; 3] = ["contradiction", "entailment", "neutral"];

struct GoldenPair {
    index: usize,
    premise: String,
    hypothesis: String,
    token_ids: Vec<u32>,
    probs_bf16: Vec<f32>,
    probs_fp32: Vec<f32>,
    hidden_bf16: Option<Vec<f32>>,
}

struct Golden {
    pairs: Vec<GoldenPair>,
    /// max |p_bf16 - p_fp32| over the fixture: the reference disagreeing with itself.
    ref_spread: f32,
}

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var_os(k)
        .map(PathBuf::from)
        .filter(|p| p.is_file())
}

fn fixture_path() -> PathBuf {
    std::env::var_os("OPENJEV_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/golden/nli.json")
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from("tests/golden/nli.json"))
        })
}

fn load_golden() -> Option<Golden> {
    let raw = std::fs::read_to_string(fixture_path()).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).expect("golden fixture must be JSON");
    let nums = |x: Option<&serde_json::Value>| -> Option<Vec<f32>> {
        Some(
            x?.as_array()?
                .iter()
                .map(|n| n.as_f64().expect("number") as f32)
                .collect(),
        )
    };
    let pairs = v["pairs"]
        .as_array()
        .expect("pairs")
        .iter()
        .map(|p| GoldenPair {
            index: p["index"].as_u64().expect("index") as usize,
            premise: p["premise"].as_str().expect("premise").to_string(),
            hypothesis: p["hypothesis"].as_str().expect("hypothesis").to_string(),
            token_ids: p["token_ids"]
                .as_array()
                .expect("token_ids")
                .iter()
                .map(|n| n.as_u64().expect("token id") as u32)
                .collect(),
            probs_bf16: nums(p.get("probs_bf16")).expect("probs_bf16"),
            probs_fp32: nums(p.get("probs_fp32")).expect("probs_fp32"),
            hidden_bf16: nums(p.get("hidden_bf16")),
        })
        .collect();
    Some(Golden {
        pairs,
        ref_spread: v["reference_spread"]["bf16_vs_fp32_max_abs_prob"]
            .as_f64()
            .expect("reference spread") as f32,
    })
}

/// The production model entry, with the three local artefact paths substituted in. The
/// template, labels, pad token and padding side are the real ones — if they drift from
/// `src/models.toml` this test stops testing what ships.
fn session() -> Option<(Session, Golden)> {
    let gguf = env_path("OPENJEV_TEST_GGUF")?;
    let tokenizer = env_path("OPENJEV_TEST_TOKENIZER")?;
    let head_file = env_path("OPENJEV_TEST_HEAD")?;
    let golden = load_golden()?;

    let reg = Registry::builtin().expect("the built-in registry must parse");
    let spec = reg
        .get("openjev-4b-nli-v2")
        .expect("the built-in registry must still carry openjev-4b-nli-v2")
        .clone();

    let encoder = Encoder::from_file(&spec, &tokenizer).expect("tokenizer must load");
    let head = Head::load(&spec.head, &head_file).expect("head must load");

    let device = match std::env::var("OPENJEV_TEST_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ if llamacpp::factory().supports_device(Device::Metal) => Device::Metal,
        _ => Device::Cpu,
    };
    let dtype_label: &'static str = Box::leak(
        std::env::var("OPENJEV_TEST_DTYPE")
            .unwrap_or_else(|_| "q8_0".into())
            .into_boxed_str(),
    );

    let req = OpenRequest {
        spec: spec.clone(),
        weights: gguf,
        device,
        dtype: Dtype::Quant(dtype_label),
        context: 8192,
        n_threads: None,
    };
    let backend = llamacpp::factory().open(&req).expect("backend must open");
    Some((
        Session::new(spec, encoder, head, backend).expect("session"),
        golden,
    ))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum::<f64>()
        .sqrt();
    let n: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    d / n
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, x)| {
            if *x > bv { (i, *x) } else { (bi, bv) }
        })
        .0
}

/// **R6 — template and tokenizer drift.** Before any probability is compared, assert the
/// exact token ids. A wrong newline, a stray BOS or an untrimmed field moves every number
/// downstream without failing anything, and debugging that from probabilities alone is
/// how a day disappears.
#[test]
fn tokenisation_matches_the_reference_exactly() {
    let Some((s, g)) = session() else { return };
    let mut bad = Vec::new();
    for p in &g.pairs {
        let ours = s
            .encoder
            .encode_text(&s.encoder.render(&p.premise, &p.hypothesis))
            .expect("encode");
        if ours != p.token_ids {
            bad.push((p.index, ours.len(), p.token_ids.len()));
        }
    }
    assert!(
        bad.is_empty(),
        "token ids differ from the reference on {} of {} pairs (index, ours, theirs): {:?}",
        bad.len(),
        g.pairs.len(),
        &bad[..bad.len().min(8)]
    );
}

/// **The ship gate.** Probabilities and labels against the reference.
///
/// Tolerance is stated, not discovered: labels must agree everywhere, and the probability
/// deviation is allowed to be a small multiple of the reference's own bf16-vs-fp32 spread.
/// Quantisation error is real and is reported per level rather than assumed harmless.
#[test]
fn probabilities_match_the_reference() {
    let Some((s, g)) = session() else { return };

    let pairs: Vec<(&str, &str)> = g
        .pairs
        .iter()
        .map(|p| (p.premise.as_str(), p.hypothesis.as_str()))
        .collect();
    let got = s.predict(&pairs).expect("predict");

    let mut worst = 0.0f32;
    let mut worst_idx = 0usize;
    // Also measured against the fp32 reference. If we sit closer to fp32 than bf16 does,
    // the deviation from bf16 is the reference's own rounding, not our error — and that
    // is a materially different story from "we are 0.03 off".
    let mut worst_fp32 = 0.0f32;
    let mut disagreements = Vec::new();
    for (p, out) in g.pairs.iter().zip(&got) {
        let d = max_abs(&out.probs, &p.probs_bf16);
        worst_fp32 = worst_fp32.max(max_abs(&out.probs, &p.probs_fp32));
        if d > worst {
            worst = d;
            worst_idx = p.index;
        }
        if argmax(&out.probs) != argmax(&p.probs_bf16) {
            disagreements.push((
                p.index,
                LABELS[argmax(&p.probs_bf16)],
                LABELS[argmax(&out.probs)],
                p.probs_bf16.clone(),
                out.probs.clone(),
            ));
        }
    }

    let agreement = 1.0 - disagreements.len() as f64 / g.pairs.len() as f64;
    eprintln!(
        "golden: {} pairs  label agreement {:.1}%  max |dp| vs bf16 {:.4} (pair {})  \
         vs fp32 {:.4}  reference's own bf16-vs-fp32 spread {:.4}",
        g.pairs.len(),
        agreement * 100.0,
        worst,
        worst_idx,
        worst_fp32,
        g.ref_spread
    );
    for d in &disagreements {
        eprintln!(
            "  pair {:>2}: reference {} {:?} -> ours {} {:?}",
            d.0, d.1, d.3, d.2, d.4
        );
    }

    // A tolerance has to be a number someone chose. This one: eight times the
    // reference's own bf16-vs-fp32 disagreement, floored at 0.02 so a suspiciously tight
    // reference spread cannot make the gate unfalsifiable. Deviation above that is not
    // "quantisation noise", it is a different model.
    let tol = (g.ref_spread * 8.0).max(0.02);
    assert!(
        disagreements.is_empty(),
        "{} of {} labels disagree with the reference",
        disagreements.len(),
        g.pairs.len()
    );
    assert!(
        worst <= tol,
        "max probability deviation {worst:.4} exceeds tolerance {tol:.4}"
    );
}

/// Localises a mismatch to trunk-vs-head. If the hidden states agree and the
/// probabilities do not, the head or the softmax is wrong; if the hidden states already
/// disagree, nothing downstream is worth looking at.
#[test]
fn hidden_states_match_the_reference() {
    let Some((s, g)) = session() else { return };
    for p in g.pairs.iter().filter(|p| p.hidden_bf16.is_some()) {
        let want = p.hidden_bf16.as_ref().expect("filtered");
        let got = s
            .latents(&[&s.encoder.render(&p.premise, &p.hypothesis)])
            .expect("latents");
        let r = rel_l2(&got[0], want);
        eprintln!("hidden pair {:>2}: relL2 {:.4e}", p.index, r);
        assert!(
            r < 0.25,
            "pair {} hidden state relL2 {r:.4e} — the trunk, not the head, is wrong",
            p.index
        );
    }
}

/// **R7 — the padding/pooling bug, on real weights.**
///
/// A batch of different-length pairs must give every pair the answer it gets alone. This
/// is the failure the weightless tests structurally cannot catch: right-padding plus
/// naive last-token pooling reads a pad embedding and returns plausible garbage, and on a
/// recurrent trunk a leaked state does the same. Mixed lengths are the whole point — a
/// batch of equal-length inputs proves nothing.
#[test]
fn a_mixed_length_batch_agrees_with_the_same_pairs_alone() {
    let Some((s, g)) = session() else { return };

    // Deliberately jagged: the longest pair in the fixture next to the shortest.
    let mut by_len: Vec<&GoldenPair> = g.pairs.iter().collect();
    by_len.sort_by_key(|p| p.token_ids.len());
    let picks: Vec<&GoldenPair> = [
        by_len[0],
        by_len[by_len.len() - 1],
        by_len[1],
        by_len[by_len.len() / 2],
        by_len[by_len.len() - 2],
    ]
    .into_iter()
    .collect();

    let pairs: Vec<(&str, &str)> = picks
        .iter()
        .map(|p| (p.premise.as_str(), p.hypothesis.as_str()))
        .collect();
    let batched = s.predict(&pairs).expect("batched");

    let mut worst = 0.0f32;
    for (p, b) in picks.iter().zip(&batched) {
        let alone = s
            .predict(&[(p.premise.as_str(), p.hypothesis.as_str())])
            .expect("alone");
        let d = max_abs(&b.probs, &alone[0].probs);
        worst = worst.max(d);
        assert_eq!(
            b.label,
            alone[0].label,
            "pair {} ({} tokens) changes label between batched and alone — padding or \
             pooling is wrong, or recurrent state is leaking",
            p.index,
            p.token_ids.len()
        );
    }
    eprintln!("mixed-length batch vs alone: max |dp| {worst:.4e}");
    assert!(
        worst < 1e-3,
        "batched and unbatched probabilities differ by {worst:.4e}"
    );
}

/// **ADR 0003, on real weights.** An all-zero hidden state is a uniform softmax whose
/// argmax is always label 0 — a silent, deterministic wrong answer. The guard exists; this
/// asserts it still holds when the weights are real rather than synthetic.
#[test]
fn no_pair_produces_an_all_zero_hidden_state() {
    let Some((s, g)) = session() else { return };
    for p in &g.pairs {
        let h = s
            .latents(&[&s.encoder.render(&p.premise, &p.hypothesis)])
            .expect("latents");
        let norm: f64 = h[0]
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            norm > 1e-6,
            "pair {} produced an all-zero hidden state (norm {norm:e})",
            p.index
        );
    }
}

/// `rerank` and `grade` end to end on real weights, not just `predict`.
///
/// Asserted on meaning, not on a recorded number: a correct multiple-choice rerank puts
/// the true answer first, and a correct grade scores a faithful answer above a wrong one.
/// Those are the claims the plugin actually relies on.
#[test]
fn rerank_and_grade_behave_on_real_weights() {
    let Some((s, _)) = session() else { return };

    let cases: [(&str, [&str; 3], usize); 3] = [
        (
            "Which gas do plants absorb during photosynthesis?",
            ["oxygen", "carbon dioxide", "nitrogen"],
            1,
        ),
        (
            "What is the capital of France?",
            ["Lyon", "Berlin", "Paris"],
            2,
        ),
        (
            "Which planet is closest to the Sun?",
            ["Mercury", "Venus", "Mars"],
            0,
        ),
    ];
    for (q, opts, want) in cases {
        let r = s.rerank(q, &opts).expect("rerank");
        eprintln!(
            "rerank {q:?} -> {:?} (scores {:?})",
            opts[r[0].index],
            r.iter().map(|x| x.score).collect::<Vec<_>>()
        );
        assert_eq!(
            r[0].index, want,
            "rerank put {:?} first for {q:?}, expected {:?}",
            opts[r[0].index], opts[want]
        );
    }

    let good = s
        .grade("The Eiffel Tower is in Paris.", "The tower is in Paris.")
        .expect("grade");
    let bad = s
        .grade("The Eiffel Tower is in Paris.", "The tower is in Rome.")
        .expect("grade");
    eprintln!(
        "grade: faithful {:.3} ({}), wrong {:.3} ({})",
        good.score, good.prediction.label, bad.score, bad.prediction.label
    );
    assert!(
        good.score > bad.score,
        "a faithful answer must score above a wrong one ({} vs {})",
        good.score,
        bad.score
    );
    assert_eq!(good.prediction.label, "entailment");
    assert_eq!(bad.prediction.label, "contradiction");
}
