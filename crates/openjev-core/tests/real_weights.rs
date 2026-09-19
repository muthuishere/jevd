//! Integration tests that need real weights and a real forward pass.
//!
//! Gated behind `--features real-weights,backend-llamacpp` and an explicit GGUF path, so
//! `cargo test` on a laptop with no model stays fast and green:
//!
//! ```text
//! OPENJEV_TEST_GGUF=/path/to/Qwen3.5-0.8B-Q8_0.gguf \
//!   cargo test -p openjev-core --features real-weights,backend-llamacpp -- --nocapture
//! ```
//!
//! The load-bearing test is [`recurrent_state_does_not_leak_between_sequences`]. Design
//! risk R2: the 24 linear-attention layers of a Qwen3.5 hybrid carry a gated-DeltaNet
//! recurrent state, and a leak across sequences is a **silently wrong label**, not a
//! crash. Nothing else in this crate can detect that, so it is asserted here against the
//! real runtime.

#![cfg(all(feature = "real-weights", feature = "backend-llamacpp"))]

use openjev_core::backend::{Backend, EncodedInput, OpenRequest};
use openjev_core::backends::llamacpp;
use openjev_core::device::{Device, Dtype};
use openjev_core::registry::Registry;
use std::path::PathBuf;

const TARGET: &str = "Premise: A man is playing a guitar on stage. \
                      Hypothesis: Someone is performing music.";
const DISTRACTOR: &str = "Premise: The quantum chromodynamics of confinement involve gluon \
    flux tubes stretching between colour charges, and lattice simulations at finite \
    temperature show a deconfinement transition near 155 MeV. Hypothesis: This is a long \
    distractor whose only job is to leave a large recurrent state behind in every one of \
    the linear-attention layers of this model.";

fn gguf() -> Option<PathBuf> {
    std::env::var_os("OPENJEV_TEST_GGUF")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
}

/// A registry entry describing whatever GGUF the operator pointed us at. Hidden size and
/// label count come from env so the test is not welded to one checkpoint.
fn open_backend(hidden: usize) -> Option<Box<dyn Backend>> {
    let path = gguf()?;
    let mut reg = Registry::default();
    reg.merge_str(
        &format!(
            r#"
[models.test]
repo = "local/test"
arch = "qwen3_5"
template = "Premise: {{premise}} Hypothesis: {{hypothesis}}"
labels = ["contradiction", "entailment", "neutral"]
entailment_label = "entailment"
context = 8192
hidden_size = {hidden}
tokenizer = {{ repo = "local/test", file = "tokenizer.json", pad_token_id = 248044 }}
head = {{ kind = "linear", in_features = {hidden}, out_features = 3, tensor = "score.weight", repo = "r", file = "f" }}
[models.test.backends.llamacpp]
requires = ["qwen3_5-hybrid"]
weights = {{ repo = "r", file = "w.gguf" }}
"#
        ),
        std::path::Path::new("<test>"),
    )
    .expect("test registry entry must be valid");
    let spec = reg.get("test").expect("spec").clone();

    let device = if llamacpp::factory().supports_device(Device::Metal) {
        Device::Metal
    } else {
        Device::Cpu
    };

    let req = OpenRequest {
        spec,
        weights: path,
        device,
        dtype: Dtype::Quant("q8_0"),
        context: 8192,
        n_threads: None,
        max_seqs: env_max_seqs(),
    };
    Some(llamacpp::factory().open(&req).expect("backend must open"))
}

/// Tokenise without a tokenizer.json: the GGUF under test is a *different* checkpoint
/// from ours, so we feed synthetic token ids. The recurrent-state question is about ids,
/// not text, and this keeps the test independent of any tokenizer file.
/// Sequences per decode. Defaults to 1 (the shipping default); set `OPENJEV_MAX_SEQS`
/// above 1 to run the whole suite against the batching path, which is how batching is
/// shown to be correct rather than merely fast.
fn env_max_seqs() -> usize {
    std::env::var("OPENJEV_MAX_SEQS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

fn ids(seed: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| 1000 + (seed * 97 + i as u32 * 31) % 40000)
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    dot / (na * nb)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum::<f64>()
        .sqrt();
    let n: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    d / n
}

fn hidden_size() -> usize {
    std::env::var("OPENJEV_TEST_HIDDEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
}

/// **R2.** The same sequence must produce the same hidden state whether it is forwarded
/// alone or after a long, unrelated sequence in the same backend. A gated-DeltaNet state
/// that survives across sequences would shift the second result, and the shift would grow
/// with the length of what preceded it.
#[test]
fn recurrent_state_does_not_leak_between_sequences() {
    let Some(backend) = open_backend(hidden_size()) else {
        eprintln!("skipped: set OPENJEV_TEST_GGUF to a Qwen3.5 GGUF to run this");
        return;
    };

    let target = EncodedInput::unpadded(ids(1, 22)).expect("target");
    let alone = backend
        .forward(std::slice::from_ref(&target))
        .expect("alone");

    // Increasing distractor lengths. If state leaked, the deviation would grow with the
    // state left behind; a flat, tiny deviation is float reduction order, not contamination.
    for n in [8usize, 64, 256, 1024] {
        let distractor = EncodedInput::unpadded(ids(7, n)).expect("distractor");
        let _ = backend
            .forward(std::slice::from_ref(&distractor))
            .expect("distractor fwd");
        let after = backend
            .forward(std::slice::from_ref(&target))
            .expect("after");

        let r = rel_l2(&alone[0].0, &after[0].0);
        let c = 1.0 - cos(&alone[0].0, &after[0].0);
        eprintln!("distractor {n:>5} tokens -> relL2 {r:.3e}  1-cos {c:.3e}");
        assert!(
            r < 1e-2,
            "hidden state moved by relL2 {r:.3e} after a {n}-token distractor — that is a \
             leaked recurrent state, and it is a silently wrong label, not a crash"
        );
    }
}

/// The same input twice in a row must agree. Determinism is the floor every other
/// assertion in this file stands on.
#[test]
fn the_same_input_twice_agrees() {
    let Some(backend) = open_backend(hidden_size()) else {
        return;
    };
    let x = EncodedInput::unpadded(ids(3, 30)).expect("x");
    let a = backend.forward(std::slice::from_ref(&x)).expect("a");
    let b = backend.forward(std::slice::from_ref(&x)).expect("b");
    let r = rel_l2(&a[0].0, &b[0].0);
    eprintln!("determinism floor: relL2 {r:.3e}");
    assert!(r < 1e-3, "same input, different answer: relL2 {r:.3e}");
}

/// `LlamaPoolingType::Last` is a name; this asserts what it actually read.
///
/// Two sequences with an identical prefix and a different final token. Under CLS/first
/// pooling these would be **identical** — the first token's state cannot see the last.
/// Under last-token pooling they differ. Organised against silently pooling the wrong
/// position, which is the same class of bug as pooling a pad token.
#[test]
fn pooling_reads_the_last_token_not_the_first() {
    let Some(backend) = open_backend(hidden_size()) else {
        return;
    };
    let mut a = ids(5, 24);
    let mut b = a.clone();
    let last = a.len() - 1;
    a[last] = 2000;
    b[last] = 31000;

    let ha = backend
        .forward(&[EncodedInput::unpadded(a).expect("a")])
        .expect("fa");
    let hb = backend
        .forward(&[EncodedInput::unpadded(b).expect("b")])
        .expect("fb");
    let d = rel_l2(&ha[0].0, &hb[0].0);
    eprintln!("last-token sensitivity: relL2 {d:.3e}");
    assert!(
        d > 1e-2,
        "changing only the final token moved the pooled state by relL2 {d:.3e} — pooling is \
         not reading the last position"
    );
}

/// Sanity check on the hidden width, so a config/weights mismatch is caught here rather
/// than as a confident wrong label from a head reading the wrong number of floats.
#[test]
fn hidden_width_matches_the_registry() {
    let Some(backend) = open_backend(hidden_size()) else {
        return;
    };
    let x = EncodedInput::unpadded(ids(9, 12)).expect("x");
    let h = backend.forward(std::slice::from_ref(&x)).expect("fwd");
    assert_eq!(h[0].dim(), hidden_size());
    assert_eq!(backend.describe().hidden_size, hidden_size());
    assert!(h[0].0.iter().any(|v| *v != 0.0), "all-zero hidden state");
    assert!(
        h[0].0.iter().all(|v| v.is_finite()),
        "non-finite hidden state"
    );
}

/// Throughput, reported not asserted. The brief's estimate is CUDA 60-150 pairs/s,
/// Metal 8-18, CPU 1-3 for a 4B; a far lower number means something is wrong and should
/// be said out loud rather than accepted.
#[test]
fn report_throughput() {
    let Some(backend) = open_backend(hidden_size()) else {
        return;
    };
    let inputs: Vec<EncodedInput> = (0..8)
        .map(|i| EncodedInput::unpadded(ids(i, 64)).expect("input"))
        .collect();
    // Warm the graph so the number is steady-state, not first-call allocation.
    let _ = backend.forward(&inputs[..1]).expect("warmup");
    let t = std::time::Instant::now();
    let _ = backend.forward(&inputs).expect("bench");
    let secs = t.elapsed().as_secs_f64();
    let info = backend.describe();
    eprintln!(
        "throughput: {:.1} pairs/s at 64 tokens on {} ({} hidden)",
        inputs.len() as f64 / secs,
        info.device,
        info.hidden_size
    );
    let _ = (TARGET, DISTRACTOR);
}
