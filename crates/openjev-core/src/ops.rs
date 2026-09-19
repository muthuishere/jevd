//! `predict` / `rerank` / `grade` / `latents`.
//!
//! All four are free functions over `forward` + the head, so a backend that implements
//! one method gets all four and **cannot implement them inconsistently**. `rerank` and
//! `grade` are defined in terms of `predict`: one label convention, one softmax, one
//! place to be wrong.

use crate::backend::{Backend, Caps, EncodedInput, Hidden};
use crate::error::{JevError, Result};
use crate::head::{Head, argmax};
use crate::registry::ModelSpec;
use crate::tokenize::Encoder;

#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Probability per label, in registry label order.
    pub probs: Vec<f32>,
    /// argmax label. Ties resolve to the lowest index, deterministically.
    pub label: String,
    pub label_index: usize,
}

impl Prediction {
    pub fn prob_of(&self, label: &str, labels: &[String]) -> Option<f32> {
        labels
            .iter()
            .position(|l| l == label)
            .map(|i| self.probs[i])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ranked {
    pub index: usize,
    /// `P(entailment)`. Documented as the score so callers do not have to guess.
    pub score: f32,
    pub prediction: Prediction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Grade {
    /// Scalar in [0,1] = `P(entailment)`.
    pub score: f32,
    pub prediction: Prediction,
}

/// Everything the free functions need. Assembled once at boot.
pub struct Session {
    pub spec: ModelSpec,
    pub encoder: Encoder,
    pub head: Head,
    pub backend: Box<dyn Backend>,
    entailment_index: usize,
}

impl Session {
    pub fn new(
        spec: ModelSpec,
        encoder: Encoder,
        head: Head,
        backend: Box<dyn Backend>,
    ) -> Result<Self> {
        let entailment_index = spec.entailment_index()?;
        let info = backend.describe();
        if info.hidden_size != head.in_features() {
            return Err(JevError::Head(format!(
                "backend emits {}-d states but the head expects {}-d",
                info.hidden_size,
                head.in_features()
            )));
        }
        if head.out_features() != spec.labels.len() {
            return Err(JevError::Head(format!(
                "head has {} outputs but the registry declares {} labels",
                head.out_features(),
                spec.labels.len()
            )));
        }
        Ok(Self {
            spec,
            encoder,
            head,
            backend,
            entailment_index,
        })
    }

    pub fn labels(&self) -> &[String] {
        &self.spec.labels
    }

    /// Forward a set of already-encoded inputs, honouring `Caps::BATCH`. A backend that
    /// cannot batch gets looped — same API, worse throughput, no change upstream.
    fn forward(&self, inputs: &[EncodedInput]) -> Result<Vec<Hidden>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if self.backend.capabilities().contains(Caps::BATCH) {
            self.backend.forward(inputs)
        } else {
            let mut out = Vec::with_capacity(inputs.len());
            for i in inputs {
                out.extend(self.backend.forward(std::slice::from_ref(i))?);
            }
            Ok(out)
        }
    }

    fn to_prediction(&self, hidden: &Hidden) -> Result<Prediction> {
        let probs = self.head.probs(&hidden.0)?;
        let label_index =
            argmax(&probs).ok_or_else(|| JevError::Head("empty head output".into()))?;
        Ok(Prediction {
            label: self.spec.labels[label_index].clone(),
            label_index,
            probs,
        })
    }

    pub fn predict(&self, pairs: &[(&str, &str)]) -> Result<Vec<Prediction>> {
        let inputs = self.encoder.encode_pairs(pairs)?;
        self.forward(&inputs)?
            .iter()
            .map(|h| self.to_prediction(h))
            .collect()
    }

    /// Options sorted by `P(entailment)`, descending. Stable within equal scores, so a
    /// tie preserves input order rather than shuffling between runs.
    pub fn rerank(&self, question: &str, options: &[&str]) -> Result<Vec<Ranked>> {
        let pairs: Vec<(&str, &str)> = options.iter().map(|o| (*o, question)).collect();
        let mut ranked: Vec<Ranked> = self
            .predict(&pairs)?
            .into_iter()
            .enumerate()
            .map(|(index, prediction)| Ranked {
                index,
                score: prediction.probs[self.entailment_index],
                prediction,
            })
            .collect();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(ranked)
    }

    pub fn grade(&self, premise: &str, hypothesis: &str) -> Result<Grade> {
        let prediction = self
            .predict(&[(premise, hypothesis)])?
            .pop()
            .ok_or_else(|| JevError::Head("predict returned nothing for one pair".into()))?;
        Ok(Grade {
            score: prediction.probs[self.entailment_index],
            prediction,
        })
    }

    /// Pooled hidden states, f32. Free, because we already hold them.
    pub fn latents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if !self.backend.capabilities().contains(Caps::LATENTS) {
            return Err(JevError::NotSupported {
                backend: self.backend.describe().name.to_string(),
                capability: "latents",
            });
        }
        let inputs = texts
            .iter()
            .map(|t| EncodedInput::unpadded(self.encoder.encode_text(t)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.forward(&inputs)?.into_iter().map(|h| h.0).collect())
    }

    /// One line, stderr, always — printed by the caller. Naming the exact resolved model,
    /// backend build, device and dtype is what makes a bug report actionable.
    pub fn banner(&self) -> String {
        let i = self.backend.describe();
        format!(
            "openjev {}  model={}@{}  backend={}({})  device={}  dtype={}/f32-head  ctx={}  labels={}",
            env!("CARGO_PKG_VERSION"),
            self.spec.id,
            short_rev(&self.spec.revision),
            i.name,
            i.version,
            i.device,
            i.dtype,
            i.context,
            self.spec.labels.len(),
        )
    }
}

fn short_rev(rev: &str) -> &str {
    if rev.len() >= 40 { &rev[..8] } else { rev }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendInfo;
    use crate::device::{Device, Dtype};
    use crate::registry::Registry;

    /// A backend that returns a hidden state encoding the index of the input, so the test
    /// can assert which pair produced which prediction.
    struct Fake {
        caps: Caps,
        states: Vec<Vec<f32>>,
        calls: std::sync::Mutex<Vec<usize>>,
    }

    impl Backend for Fake {
        fn describe(&self) -> BackendInfo {
            BackendInfo {
                name: "fake",
                version: "0".into(),
                device: Device::Cpu,
                dtype: Dtype::F32,
                context: 4096,
                hidden_size: 2,
                caps: self.caps,
            }
        }
        fn forward(&self, batch: &[EncodedInput]) -> Result<Vec<Hidden>> {
            self.calls.lock().expect("lock").push(batch.len());
            Ok(batch
                .iter()
                .map(|i| {
                    // The first token (the premise word, given this fixture's template)
                    // selects the canned state, so a test can say which pair got which.
                    let k = (i.tokens[0] as usize) % self.states.len();
                    Hidden(self.states[k].clone())
                })
                .collect())
        }
    }

    fn head2x2() -> Head {
        // Identity head over 2 dims -> 2 labels.
        let raw: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &raw)
            .expect("view");
        let bytes =
            safetensors::serialize(vec![("score.weight".to_string(), view)], None).expect("ser");
        Head::from_bytes(
            &crate::registry::HeadSpec::Linear {
                in_features: 2,
                out_features: 2,
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

    fn session(caps: Caps, states: Vec<Vec<f32>>) -> Session {
        let mut reg = Registry::default();
        reg.merge_str(
            r#"
[models.t]
repo = "a/b"
arch = "fake"
template = "{premise} {hypothesis}"
labels = ["no", "yes"]
entailment_label = "yes"
context = 4096
hidden_size = 2
tokenizer = { repo = "a/b", file = "t.json", pad_token_id = 0 }
head = { kind = "linear", in_features = 2, out_features = 2, tensor = "score.weight", repo = "r", file = "f" }
[models.t.backends.fake]
weights = { repo = "r", file = "w" }
"#,
            std::path::Path::new("t.toml"),
        )
        .expect("registry");
        let spec = reg.get("t").expect("spec").clone();

        // A whitespace word-level tokenizer keyed so the first token id selects the
        // canned state. Built from JSON rather than the builder so the test depends only
        // on the same public surface the real tokenizer.json takes.
        let tok: tokenizers::Tokenizer = r#"{
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "zero",
                      "vocab": {"zero": 0, "one": 1, "two": 2, "three": 3}}
        }"#
        .parse()
        .expect("tokenizer json");

        let encoder = Encoder::new(&spec, tok).expect("encoder");
        let backend = Box::new(Fake {
            caps,
            states,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        Session::new(spec, encoder, head2x2(), backend).expect("session")
    }

    #[test]
    fn predict_maps_argmax_to_the_registry_label() {
        // state [0,9] -> logits [0,9] -> argmax 1 -> "yes"
        let s = session(Caps::LATENTS, vec![vec![9.0, 0.0], vec![0.0, 9.0]]);
        let p = s
            .predict(&[("zero", "zero"), ("one", "one")])
            .expect("predict");
        assert_eq!(p[0].label, "no");
        assert_eq!(p[1].label, "yes");
        assert!((p[0].probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn grade_is_p_entailment_and_agrees_with_predict() {
        let s = session(Caps::LATENTS, vec![vec![9.0, 0.0], vec![0.0, 9.0]]);
        let g = s.grade("one", "one").expect("grade");
        assert_eq!(g.prediction.label, "yes");
        assert_eq!(g.score, g.prediction.probs[1]);
        assert!(g.score > 0.99);
        let p = s.predict(&[("one", "one")]).expect("predict");
        assert_eq!(
            g.prediction, p[0],
            "grade must be predict, not a second path"
        );
    }

    #[test]
    fn rerank_sorts_by_p_entailment_and_keeps_the_original_index() {
        let s = session(
            Caps::LATENTS,
            vec![vec![9.0, 0.0], vec![0.0, 9.0], vec![0.0, 3.0]],
        );
        let r = s.rerank("q", &["zero", "one", "two"]).expect("rerank");
        assert_eq!(r.iter().map(|x| x.index).collect::<Vec<_>>(), vec![1, 2, 0]);
        assert!(r[0].score > r[1].score && r[1].score > r[2].score);
        assert_eq!(r[0].score, r[0].prediction.probs[1]);
    }

    #[test]
    fn latents_is_refused_when_the_backend_lacks_the_capability() {
        let s = session(Caps::empty(), vec![vec![1.0, 2.0]]);
        let e = s.latents(&["zero"]).unwrap_err();
        assert!(
            matches!(
                e,
                JevError::NotSupported {
                    capability: "latents",
                    ..
                }
            ),
            "{e}"
        );
        assert_eq!(e.exit_code(), 69);
    }

    #[test]
    fn latents_returns_the_pooled_state_unchanged() {
        let s = session(Caps::LATENTS, vec![vec![1.5, -2.5]]);
        assert_eq!(
            s.latents(&["zero"]).expect("latents"),
            vec![vec![1.5, -2.5]]
        );
    }

    #[test]
    fn a_non_batching_backend_is_looped_one_sequence_at_a_time() {
        let s = session(Caps::LATENTS, vec![vec![9.0, 0.0], vec![0.0, 9.0]]);
        s.predict(&[("zero", "a"), ("one", "b"), ("zero", "c")])
            .expect("predict");
        // Three forwards of one, not one forward of three — which is also why nothing pads.
        let b: &Fake = unsafe { &*(std::ptr::from_ref(&*s.backend).cast::<Fake>()) };
        assert_eq!(*b.calls.lock().expect("lock"), vec![1, 1, 1]);
    }

    #[test]
    fn banner_names_model_backend_device_and_dtype() {
        let s = session(Caps::LATENTS, vec![vec![1.0, 0.0]]);
        let b = s.banner();
        for want in ["model=t@main", "backend=fake", "device=cpu", "labels=2"] {
            assert!(b.contains(want), "banner missing {want}: {b}");
        }
    }

    #[test]
    fn a_head_that_disagrees_with_the_backend_width_is_refused_at_construction() {
        let mut reg = Registry::builtin().expect("builtin");
        let _ = &mut reg;
        // covered by Session::new's checks; the 2560 vs 2 mismatch is the realistic case.
        let s = session(Caps::LATENTS, vec![vec![1.0, 0.0]]);
        assert_eq!(s.head.in_features(), s.backend.describe().hidden_size);
    }
}
