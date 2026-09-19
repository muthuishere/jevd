//! The classification head. `Linear(hidden -> labels)` from a safetensors file, then a
//! softmax. That is the whole thing — 30 720 values, 60 KB.
//!
//! Organised against: asking GGUF to carry the classifier. That path exists but is built
//! and tested for yes/no rerankers on dense-attention models. Doing the matmul here also
//! makes `latents` free and makes a backend swap cheap, because every backend only has to
//! answer one question.
//!
//! The head matmul is always f32 regardless of trunk dtype. At 60 KB, precision is free.

use crate::error::{JevError, Result};
use crate::registry::HeadSpec;
use safetensors::SafeTensors;
use std::path::Path;

#[derive(Debug)]
pub struct Head {
    /// Row-major `[out, in]`, exactly as `score.weight` is stored by `transformers`.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl Head {
    pub fn in_features(&self) -> usize {
        self.in_features
    }
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn load(spec: &HeadSpec, path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| JevError::io(path, e))?;
        Self::from_bytes(spec, &bytes)
    }

    pub fn from_bytes(spec: &HeadSpec, bytes: &[u8]) -> Result<Self> {
        let HeadSpec::Linear {
            in_features,
            out_features,
            bias,
            tensor,
            ..
        } = spec;

        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| JevError::Head(format!("safetensors: {e}")))?;

        let w = st.tensor(tensor).map_err(|_| {
            JevError::Head(format!(
                "no tensor '{tensor}' in the head file (has: {})",
                st.names().join(", ")
            ))
        })?;

        if w.shape() != [*out_features, *in_features] {
            return Err(JevError::Head(format!(
                "tensor '{tensor}' has shape {:?}, the registry says [{out_features}, {in_features}] \
                 — a transposed head produces confident nonsense, so this is fatal",
                w.shape()
            )));
        }

        let weight = to_f32(w.dtype(), w.data(), out_features * in_features)?;

        let bias_vec = if *bias {
            let name = tensor.replace("weight", "bias");
            let b = st.tensor(&name).map_err(|_| {
                JevError::Head(format!(
                    "registry declares bias = true but '{name}' is absent"
                ))
            })?;
            Some(to_f32(b.dtype(), b.data(), *out_features)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_vec,
            in_features: *in_features,
            out_features: *out_features,
        })
    }

    /// Logits for one pooled hidden state.
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.in_features {
            return Err(JevError::Head(format!(
                "hidden state is {}-d, head expects {}-d",
                hidden.len(),
                self.in_features
            )));
        }
        let mut out = Vec::with_capacity(self.out_features);
        for o in 0..self.out_features {
            let row = &self.weight[o * self.in_features..(o + 1) * self.in_features];
            let mut acc = 0.0f32;
            for (w, h) in row.iter().zip(hidden) {
                acc += w * h;
            }
            if let Some(b) = &self.bias {
                acc += b[o];
            }
            out.push(acc);
        }
        Ok(out)
    }

    pub fn probs(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        Ok(softmax(&self.logits(hidden)?))
    }
}

fn to_f32(dtype: safetensors::Dtype, data: &[u8], want: usize) -> Result<Vec<f32>> {
    use safetensors::Dtype as D;
    let v: Vec<f32> = match dtype {
        D::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        D::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::BF16 => data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => {
            return Err(JevError::Head(format!(
                "head dtype {other:?} is not supported (want f32, f16 or bf16)"
            )));
        }
    };
    if v.len() != want {
        return Err(JevError::Head(format!(
            "head tensor has {} values, expected {want}",
            v.len()
        )));
    }
    Ok(v)
}

/// Max-subtracted softmax. The subtraction is not an optimisation: a 3-way head on a
/// 2560-d vector can produce logits large enough to overflow `exp` in f32, and the
/// failure is `NaN` probabilities that argmax happily ranks.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        // Unreachable after the max subtraction, but a uniform distribution beats NaN.
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Index of the largest value; ties go to the lowest index, deterministically.
pub fn argmax(v: &[f32]) -> Option<usize> {
    v.iter()
        .enumerate()
        .fold(None::<(usize, f32)>, |best, (i, &x)| match best {
            Some((_, b)) if b >= x => best,
            _ => Some((i, x)),
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_preserves_order() {
        let p = softmax(&[1.0, 3.0, 2.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[1] > p[2] && p[2] > p[0]);
    }

    #[test]
    fn softmax_survives_logits_that_would_overflow_exp() {
        let p = softmax(&[200.0, 100.0, 0.0]);
        assert!(p.iter().all(|x| x.is_finite()), "{p:?}");
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > 0.99);
    }

    #[test]
    fn softmax_of_equal_logits_is_uniform() {
        let p = softmax(&[5.0, 5.0, 5.0]);
        for x in p {
            assert!((x - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn argmax_breaks_ties_at_the_lowest_index() {
        assert_eq!(argmax(&[0.5, 0.5, 0.5]), Some(0));
        assert_eq!(argmax(&[0.1, 0.9, 0.2]), Some(1));
        assert_eq!(argmax(&[]), None);
    }

    fn spec(out: usize, in_: usize, bias: bool) -> HeadSpec {
        HeadSpec::Linear {
            in_features: in_,
            out_features: out,
            bias,
            tensor: "score.weight".into(),
            repo: "r".into(),
            file: "f".into(),
            revision: None,
            sha256: String::new(),
        }
    }

    fn st_bytes(pairs: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let views: Vec<_> = pairs
            .iter()
            .map(|(n, shape, data)| {
                let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                (n.to_string(), shape.clone(), raw)
            })
            .collect();
        let tensors: Vec<(String, safetensors::tensor::TensorView<'_>)> = views
            .iter()
            .map(|(n, shape, raw)| {
                (
                    n.clone(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.clone(),
                        raw,
                    )
                    .expect("view"),
                )
            })
            .collect();
        safetensors::serialize(tensors, None).expect("serialize")
    }

    #[test]
    fn loads_a_linear_head_and_computes_logits() {
        // out=2, in=3; identity-ish rows so the answer is hand-checkable.
        let bytes = st_bytes(&[(
            "score.weight",
            vec![2, 3],
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        )]);
        let h = Head::from_bytes(&spec(2, 3, false), &bytes).unwrap();
        assert_eq!(h.logits(&[5.0, 7.0, 9.0]).unwrap(), vec![5.0, 14.0]);
        assert_eq!(argmax(&h.probs(&[5.0, 7.0, 9.0]).unwrap()), Some(1));
    }

    #[test]
    fn a_transposed_head_is_rejected_not_silently_used() {
        let bytes = st_bytes(&[("score.weight", vec![3, 2], vec![1.0; 6])]);
        let e = Head::from_bytes(&spec(2, 3, false), &bytes).unwrap_err();
        assert!(e.to_string().contains("transposed"), "{e}");
    }

    #[test]
    fn a_missing_tensor_lists_what_is_there() {
        let bytes = st_bytes(&[("classifier.weight", vec![2, 3], vec![0.0; 6])]);
        let e = Head::from_bytes(&spec(2, 3, false), &bytes).unwrap_err();
        assert!(e.to_string().contains("classifier.weight"), "{e}");
    }

    #[test]
    fn bias_is_applied_when_declared() {
        let bytes = st_bytes(&[
            ("score.weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            ("score.bias", vec![2], vec![10.0, -10.0]),
        ]);
        let h = Head::from_bytes(&spec(2, 2, true), &bytes).unwrap();
        assert_eq!(h.logits(&[1.0, 1.0]).unwrap(), vec![11.0, -9.0]);
    }

    #[test]
    fn declared_bias_that_is_absent_is_an_error() {
        let bytes = st_bytes(&[("score.weight", vec![2, 2], vec![1.0; 4])]);
        assert!(Head::from_bytes(&spec(2, 2, true), &bytes).is_err());
    }

    #[test]
    fn wrong_width_hidden_state_is_rejected() {
        let bytes = st_bytes(&[("score.weight", vec![2, 3], vec![1.0; 6])]);
        let h = Head::from_bytes(&spec(2, 3, false), &bytes).unwrap();
        assert!(h.logits(&[1.0, 2.0]).is_err());
    }
}
