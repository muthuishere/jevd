//! Template rendering, tokenisation, and the padding/pooling rules.
//!
//! **This is the module where a silent wrong answer is easiest to write.** The head reads
//! the last *non-pad* token's hidden state. Right-padding plus naive last-token pooling
//! pools a pad embedding and produces plausible garbage — the classic cross-encoder bug,
//! and it never throws.
//!
//! Two rules, both enforced here rather than trusted to each backend:
//!   1. An unpadded batch pools at `len - 1`. That is v0.1's llama.cpp path: each pair is
//!      its own sequence, no padding exists.
//!   2. A padded batch is **left**-padded, so the last index is always real text and
//!      pooling is index arithmetic-free. Right padding is representable in the config
//!      but carries a per-row pool index, and the test suite asserts the two agree.

use crate::backend::EncodedInput;
use crate::error::{JevError, Result};
use crate::registry::{ModelSpec, PaddingSide};
use tokenizers::Tokenizer;

/// Render the NLI template. No trimming, no normalisation: any drift here changes the
/// label distribution, so the substitution is dumb on purpose and covered by a test.
pub fn render(template: &str, premise: &str, hypothesis: &str) -> String {
    template
        .replace("{premise}", premise)
        .replace("{hypothesis}", hypothesis)
}

pub struct Encoder {
    tokenizer: Tokenizer,
    template: String,
    pad_token_id: u32,
    padding_side: PaddingSide,
    context: usize,
}

impl Encoder {
    pub fn from_file(spec: &ModelSpec, tokenizer_json: &std::path::Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_json)
            .map_err(|e| JevError::Tokenizer(format!("{}: {e}", tokenizer_json.display())))?;
        Self::new(spec, tokenizer)
    }

    pub fn new(spec: &ModelSpec, tokenizer: Tokenizer) -> Result<Self> {
        Ok(Self {
            tokenizer,
            template: spec.template.clone(),
            pad_token_id: spec.tokenizer.pad_token_id,
            padding_side: spec.tokenizer.padding_side,
            context: spec.context,
        })
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn render(&self, premise: &str, hypothesis: &str) -> String {
        render(&self.template, premise, hypothesis)
    }

    /// Tokenise one already-rendered string. `add_special_tokens` is whatever
    /// `tokenizer.json` says — we add no BOS of our own, because an extra BOS is exactly
    /// the kind of drift that moves every number without failing anything.
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| JevError::Tokenizer(e.to_string()))?;
        let ids = enc.get_ids().to_vec();
        if ids.len() > self.context {
            return Err(JevError::ContextOverflow {
                tokens: ids.len(),
                limit: self.context,
            });
        }
        Ok(ids)
    }

    pub fn encode_pair(&self, premise: &str, hypothesis: &str) -> Result<EncodedInput> {
        EncodedInput::unpadded(self.encode_text(&self.render(premise, hypothesis))?)
    }

    pub fn encode_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<EncodedInput>> {
        pairs.iter().map(|(p, h)| self.encode_pair(p, h)).collect()
    }

    /// Pad a batch to a common length for a backend that declares `Caps::BATCH`.
    /// Not used on the llama.cpp path, which forwards one sequence at a time and so pads
    /// nothing at all.
    pub fn pad_batch(&self, inputs: &[EncodedInput]) -> Vec<EncodedInput> {
        pad_batch_with(inputs, self.pad_token_id, self.padding_side)
    }
}

/// Free function so the rule is testable without a tokenizer file.
pub fn pad_batch_with(
    inputs: &[EncodedInput],
    pad_token_id: u32,
    side: PaddingSide,
) -> Vec<EncodedInput> {
    let max = inputs.iter().map(|i| i.tokens.len()).max().unwrap_or(0);
    inputs
        .iter()
        .map(|i| {
            let n = max - i.tokens.len();
            match side {
                PaddingSide::Left => {
                    let mut tokens = vec![pad_token_id; n];
                    tokens.extend_from_slice(&i.tokens);
                    // The last index is always real text. That is the entire point.
                    EncodedInput {
                        pool_index: max - 1,
                        tokens,
                    }
                }
                PaddingSide::Right => {
                    let mut tokens = i.tokens.clone();
                    tokens.extend(std::iter::repeat_n(pad_token_id, n));
                    // Per-row gather. Correct, but only because the index is carried
                    // explicitly — `tokens.len() - 1` here is the bug this type prevents.
                    EncodedInput {
                        pool_index: i.tokens.len().saturating_sub(1),
                        tokens,
                    }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> Vec<EncodedInput> {
        vec![
            EncodedInput::unpadded(vec![10, 11, 12, 13]).unwrap(),
            EncodedInput::unpadded(vec![20]).unwrap(),
            EncodedInput::unpadded(vec![30, 31]).unwrap(),
        ]
    }

    const PAD: u32 = 248044;

    #[test]
    fn template_substitutes_both_slots_and_keeps_the_newline() {
        let t = "Premise: {premise}\nHypothesis: {hypothesis}";
        assert_eq!(
            render(t, "A dog runs.", "An animal moves."),
            "Premise: A dog runs.\nHypothesis: An animal moves."
        );
    }

    #[test]
    fn template_does_not_trim_or_normalise() {
        // Whitespace is the caller's problem. We must not silently change the input.
        assert_eq!(
            render("{premise}|{hypothesis}", "  a  ", "\tb\n"),
            "  a  |\tb\n"
        );
    }

    #[test]
    fn template_braces_in_user_text_are_not_re_expanded() {
        // premise is substituted first; a "{hypothesis}" literal inside the premise would
        // be clobbered by the second pass. Assert the failure so it is a known shape.
        let out = render("{premise}/{hypothesis}", "{hypothesis}", "H");
        assert_eq!(
            out, "H/H",
            "known limitation: two-pass replace is not hygienic"
        );
    }

    #[test]
    fn unpadded_pools_the_last_token() {
        let i = EncodedInput::unpadded(vec![1, 2, 3]).unwrap();
        assert_eq!(i.pool_index, 2);
    }

    #[test]
    fn empty_sequence_is_rejected_not_pooled() {
        assert!(EncodedInput::unpadded(vec![]).is_err());
    }

    #[test]
    fn left_padding_puts_real_text_at_the_last_index() {
        let padded = pad_batch_with(&inputs(), PAD, PaddingSide::Left);
        for (p, orig) in padded.iter().zip(inputs()) {
            assert_eq!(p.tokens.len(), 4);
            assert_eq!(p.pool_index, 3);
            // The pooled token is the real final token, never the pad.
            assert_eq!(p.tokens[p.pool_index], *orig.tokens.last().unwrap());
            assert_ne!(p.tokens[p.pool_index], PAD);
        }
        assert_eq!(padded[1].tokens, vec![PAD, PAD, PAD, 20]);
    }

    #[test]
    fn right_padding_carries_a_per_row_index_and_never_pools_a_pad() {
        let padded = pad_batch_with(&inputs(), PAD, PaddingSide::Right);
        assert_eq!(padded[1].tokens, vec![20, PAD, PAD, PAD]);
        for (p, orig) in padded.iter().zip(inputs()) {
            assert_ne!(
                p.tokens[p.pool_index], PAD,
                "right-padded pooling must not read a pad token"
            );
            assert_eq!(p.tokens[p.pool_index], *orig.tokens.last().unwrap());
        }
    }

    #[test]
    fn the_naive_right_padded_bug_is_what_we_are_preventing() {
        // Documents the failure this module exists to stop: `tokens.len() - 1` on a
        // right-padded row reads the pad token and returns a confident wrong label.
        let padded = pad_batch_with(&inputs(), PAD, PaddingSide::Right);
        let naive = padded[1].tokens.len() - 1;
        assert_eq!(padded[1].tokens[naive], PAD);
        assert_ne!(naive, padded[1].pool_index);
    }

    #[test]
    fn both_padding_sides_pool_the_same_tokens() {
        let l = pad_batch_with(&inputs(), PAD, PaddingSide::Left);
        let r = pad_batch_with(&inputs(), PAD, PaddingSide::Right);
        let lp: Vec<u32> = l.iter().map(|i| i.tokens[i.pool_index]).collect();
        let rp: Vec<u32> = r.iter().map(|i| i.tokens[i.pool_index]).collect();
        assert_eq!(lp, rp);
        assert_eq!(lp, vec![13, 20, 31]);
    }

    #[test]
    fn padding_an_equal_length_batch_is_a_no_op() {
        let same = vec![
            EncodedInput::unpadded(vec![1, 2]).unwrap(),
            EncodedInput::unpadded(vec![3, 4]).unwrap(),
        ];
        assert_eq!(pad_batch_with(&same, PAD, PaddingSide::Left), same);
        assert_eq!(pad_batch_with(&same, PAD, PaddingSide::Right), same);
    }
}
