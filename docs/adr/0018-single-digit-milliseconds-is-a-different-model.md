# 0018 — single-digit milliseconds is a different model, and one exists that llama.cpp already runs

Status: **proposed** — measured 2026-09-20, **not built**. The numbers are here so the
decision is the owner's and not an engineer's preference wearing a table.

## The failure this is organised against

Answering a latency target with more engineering on the model you already have. ADR 0017
spent a day proving there is at most 18% in the 4B path. The target is 6x. Every hour after
that measurement spent on kernels would have been an hour spent on the wrong question.

## The small openjev checkpoints do not exist

Checked, rather than assumed. `AlexWortega/openjev` at `4b5f9a67`, full recursive tree:
**three** model subfolders and **two** distinct sizes.

| subfolder | bytes | architecture | params |
| --- | --- | --- | --- |
| `qwen3.5-4b-nli-v2/` (ours) | 9,078,635,984 | `Qwen3_5ForSequenceClassification` | ~4.5B |
| `qwen3.5-4b-nli/` (v1) | 9,078,635,984 | same | ~4.5B |
| `qwen3.5-35b-a3b-nli/` | 69,197,389,504 | `Qwen3_5MoeForSequenceClassification` | ~34.6B |

The 0.8B *was trained*. `results/qwen0.8b_mnli_gpqa.json` is keyed on `ckpt/qwen3.5-0.8b-nli`
— the author's local training directory — and scores MNLI-m 0.8686. `results/qwen2b_full.json`
likewise reports `ckpt/qwen3.5-2b-nli` at MNLI-m 0.8862. **Neither path exists in the repo
tree, in the author's 169-model account listing, or anywhere else on the Hub.** The earlier
"0.8B at 57 pairs/s" figure was an estimate for a checkpoint that cannot be downloaded.

**The 0.6B / 2B / 4B on the website are not NLI checkpoints.** `openjev.com` now serves
SemIf, and its three variants, read out of its `app.js`, are `Qwen/Qwen3-0.6B-GGUF`,
`openbmb/MiniCPM5-2B-GGUF` and `bartowski/Qwen_Qwen3.5-4B-GGUF` — stock general LLMs run
in-browser for typed option-logits. No 3-label score head is involved. Reading the site as
a menu of NLI trunks would have been a confident wrong turn.

**Distillation, as an option and not a project.** Teacher is on disk; the student would be
Qwen3.5-0.6B with a fresh 3-label head, trained on MNLI+ANLI+FEVER (~1M pairs) soft-labelled
by the 4B. A few GPU-days on one rented A100/H100, so low hundreds of dollars, plus the
labelling pass. The reason not to start is the next section: it would be reproducing, at
cost and at risk, a model that already exists and is better.

## What does exist, runs in our runtime today, and answers in 10 ms

llama.cpp has **no DeBERTa support at any quantisation** — zero matches for `deberta` in
`src/llama-arch.cpp` or anywhere in the tree — which rules out the whole
mDeBERTa/DeBERTa-v3 family that the question naturally reaches for. It *does* register
`ModernBertForSequenceClassification`, and converts the classification head with it
(`cls.output.weight`, `{arch}.classifier.output_labels`).

Two candidates converted and measured on this machine:

| | ModernCE-base-nli | nli-distilroberta-base |
| --- | --- | --- |
| params / arch | 149.6M ModernBERT | 82.1M RoBERTa |
| GGUF F16 | 301 MB | 167 MB |
| **warm p50, real pair, end to end** | **10.4 ms** | **8.6 ms** |
| p90 | 15.2 ms | 9.9 ms |
| **label agreement vs the 4B, 35 pairs** | **32 / 35** | **29 / 35** |
| llama.cpp reproduces its own HF result | 35/35 argmax, mean abs dp 0.0015 | 35/35, 0.0018 |
| licence | MIT | Apache-2.0 |

**6x faster than the 4B, and it is the real number** — warm, over a running server, median
of 100 calls, not a `llama-bench` extrapolation (`llama-bench` reports 12–27 ms for these
and is dominated by fixed overhead at 32 tokens, with ±16% variance; it overstates the
gap in our favour, which is why it is not the number quoted).

Q8_0 buys nothing on speed for either — Metal's F16 path is already the fast one — and
halves the file at zero argmax flips.

### 32/35 is not 35/35, and the three are named

* **29** — *"Die Müller-Straße ist wegen Bauarbeiten gesperrt." / "Die Straße ist offen."*
  4B contradiction 0.994, ModernCE neutral 0.531. German. **Low confidence, so a gate
  catches it.**
* **30** — a long AGM passage / *"The company reduced its dividend."* 4B neutral 0.444,
  ModernCE entailment 0.743. The 4B is arguably the one that is wrong here; the passage
  says the board would recommend a cut.
* **32** — a frozen-river/mill passage / *"The mill was restored as a museum."* 4B
  contradiction 0.640, ModernCE entailment 0.979. **Confidently wrong, and a confidence
  gate does not catch it.** This is the honest cost.

distilroberta's six failures are a pattern rather than a scatter: implicature and
veridicality (*"almost won"* read as entailing *"won"*; *"probably come"* as entailing
*"He came"*) and non-English. It is 2 ms faster and meaningfully dumber.

### Two traps, both silent, both found by checking rather than reading

1. **`dleemiller/ModernCE-base-nli`'s `config.json` declares the wrong `id2label`.** It
   says `{0: entailment, 1: neutral, 2: contradiction}`. The head is empirically
   `[contradiction, entailment, neutral]` — the same order as ours. Trusting the card
   produces a clean 3-way rotation and **2/35 agreement**; the identity mapping gives
   32/35. The bad labels propagate into the GGUF metadata, so llama.cpp prints them too.
   Anything adopting this model must override positionally, or fix `config.json` before
   converting.
2. **`llama-server`'s `/v1/rerank` throws away two of the three logits.**
   `send_rerank()` ends `res->score = embd[0]` — it keeps logit 0, which for this label
   order is *contradiction*. Not an error; a silent truncation, and the wrong one to keep.
   The working 3-label path is `--pooling rank` with **`--embd-normalize -1`** (the default
   L2 normalisation destroys the logits and prints all three as `0.000`), or the C API
   equivalent.

## Decision

**Not built.** What is recorded is that the option is real, the runtime is the one we
already ship, and the price is 3 pairs in 35.

The shape it would take, if taken: a second registry entry — a model is already
configuration — with `pooling = rank` and llama.cpp's own head instead of the Rust one, a
sentence-pair template with `[SEP]` rather than our `Premise:/Hypothesis:` prose, and the
plugin's existing confidence machinery as the escalation gate. **A gate at 0.9 routes
pair 29 and every multilingual case up to the 4B and costs 61 ms only when it fires.**
Pair 32 goes through the gate confidently wrong; there is no version of this that does not.

Three questions the owner owns, not the engineer:

1. **Is 32/35 acceptable as a first answer** when the 4B is one confidence bar away? The
   tier is not a replacement; it is a filter with a named false-confidence rate that 35
   pairs bounds only loosely (ADR 0014's caveat applies here with more force, not less).
2. **A second architecture is a second thing that breaks.** ModernBERT means the
   `cls.output` path this codebase's module docstring explicitly says it avoided. That
   avoidance was correct for a 3-way head on a hybrid recurrent trunk; it is not
   automatically correct for a bidirectional encoder that ships the head in the GGUF.
3. **The golden fixture would need to grow first.** 35 pairs is enough to prove the 4B
   matches its own reference. It is not enough to sign off a second model that disagrees
   with it on 3 of them — a few hundred SNLI/ANLI dev pairs is design risk R1, and this is
   the decision that finally makes it load-bearing.

## Consequences

* **The 0.8B line in STATUS.md was wrong and is withdrawn.** "57 pairs/s available today"
  described a checkpoint that has never been published.
* The fast tier, if adopted, is 6x and not 3.5x — and it is a model whose MNLI score
  (0.909/0.921 for the ModernCE family) *beats* the unreleased 0.8B openjev (0.869), which
  removes most of the reason to want the unreleased one.
* llama.cpp's DeBERTa gap is now a known fact rather than a thing to rediscover: the
  strongest small NLI models in the literature are all DeBERTa-v3, and none of them can run
  on this runtime at any quantisation. That constraint, not accuracy, is what selected
  ModernBERT.
