# jevd

An inference server for [openjev](https://huggingface.co/AlexWortega/openjev), a small
NLI cross-encoder: premise and hypothesis in, probabilities over `contradiction` /
`entailment` / `neutral` out.

One primitive, no task-specific training, and cheap enough to sit in a hot path. It is
a good enough decision engine that you can rank, route, gate and grade with it instead
of prompting a chat model and hoping.

```sh
openjev serve
```

First run says what it is about to download and how big it is *before* it downloads
anything. The server reports *downloading* and *loading* as different states, so a
supervisor cannot kill a slow model load.

| crate | what |
| --- | --- |
| `openjev-core` | model acquisition, device selection, inference. `predict` / `rerank` / `grade` |
| `openjev-cli` | the `openjev` binary and the HTTP API |

## Engine

llama.cpp as the trunk only — hidden states out — with the three-label head applied in
Rust from a small safetensors. Keeping the classifier out of GGUF means every backend's
contract is one method, `forward(batch) -> last hidden state`, and everything else is a
free function that backends cannot disagree about.

CPU, Metal, CUDA, Vulkan and ROCm are cargo features over one codebase. The device is
detected, with an explicit override and a logged demotion chain; a device whose backend
was not compiled in fails loudly naming the missing feature rather than quietly falling
back.

**A model is config, not code.** An embedded registry plus `~/.config/openjev/models.toml`
carries the repo, revision, template, label map, tokenizer and head shape, so a different
checkpoint — or later a different architecture — is an edit, not a release.

## Verified

100% label agreement (35/35) against reference `transformers` outputs at every
quantisation level, with token ids matching exactly. Q8_0 is both the most accurate quant
and the fastest, so there is no accuracy/speed trade to make.

See `STATUS.md` for the numbers, `docs/design/` for the decisions and what each is
organised against, and `docs/adr/` for how they changed.

## Using it

The HTTP API is the whole interface. Nothing in the engine knows about any particular
client.

A Herdr plugin was built on it — a three-way dispatcher that answers decision-shaped
tasks locally, routes the rest to an agent at the right model tier, and passes anything
uncertain straight through. It lives outside this repo; the engine deliberately carries
no knowledge of it.

MIT.
