# 0017 — `/v1/systemone` copies TypeSafe exactly, except where copying it would be a lie

Status: **accepted** (openjev-cli, 2026-09-20). **Organised against the failure where a
"compatible" endpoint is 95% compatible and the missing 5% is discovered in production by
someone else's client.**

## Context

TypeSafe's System One API (`POST api.typesafe.ai/v1/systemone`) takes one unstructured
`state` and a map of typed questions, and returns typed answers with no free-form text.
Clients already exist and are shipped: the `jev-model-router` Claude Code mod and
`tamaratran/fast-jev-compaction` both post that body and read that response. Serving the
same shape means those clients work against a local server by changing a base URL — no
code change, nothing leaving the machine.

**Compatibility is the feature.** Every gratuitous difference destroys the reason to build
this at all. So the shape was not designed from the one example we were handed; it was
read off the published docs (`docs.typesafe.ai/api.md`, `/models.md`, `/primitives/*.md`)
and off the clients, which encode it precisely.

What that reading settled, which the example alone would not have:

* **`state` is not a string.** It is string *or* object *or* array.
  `fast-jev-compaction` sends `{context, goal, history:[...]}`. Typing it as `String`
  would have 400'd a shipped client on its first request.
* **`instructions` is also any JSON** — the "structured instructions" of
  `primitives/advanced.md`.
* **A `noul` may carry no `criteria` at all.** `jev-model-router`'s `risky` question is
  instructions only. Requiring criteria would have broken it.
* **A `score` answer must carry a `legend`.** It is required by
  `primitives/score.md` and is absent from the example we were given.
* **A `noul` answer carries the probability and nothing else** — no `confidence`, no
  `probabilities`. Adding them would be as wrong as omitting them.
* **`score` is the probability-weighted mean of the level indices**, in
  `[0, levels-1]`, and may land between levels. `jev-model-router` rounds it
  (`effortLevel` does `Math.round`), so a fractional value is what the client expects.
* **`boolean` is the Vercel AI Gateway's rename of `noul`**, answered as `probability`,
  not a fourth TypeSafe primitive. We accept it anyway: a client configured for the
  Gateway shape then works here too, and refusing it buys nothing.

There is no published OpenAPI document and no documented error-body schema on
typesafe.ai; those are the parts we had to decide for ourselves.

## Decision

Serve `POST /v1/systemone` alongside `/v1/predict|rerank|grade`, which do not change. Four
question types: `noul`, `choice`, `score`, `boolean`. Request and answer fields are
copied. Three things are deliberately ours:

1. **`provider` reports `"openjev"`, never `"TypeSafe"`.** This is the one divergence a
   client can branch on, and it is the one we refuse to hide. A field whose job is to say
   who answered must say who answered. A server that claims to be TypeSafe makes every
   downstream log, cost report and incident timeline wrong in a way nobody can debug from
   the outside — and a client that behaves differently per provider would be choosing that
   behaviour on a forged premise. Impersonation is not compatibility.
2. **Errors use this server's one envelope** — `{"error":{"code",...}}` with `code` as the
   contract — not TypeSafe's `{"message","error_type"}`. One error shape across every
   endpoint beats two, and the clients we read treat any non-2xx as a failure without
   parsing the body. New codes: `invalid_question`, `unknown_question_type`,
   `empty_criteria` (422), `too_many_questions`, `state_too_long` (413).
3. **`usage` reports what is true here.** `input_tokens` is real, counted by the encoder
   that ran. `output_tokens` is **0**: a cross-encoder emits no tokens, it emits one
   distribution over labels per pair, and any other number would be a fabrication shaped
   like a measurement. `cost` is **0**: this runs on your machine. Both are documented as
   constants rather than left to look like a coincidence.

`id` is generated in TypeSafe's shape (`gen-dec-<unix>-<20 chars>`) because clients may
log or dedupe on it, and `model` is echoed back, because a client that asked for
`jev-latest` and got `openjev` back cannot tell a compatible server from a misrouted one.
What actually answered is on `/v1/model`.

## Assumed, not verified

Named here so nobody later mistakes a guess for a citation:

* **Max questions per request.** Not documented upstream; we impose 32
  (`server.max_questions`), surfaced in `/v1/info`. An unbounded question map is an
  unbounded number of forward passes behind one admission slot. `max_criteria` is 255,
  which *is* TypeSafe's documented cap on a choice.
* **A `choice` with one criterion, and empty criterion descriptions, are 4xx.** Upstream's
  behaviour is unknown. Answering a one-option choice would report that option with
  confidence 1 whatever the state said — a confident answer that measured nothing, which
  is worse than a refusal.
* **`score` accepts an object keyed by integers** as well as the documented array. Only
  integer keys, so the order is recoverable; any other object would make the score depend
  on key spelling.
* **A 413 for `too_many_questions` / `state_too_long`.** Upstream documents 422 for
  validation generally. These are size refusals, and the rest of this server already says
  413 for those.

## Consequences

`/v1/info` grows `capabilities: ["systemone"]` and two limits; clients feature-detect on
that, never on the version. The endpoint is additive — no existing route changed shape.

A client that branches on `provider == "TypeSafe"` will not take its TypeSafe branch here.
That is intended and is documented in the README. If such a branch ever needs supporting,
it needs supporting by name — a configured alias the operator sets deliberately — not by
this server lying about itself by default.
