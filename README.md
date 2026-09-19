# my-jev

Three things in one repo, around one idea: a small NLI cross-encoder is a good enough
decision primitive that you can route, rank and judge with it instead of prompting a
chat model and hoping.

The model is [openjev](https://huggingface.co/AlexWortega/openjev) — it reads a premise
and a hypothesis and returns probabilities over `contradiction` / `entailment` / `neutral`.
One primitive, no task-specific training, and cheap enough to sit in a hot path.

| component | what it is |
| --- | --- |
| `crates/openjev-core` | Rust library: model acquisition, device selection, inference. `predict` / `rerank` / `grade` |
| `crates/openjev-cli` | the `openjev` binary. `openjev serve` downloads on first run, then serves HTTP |
| `plugins/herdr-jev` | Go [Herdr](https://herdr.dev) plugin: a semantic router for panes |

## openjev serve

One command, no prerequisites. The first run says what it is about to download and how
big it is before it downloads anything, and the server reports *downloading* and *loading*
as different states so a supervisor cannot kill a slow model load.

The HTTP API is the public interface. Herdr is entirely optional — nothing in the engine
knows it exists.

## herdr-jev

When you send a message to an agent, the pane that happens to be focused is rarely the
pane that should get it. herdr-jev snapshots every pane, asks openjev to rank them against
the message, and delivers to the best match.

Herdr has no `UserPromptSubmit` hook — its event stream is observation-only, and
`agent_prompted` is the response to `agent.prompt`, not an event you can intercept. So the
plugin does not intercept the submit path; it **owns** one.

Below a configured confidence floor it holds the message and asks. Routing confidently to
the wrong agent is the failure the whole design is organised against.

## Design

`docs/design/` carries the decisions and what each is organised against; `docs/adr/`
records changes to them.

## Status

Early. Nothing here is released yet.

MIT.
