# herdr-jev

**Answer it, route it, or get out of the way.**

A [Herdr](https://herdr.dev) plugin that puts a local 4B cross-encoder in front of your
coding agents. Nothing leaves the machine.

```
task ─▶ 1. decision-shaped AND sure?  ──▶ answered on loopback.  NO AGENT RUNS.
        2. otherwise                  ──▶ pick tier + effort, start the agent with them
        3. anything else              ──▶ the agent gets it exactly as it was built
```

```console
$ herdr-jev dispatch "design the sharding scheme for the events table" --kind claude --pane w1:p2
[herdr-jev] jev: tier deep (0.95) · effort 2.8 → xhigh (0.90) · risky 0.04 · 211ms
[herdr-jev] routed → jev · deep 0.95 → opus/xhigh

$ herdr-jev ask "is the confidence floor 0.55?"
[herdr-jev] answered locally in 38ms (boolean, confidence 0.94) — no agent invoked

  0.940  yes
  0.060  no

yes

$ herdr-jev savings
412 task(s) dispatched
  96 answered locally  — agent calls avoided, 0 tokens, nothing left the machine
 204 routed            — model or effort chosen for them
 112 passed through    — left exactly as built
```

## Why

Most of what gets typed at a coding agent is a task. A surprising amount of it is a
*question*. Sending "is the build green?" to a 200B model over the network is an absurd
trade when a 4B cross-encoder on loopback answers it in 40ms — and when it *is* a task,
the same model can tell you whether it needs the big agent or the cheap one.

The policy is a faithful Go port of TypeSafe's [`jev-model-router`][ref] Claude Code mod,
whose asymmetric-confidence design is the right one: **spending more needs 0.30 confidence,
spending less needs 0.60, because the two mistakes do not cost the same.** What we changed
is where the answer comes from — their backend is a hosted API that receives your prompt
text, ours is `openjev serve` on 127.0.0.1.

[ref]: https://github.com/davila7/claude-code-templates

## Install

```sh
herdr plugin link .        # registers this checkout
herdr-jev doctor           # every failing check names its own fix
herdr-jev skill --install  # symlink the agent skill into ~/.claude and ~/.agents
```

Needs a reachable `openjev serve` (attach mode, the default) or an `openjev` binary on
PATH (spawn mode). Copy `config.example.toml` to `$HERDR_PLUGIN_CONFIG_DIR/config.toml`
to change anything.

**The short circuit ships off.** Answering instead of invoking your agent is a big claim
to make on your behalf. Set `answer.shadow = true`, let it log what it *would* have
answered for a week, then turn `answer.enabled` on with evidence. See
[ADR 0006](docs/adr/0006-short-circuit-off-by-default-with-shadow.md).

## What it can and cannot do

Herdr has no hooks and no prompt interception, so herdr-jev owns a submit path instead of
intercepting one. It picks a model where Herdr genuinely offers a choice — `agent.start`'s
arguments and `pane.split`'s environment — which means **routing happens when an agent is
started, never mid-session**. A long session on the wrong tier stays there; the fix is a
new pane. [ADR 0007](docs/adr/0007-route-at-agent-start-not-mid-session.md) is explicit
about this rather than implying more.

## Design

[`SPEC.md`](SPEC.md) is the build contract. The ADRs carry the arguments:

| | |
|---|---|
| [0001](docs/adr/0001-three-way-decision-answer-route-pass.md) | the decision is three-way, and pass-through is the default the others must earn |
| [0002](docs/adr/0002-port-the-asymmetric-confidence-policy.md) | port the asymmetric-confidence policy rather than invent one |
| [0003](docs/adr/0003-shape-before-confidence.md) | shape is checked before confidence, from the text alone |
| [0004](docs/adr/0004-the-answer-bar-is-separate-and-higher.md) | the answer bar is separate from, and above, the routing bars |
| [0005](docs/adr/0005-normalise-scores-into-a-confidence.md) | a raw entailment score is not a confidence |
| [0006](docs/adr/0006-short-circuit-off-by-default-with-shadow.md) | the short circuit ships off, with a shadow mode |
| [0007](docs/adr/0007-route-at-agent-start-not-mid-session.md) | route at agent start, because Herdr offers nothing else |

## Develop

```sh
./scripts/build.sh
go vet ./... && go test ./...
```

Every HTTP call sits behind an interface, so the whole test suite runs with no model on
disk and no network. The tests in `internal/policy/policy_test.go` are ported from the
reference implementation's own suite — every case in them is a bug somebody already found.
