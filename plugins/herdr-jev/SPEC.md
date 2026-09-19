# herdr-jev — build contract

**Answer it, route it, or get out of the way.**

```
task ─▶ 1. decision-shaped AND sure?  ──▶ openjev answers it.  NO AGENT RUNS.
        2. otherwise                  ──▶ pick tier + effort, start the agent with them
        3. anything else              ──▶ the agent gets it exactly as it was built
```

A local 4B cross-encoder on loopback decides, in milliseconds, whether your question
even needs an agent — and when it does, how big an agent and how much thinking. Nothing
leaves the machine.

ONE Go binary. Verified against **Herdr 0.9.x, protocol 22** and **openjev api v1**.
Module: `github.com/muthuishere/herdr-jev` · Binary: `bin/herdr-jev`

**The CLI, the plugin actions and the skill expose the same verbs.** If a verb exists in
one and not the others, that is a bug.

---

## 0. The three failures this is organised against

Stated first because every number and every default below is derived from them.

1. **Answering a generative task with a probability.** "Rewrite the parser" is not a
   question, and a two-word reply where code was wanted is worse than any latency. This
   is what §4's shape gate exists for, and why shape is checked **before** confidence.
2. **Spending less on an unmeasured hunch.** A task handed to too small a model fails
   quietly and expensively in human time. This is why the confidence bars are
   **asymmetric** (§6).
3. **Blocking a task.** A classifier is a convenience; a classifier that can stop work
   is a liability. Every failure path lands in stage 3, so a broken, slow, absent or
   confused openjev is indistinguishable from not having installed this at all.

---

## 1. Verified upstream facts (do not re-derive)

Checked against a live Herdr 0.9.x server on 2026-09-19:

- **Herdr has NO hooks.** No `UserPromptSubmit`, no prompt interception, no function
  hooks. The event stream is observation-only: `pane_*`, `tab_*`, `workspace_*`,
  `layout_updated`, `worktree_*`. `agent_prompted` is the *response* to `agent.prompt`,
  not an event. **So herdr-jev owns a submit path instead of intercepting one** — you
  dispatch through it, and it cannot reach into Herdr's own prompt box. A design that
  assumed interception would be waiting for a hook that does not exist.
- **`agent.start` takes `{name, kind, pane_id, args[], timeout_ms}`** — the agent's own
  CLI arguments are ours to choose at start. `kind` is a closed enum (`claude`, `codex`,
  `gemini`, `cursor`, `copilot`, … 23 of them). The pane must be at an interactive shell
  prompt. **This is what "pick the model" concretely reaches.**
- **`pane.split` takes `{direction, cwd, env{}, ratio, workspace_id, focus}`** — the
  launched process's environment is ours to set. That is how an agent with no model flag
  is still routable.
- **Nothing changes a RUNNING agent's model.** There is no request for it in protocol 22
  and no hook to intercept one. Routing therefore happens at **dispatch** — a pane, an
  agent start, chosen arguments — and never mid-session. See §7 for what that costs.
- `pane.list` → `PaneInfo{pane_id, terminal_id, workspace_id, tab_id, focused, agent,
  agent_status, cwd, foreground_cwd, terminal_title_stripped, label, agent_session}`.
- `pane.read` → `{pane_id, source: visible|recent|recent_unwrapped|detection, lines,
  format, strip_ansi}`.
- `agent.prompt` → `{target, text, wait?{until,timeout_ms}}`, and it **rejects a blocked
  agent with `agent_blocked` before any input is sent** — the server is the last line of
  defence, not us.
- **Pane ids are NOT stable across server restarts; `terminal_id` is.**
- Action ids must be `:`-namespaced (`--plugin` is optional on `plugin action invoke`);
  `:` is legal in an id, `.` is rejected. Ours is `jev:`.
- `[[startup]]` hooks are one-shot and unsupervised → `daemon` fork-execs, `serve` holds
  an flock (§9).
- Plugin env: `HERDR_SOCKET_PATH`, `HERDR_BIN_PATH`, `HERDR_ENV`, `HERDR_PLUGIN_ID`,
  `HERDR_PLUGIN_ROOT`, `HERDR_PLUGIN_CONFIG_DIR`, `HERDR_PLUGIN_STATE_DIR`,
  `HERDR_PANE_ID`.

openjev, from `docs/design/02-api-and-cli.md`, which wins on any conflict:

- `POST /v1/rerank` `{question, options[], top_k?, return_documents?}` →
  `{results:[{rank,index,score}]}`. `POST /v1/predict` `{pairs:[{premise,hypothesis}]}`
  → `{results:[{label, scores{contradiction,entailment,neutral}}]}` in request order.
  `POST /v1/grade` `{answer, reference, threshold}` → `{label, scores, pass}`.
- `GET /readyz` distinguishes `starting|resolving|downloading|loading|ready|draining|
  failed`, with bytes and ETA while downloading. `/healthz` stays 200 throughout. **A
  router that reports only "not ready" during a four-minute first-run download gets
  killed by an impatient human.**
- Error envelope `{error:{code,message,detail,request_id}}`. **`code` is the contract,
  `message` is not.**
- Discovery is ordered and stops at the first hit: config URL → `OPENJEV_URL` → state
  file `~/.local/state/openjev/server.json` → probe `127.0.0.1:21131` → spawn or report.
  Confirm every hit with `/healthz`; a state file outlives its process.
- `GET /v1/info` gives `api_version` (must equal 1), `capabilities` (feature-detect,
  never version-compare) and `limits` (`max_options`, `max_field_chars` — read them,
  never hardcode).

---

## 2. Inference is not ours, and it is local

herdr-jev is a **pure HTTP client** of `openjev serve`. No model, no tokenizer, no
weights, no scoring maths beyond comparing floats it was handed.

Two modes, both first-class, chosen by config:

| mode | behaviour |
|---|---|
| `attach` (default) | discover a server (§1) and use it. Never SIGTERM it, never delete its state file. Not ours to own. |
| `spawn` | `openjev serve --port 0 --state-file <ours> --print-ready-json`, block on the one ready line, supervise it, SIGTERM it on exit. |

`--port 0` + a **private** state file is why a spawned server never collides with the
user's own and is never found by somebody else's discovery. Whoever spawned it kills it.

**Privacy is a feature, not a side effect.** The reference implementation this policy is
ported from posts the prompt text to a hosted API. We post it to loopback. The task
text, the pane titles, the recent output — none of it leaves the machine. A router that
reads everything you type is a router you must trust with everything you type, and this
one removes the need. Remote backends remain a configurable fallback (§8) and are
**off**, so choosing to send text away is an explicit act.

Every HTTP call sits behind an interface, so every test in this repo runs with no model
on disk.

---

## 3. What Herdr actually lets us set — and what it does not

Being precise here matters more than being impressive.

| we want | herdr gives us | so we |
|---|---|---|
| the model a new agent runs on | `agent.start{args[]}` | pass the agent's own flag, e.g. `--model opus` |
| a setting with no CLI flag | `pane.split{env{}}` | set it in the launched process's environment |
| the reasoning effort | same two | pass the agent's flag or env; **say so when it has neither** |
| the model of a RUNNING agent | **nothing** | do not claim it, do not attempt it |
| intercepting what you type into a pane | **nothing** | own a submit path instead (§1) |

So the routed unit is **a dispatched task**: `herdr-jev dispatch "<task>" --kind claude
--pane w1:p2`. The pane is at a shell prompt; we start the agent there with the
arguments the policy chose; we prompt it with the task.

The per-agent-kind mapping is **config**, never hardcoded (§8), because claude, codex
and gemini name their models differently and a hardcoded haiku/sonnet/opus is a router
that works for exactly one agent. `doctor` prints every mapping in effect, because a
vendor renaming a flag produces an agent that is silently never routed.

---

## 4. Stage 1 — the short circuit

**When a task is decision-shaped and the local model is sure, openjev answers it and no
agent runs.** Milliseconds, zero tokens, nothing off the machine.

### 4.1 What counts as decision-shaped

An NLI cross-encoder answers entailment questions. That natively covers, and is limited
to:

| shape | framing |
|---|---|
| `boolean` | yes/no, true/false. The claim and its negation as two hypotheses over the task as premise. |
| `pick_one` | choose-one-of-N, is-this-relevant, classify-into-these-labels, did-this-succeed. Rerank over the options. |
| `grade` | answer-vs-reference, does-this-output-satisfy-this-requirement. `/v1/grade`. **Never inferred — a reference must be supplied.** |

It does **not** cover anything generative — write, edit, refactor, explain, summarise,
debug, run, deploy — **and must never try.**

### 4.2 The gate is two parts, in this order

1. **SHAPE**, decided from the text alone, with no model involved.
2. **CONFIDENCE**, decided by the model against its own bar.

**A task that is not decision-shaped is never short-circuited no matter how confident
the model is.** Confidence cannot rescue the wrong question, and the ordering is the
only thing that guarantees it never gets the chance to try.

Shape detection rules, each with its failure:

- **A generative verb anywhere vetoes**, unconditionally and first. "Should we use a
  mutex here, and if so write it?" is a yes/no clause in front of an implementation
  request; answering the clause and stopping is a silent, confident failure to do the
  work. A false veto costs one agent call — which is what would have happened anyway.
- **One narrowing only: a verb used as a noun does not veto.** "Does the state file
  survive a port change?" is a question. `change`, `test`, `review`, `plan`, `design`
  and `build` are nouns at least as often as verbs, and the marking carries across a
  compound ("a **port change**").
- **A question mark is necessary but not sufficient.** "How do I fix this?" has one.
  Only the yes/no openers and closed choices qualify.
- **Explicit `--option`s beat phrasing entirely** — a caller handing us a closed set has
  told us the shape directly. One option is refused: one option is not a choice.
- **Every detection carries its reason.** "Why was my question not answered locally?"
  must be answerable without reading the source.

The gate is guarded by a **corpus of ~45 real developer prompts** spanning both classes,
weighted towards the generative half, which is the one that must never pass. Ad-hoc cases
test the rules you already thought of; the corpus is there for the rules you did not — it
is how the `change` noun/verb bug was fixed and how the missing `profile` verb was found.

Its companion table, `knownConservativeVetoes`, is the **honest list of what the gate
gives up**: genuinely decision-shaped questions ("Can herdr change a running agent's
model?") that are vetoed because a generative verb appears in them as an ordinary word.
Separating "can herdr change X" from "can you change X" needs grammar this gate does not
have, and the fragile heuristics that would fake it are precisely how a false PASS
eventually slips through. One wasted agent call per entry is the right side of that trade,
and writing them down beats tolerating them quietly.

### 4.3 The answer bar — why 0.85

`answer.min_confidence`, default **0.85**, and it is **separate from and higher than**
the routing bars.

They are not comparable decisions. The routing bars trade money against capability, and
a wrong call is recoverable inside the same turn. This bar decides whether a human gets
a machine's two-word reply **instead of their agent's work** — a different kind of
mistake with a different kind of cost.

At 0.85, a normalised two-way answer must be roughly **6:1**, which no genuine coin flip
reaches; a four-way pick must beat a uniform 0.25 by a **factor of three**. Config
refuses a value below `min_downgrade_confidence` outright: if the answer bar were the
lower of the two, the plugin would be more willing to replace your task than to make it
cheaper, which is exactly backwards.

**Normalisation is what makes these numbers confidences.** Raw rerank scores are
independent entailment probabilities, so two options at 0.90 each are not a confident
answer — they are a tie. Dividing by the sum turns "how well does each fit" into "how
much better is the best", which is the question the bar is asking. A boolean asks both
the claim and its negation for the same reason: reading P(yes)=0.55 alone looks like a
weak yes and is indistinguishable from a coin flip.

### 4.4 Off by default, shadow first

`answer.enabled` is **false**. Answering instead of invoking the user's agent is a large
behavioural claim to make on someone's behalf, and a plugin that starts doing it the
moment it is installed has made that claim without being asked.

`answer.shadow` logs **"WOULD have answered locally (0.94)"** and dispatches to the
agent anyway. That is how the bar earns trust: run shadow for a week, read the journal,
turn `enabled` on with evidence instead of hope. Shadow wins over enabled when both are
set — shadow exists precisely so the short circuit can be watched without acting.

### 4.5 Provenance is mandatory

A short-circuited answer is the one path with **no agent transcript to read back**, so
the journal entry is the only record the question was ever asked. Every entry carries
the shape and why, the question as asked, every option with its probability, the
confidence, and the latency. Declined near-misses are kept too: they are the evidence
the bar is set right, and discarding them would make it untunable.

---

## 5. Stage 2 — tier and effort

Three questions, asked of the local model, the same three the reference policy asks:

- **`tier`** — which of three descriptions of the **work** fits: mechanical and local /
  ordinary engineering / hard or high-stakes. Reranked; the winner's normalised share is
  the confidence. **The decision model never sees a model name.** It ranks work.
- **`effort`** — a score on a four-level rubric (`almost none`, `some`, `a lot`, `as
  much as possible`). Reranked, and the **expected** rung is taken, not the argmax: the
  rubric is ordered, so a task split evenly between rungs 1 and 2 genuinely wants 1.5,
  and rounding to whichever won by 0.01 throws away the ordering the rubric is made of.
- **`risky`** — one hypothesis, `/v1/predict`, P(entailment) as the probability. No
  normalisation: the question is "how likely is this true", not "which of these".

The risk wording is load-bearing and is quoted verbatim from the reference, including
its rationale: it asks about the **act**, not the subject. "The task touches production,
money, credentials" scores near-certain on "add a refund endpoint that calls Stripe" —
ordinary code that happens to be about money — and would escalate it past a
high-confidence balanced answer.

**No tier means no decision at all.** Effort and risk alone cannot carry one, and half a
decision applied confidently is the failure this stage guards against.

---

## 6. The policy — asymmetric bars

Ported faithfully from `jev-model-router`'s `policy.ts` (ADR 0002), because its central
idea is right: **the two mistakes do not cost the same, so they do not clear the same
bar.**

| rule | value | why |
|---|---|---|
| spend **more** | `min_upgrade_confidence` **0.30** | being wrong costs money, and money is recoverable |
| spend **less** | `min_downgrade_confidence` **0.60** | being wrong means a real task gets too small a model, which is not recoverable by noticing the bill |
| `risky > 0.7` | deep tier + effort ≥ 2, **past both bars** | carrying out something final is never worth the saving. Not a confidence question. |
| no confidence reported | may only move **UP** | spending less on an unmeasured hunch is the bad trade |
| unknown model id | gentler **upgrade** bar | a direction that cannot be known must not get the strict bar — guessing wrong is how a downgrade sneaks past the bar that exists to stop it |
| numeric effort | **left alone** | it is the caller's own scale, not our ladder |
| risk clamp | risk **raises** the effort floor, never lowers one | forcing skips the thresholds, so without the clamp a task already at `xhigh`, rated mechanically simple, would be pulled down with no confidence check at all |
| `max` | ranks **above** `xhigh` | leaving `max` is a downgrade and needs the high bar |

**Both directions, both dimensions.** A task read as mechanical routes down; one read as
hard routes up — model and effort alike.

**Both switches default on, and that differs from the reference on purpose.** The Claude
Code mod ships `routeMainModel` **off**, because it rewrites the model of a conversation
that is already running: a mid-session switch invalidates the prompt cache, and on a long
context re-caching can cost more than the cheaper tier saves. **That reason does not apply
to us.** We choose the model before the agent process exists (§3), so there is no cache to
invalidate and nothing to re-pay. Turning `route_model` off here would forgo the entire
model-routing feature to avoid a cost we do not incur.

Do not "restore" the reference's default. If a future version ever gains the ability to
change a running agent's model, the trade-off returns and this paragraph is the thing to
revisit.

### 6.1 It never blocks

Non-2xx, timeout, malformed body, thrown error, latency budget exceeded
(`policy.budget_ms`, default 800) — all leave the task exactly as it was built. Stage 3
is the default that stages 1 and 2 must earn their way out of.

### 6.2 Transparency — two lines, always

Nothing in Herdr shows what a router decided: the model and effort are arguments to a
process, so no status line, header or effort box ever moves. The transcript is the only
place this work is visible, which makes the lines a contract, not decoration.

```
[herdr-jev] ready on openjev (http://127.0.0.1:21131); routing model, effort
[herdr-jev] jev: tier deep (0.95) · effort 2.8 → xhigh (0.90) · risky 0.01 · 249ms
[herdr-jev] routed → jev · deep 0.95 → opus/xhigh
[herdr-jev] jev: tier fast (0.41) · effort 0.4 → low (0.38) · risky 0.01 · 210ms
[herdr-jev] passed through: kept opus/medium, wanted haiku/low (confidence 0.41)
```

Line one is **what the model replied, before any policy**. Line two is **what the policy
then did** — including when it declined to act and why. The last pair above is a working
router deliberately doing nothing; without that line it is indistinguishable from a
plugin that never loaded.

`confidence n/d` means no confidence was reported, which may only move a request up.

### 6.3 The saving is the product

`herdr-jev savings` counts what this bought: tasks dispatched, **answered locally**
(agent calls avoided), routed, passed through, and — in shadow mode — how many *would*
have been answered. Without that number, "it sometimes answers things itself" is a claim
rather than a result.

---

## 7. What is NOT reachable, stated plainly

- **A running agent's model cannot be changed.** Not by us, not by anyone, through
  Herdr's protocol 22. A long session started on the wrong tier stays there.
- **Typing directly into a pane is not intercepted.** Only tasks dispatched through
  herdr-jev are routed. Herdr offers no hook; we do not pretend otherwise.
- **Effort is not settable for every agent kind.** Claude Code exposes no effort flag we
  can rely on across versions, so effort is not routed for it by default rather than
  guessed at — and the plan **says so in a note** rather than silently dropping it. A
  silently dropped effort would make the transcript a lie.

---

## 8. Config — TOML

`$HERDR_PLUGIN_CONFIG_DIR/config.toml`. Every field has a default that works with no
file at all.

```toml
[openjev]
mode    = "attach"        # or "spawn"
url     = ""              # explicit; wins over every other discovery step
timeout = "20s"
bin     = "openjev"       # spawn mode only
model   = ""

[answer]                  # the short circuit
enabled        = false    # opt-in: answering instead of invoking your agent is a big claim
shadow         = false    # log what it WOULD have answered; dispatch anyway
min_confidence = 0.85     # separate from, and above, the routing bars
grade_threshold = 0.5

[policy]
min_upgrade_confidence   = 0.30   # spend more: the cheap mistake, so the low bar
min_downgrade_confidence = 0.60   # spend less: the expensive mistake, so the high bar
budget_ms                = 800    # past it, the task goes through untouched
route_model  = true
route_effort = true
log_decisions = true

[agents.claude]
tiers = { fast = "haiku", balanced = "sonnet", deep = "opus" }
model_flag = "--model"
family_hints = { haiku = 0, sonnet = 1, opus = 2 }

[agents.codex]
tiers = { fast = "gpt-5-mini", balanced = "gpt-5", deep = "gpt-5" }
model_flag  = "--model"
effort_flag = "-c"                # -c model_reasoning_effort=<v>
effort_values = { xhigh = "high" }

[routing]                 # the SECONDARY pane router (§10)
floor  = 0.55
margin = 0.10

[journal]
keep = 500
```

### 8.1 The shipped agent mappings are GUESSES, and they rot

Stated plainly because the alternative is someone discovering it.

Herdr's 23-kind enum is verified against a live server. **The model names and flags under
`[agents.*]` are not.** They are best-effort defaults, written against vendor CLIs at one
moment in time, and vendors rename models and flags without asking us.

**The failure is invisible in the worst possible way.** Nothing errors. herdr-jev
classifies the task, logs a confident two-line decision, builds `--model gpt-5-mini`, and
hands it to a binary that ignores or rejects the flag. The transcript says it routed; the
agent ran on its default. Every other failure in this plugin degrades loudly into
pass-through — this one degrades into a **lie**.

So:

- `doctor` prints every mapping in effect, and reports the whole table as **UNVERIFIED**
  until it has been probed.
- **`doctor --probe`** reads each agent binary's own `--help` and reports what it really
  accepts. Opt-in, bounded, and three-state:

  | state | meaning |
  |---|---|
  | `ok` | the binary documents that flag, or names that model |
  | `FAIL` | the binary's help exists and does **not** document the flag — arguments built with it would be ignored |
  | `??` | **could not be verified here**: no binary on PATH, it timed out, or its help does not settle the question |

  `??` is never rendered as a pass. A mapping nobody could verify must read differently
  from a verified one, or the probe hands back exactly the false confidence it was built
  to remove. Model names are `??` rather than `FAIL` when absent from the help, because
  help text lists flags and almost never enumerates models — "not mentioned" is not
  evidence of refusal.

- The probe reads `--help`; it never tries the flag for real. For several of these
  binaries `<bin> --model x` starts an interactive agent or opens a session that costs
  tokens, and a check that might launch an agent is a check nobody runs.
- **It reports; it never edits config.** An auto-fix guessing a replacement model name
  would reintroduce this exact class of error with more confidence behind it.
- **Probing is never required to route.** It is a check, not a gate. A mapping that has
  never been probed still routes.

Remote backends (TypeSafe's API, the Vercel AI Gateway) are a **fallback**, configured
and off. Choosing to send task text off the machine must be an explicit act.

---

## 9. Verbs and processes

```bash
herdr-jev dispatch "<task>" --kind claude --pane w1:p2   # THE verb
herdr-jev dispatch "…" --dry-run --explain               # decide, start nothing
herdr-jev ask "is the build green?"                      # short circuit only
herdr-jev classify "<task>"                              # raw tier/effort/risky
herdr-jev why [n] · savings · status [--watch] · feed
herdr-jev doctor [--json] [--probe] · skill [--install]
herdr-jev route "<message>" · panes [--distilled] · hold list · resolve <id> --to <n>
```

Every verb takes `--json`. The flag decides the shape, never the TTY.

```
herdr
  └─ [[startup]] ./bin/herdr-jev daemon      # fork-execs, returns immediately
       └─ herdr-jev serve                    # flock pidfile -> single instance
            └─ (spawn mode only) openjev serve --port 0 --state-file … --print-ready-json
```

In `attach` mode `serve` exits 0 immediately: a supervisor with nothing to supervise
only produces logs. **`dispatch` never needs the daemon** — it is a one-shot. The daemon
exists solely so `spawn` mode has an owner for its child.

---

## 10. Secondary: the pane router

Same machinery, different question: given a message, which **already-running** pane was
it for? Each pane's live state (agent, cwd, status, title, recent output) is distilled
into one option string; the message is the question; `rerank` ranks them.

It keeps its own two numbers because it is a different decision:

```
top < floor (0.55)                  -> HOLD
top - second < margin (0.10)        -> HOLD (too close to call)
```

A **hold is not a drop**: the message is preserved with its full ranking, and
`resolve <id> --to 2` delivers it by position. The margin exists because a floor alone
is not enough — 0.86 against 0.85 clears any sane floor and is still a coin flip wearing
a number.

Distillation rules that matter: identity leads (truncation eats the tail); recent output
is capped; spinner frames and box-drawing are dropped; **the agent's own prompt box is
dropped**, because it still holds the last message routed there and leaving it in makes
the router route to wherever it last routed — confidently, consistently, and wrongly.

`panes --distilled` prints exactly what went over the wire. When a decision looks insane
the question is always "what did the model actually see", and answering it must not
require a rebuild.

---

## 11. Definition of done

- A confident yes/no question with `answer.enabled` is answered locally, `savings`
  counts it, and `why` shows both probabilities.
- The same question with the short circuit off is routed instead.
- "Rewrite the parser" is **never** answered locally, at any confidence.
- Shadow mode logs "WOULD have answered" and still dispatches.
- A 0.9/0.9 tie is refused; a 0.95/0.02/0.01 pick is answered.
- 0.51 confidence upgrades a model and does not downgrade one.
- `risky 0.93` on a "fast" task takes the deep tier and `high` effort.
- `risky 0.95` on a session already at `xhigh` does **not** lower it.
- openjev dead / slow / api v2 → the task is dispatched unchanged and `doctor` says why.
- openjev downloading → `status` says **downloading, with bytes and ETA**.
- Every prompt in the generative corpus is vetoed; every one in the decision corpus is
  recognised.
- `doctor --probe` reports ACCEPTED / REJECTED / `??` per mapping, never renders `??` as a
  pass, never edits config, and is bounded by its timeout even against a binary that hangs.
- `go vet ./...`, `go build ./...` and `go test ./...` clean.

## 12. Out of scope

- **Inference.** Not one line of model code.
- **Intercepting Herdr's prompt box.** It is not interceptable (§1).
- **Changing a running agent's model.** Herdr does not offer it (§7).
- **Compaction, summarisation, context management.** It routes; it does not rewrite.
- **Learning from corrections.** A resolve is journalled and that is deliberately all —
  an online-tuned router whose behaviour drifts is one whose misroutes cannot be
  reproduced.
- **Fan-out.** One task, one agent.
