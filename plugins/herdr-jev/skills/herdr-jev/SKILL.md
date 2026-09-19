---
name: herdr-jev
description: Answer a task locally instead of invoking an agent, or pick the model tier and reasoning effort an agent starts with — using a local openjev cross-encoder on loopback, with nothing leaving the machine. Trigger on - dispatch this task, which model should this run on, size the agent for this, answer this yes/no locally, is this worth opus, route this to the right pane, why did it pick that model, how many agent calls did we avoid, herdr-jev.
---

# herdr-jev

**Answer it, route it, or get out of the way.**

A local 4B cross-encoder (`openjev serve`, loopback) decides three things about a task,
in this order:

1. **Can it just answer it?** A yes/no, a pick-one, or a grade — answered in
   milliseconds, no agent invoked, zero tokens, nothing off the machine.
2. **If not, how big an agent?** A tier (fast / balanced / deep) and a reasoning effort,
   applied as the agent process's own flags at `agent.start`.
3. **If anything is unclear, it does nothing.** The task goes to the agent exactly as it
   was built. It never blocks.

## Verbs

```bash
# THE verb. --pane must be a pane at an interactive shell prompt.
herdr-jev dispatch "add pagination to the users endpoint" --kind claude --pane w1:p2

herdr-jev dispatch "..." --dry-run --explain     # decide, start nothing
herdr-jev dispatch "..." --model sonnet --effort medium   # what it would otherwise use
herdr-jev dispatch "which of these?" --kind claude --pane w1:p2 --option a --option b

herdr-jev ask "Is the confidence floor 0.55?"    # short circuit only
herdr-jev classify "migrate the payments table"  # raw tier/effort/risky
herdr-jev why 5                                  # the last 5 decisions, fully explained
herdr-jev savings                                # agent calls avoided
herdr-jev status · doctor · skill --install

# Secondary: which ALREADY-RUNNING pane was this message for?
herdr-jev route "the rerank test is flaky again"
herdr-jev panes --distilled                      # exactly what the model sees
herdr-jev hold list · herdr-jev resolve h-123 --to 2
```

Every verb takes `--json`.

## The judgement calls you need to know

**The short circuit is OFF by default.** Answering instead of invoking someone's agent
is a big claim. Turn on `answer.shadow` first — it logs *"WOULD have answered locally
(0.94)"* and dispatches anyway — read a week of `herdr-jev why`, then set
`answer.enabled`.

**Generative tasks are never answered locally, at any confidence.** Shape is checked
before confidence, from the text alone. "Rewrite the parser" is vetoed even if the model
is certain. If you want the local answer, phrase it as a question.

**The two confidence bars are deliberately different.** Spending *more* needs 0.30;
spending *less* needs 0.60. Being wrong about spending more costs money; being wrong
about spending less means a real task got too small a model. Do not "tidy" them into one
number — that removes the whole design.

**`risky > 0.7` takes the deep tier and real reasoning past both bars.** That is not a
confidence question. It also only ever *raises* the effort floor, never lowers one.

**A running agent's model cannot be changed.** Herdr offers no such request and no hook.
Routing happens when an agent is *started*. If a long session is on the wrong tier, the
answer is a new pane, not a flag.

## When it looks broken

Run `herdr-jev doctor` first — every failing check names the exact command that fixes
it. The three usual answers:

1. **"It never answers anything."** `answer.enabled` is false (the default). Check
   `doctor`'s "short circuit" line.
2. **"It never changes the model."** Either the agent kind has no `model_flag`
   configured, or the tier names do not match anything the agent accepts. `doctor`
   prints every mapping in effect.
3. **"Nothing happens at all."** openjev is not reachable, or is still downloading its
   weights. `herdr-jev status` says which, with bytes and ETA.

By design, none of these stop your task — which is exactly why you have to ask.
