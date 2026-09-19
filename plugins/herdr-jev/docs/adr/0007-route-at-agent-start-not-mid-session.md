# 0007 — Route at agent start, because Herdr offers nothing else

**Status:** accepted

## Context

The reference implementation is a Claude Code mod with function hooks: it intercepts
`prompt.submit`, `turn.step` and `agent.spawn`, and rewrites the model and effort of
requests the engine has already built. Herdr has no equivalent. Verified against a live
0.9.x server: there is no `UserPromptSubmit`, no prompt interception, no function hooks,
and the event stream is observation-only.

## Decision

Route where Herdr genuinely gives us a choice:

| we want | protocol 22 gives us |
|---|---|
| the model a NEW agent runs on | `agent.start{args[]}` — the agent's own CLI flags |
| a setting with no flag | `pane.split{env{}}` — the launched process's environment |
| the model of a RUNNING agent | **nothing** |
| interception of what you type | **nothing** |

So the routed unit is a **dispatched task**, and herdr-jev owns a submit path rather than
intercepting one.

## Consequences, stated plainly

- **A long session started on the wrong tier stays there.** The fix is a new pane, not a
  flag. This is a real limitation and SPEC.md §7 says so rather than implying otherwise.
- **Only tasks dispatched through herdr-jev are routed.** Typing into a pane is not.
- The reference's reason for keeping main-model routing off — mid-session switches
  invalidate the prompt cache — **does not apply to us**, because we choose before the
  process exists. So both switches default on.
- The per-agent-kind flag mapping is config and is printed by `doctor`, because a vendor
  renaming a flag produces an agent that is silently never routed: the plugin logs a
  decision, builds arguments nothing accepts, and the model never changes.
