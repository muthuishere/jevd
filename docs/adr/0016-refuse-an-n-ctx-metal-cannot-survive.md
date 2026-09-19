# 0016 — refuse an `n_ctx` ggml-metal cannot survive, because the alternative is a segfault

Status: **accepted** (openjev-core, 2026-09-20).
Organised against: **a user editing a config field and getting a dead process with no
message.**

## Context

Bisected on an M5 Pro, reproducible:

| `n_ctx` | Metal | CPU |
|---|---|---|
| <= 28672 | ok | ok |
| 32768 | **SIGSEGV** | ok |
| 65536 | **SIGSEGV** | ok |

Metal only, at `max_seqs` 1 as well as 4, so it is not the batching path. It is not a
trained-context limit either — the GGUF declares `qwen35.context_length = 262144`. It is a
ggml-metal allocation ceiling reached with no bounds check.

This crate walked into it already: the first multi-sequence implementation computed
`n_ctx = context * max_seqs` = 131072 and segfaulted. That was diagnosed at the time as
"too large an allocation", which was right by accident and for the wrong reason.

**Why it matters more than a maintainer tripping over it.** `context` is user-editable
registry data in `~/.config/openjev/models.toml`. The default of 8192 is nowhere near the
cliff, but the config surface is exactly what invites someone to raise it — and what they
get is a process that dies with no error, no log line, and nothing naming the cause. That
is ADR 0003's failure class: silent, and from a direction the user cannot diagnose.

## Decision

**Refuse at `open`, before the allocation, when the device is Metal and `n_ctx` exceeds a
conservative constant (28672).** The error names the value, the ceiling, the device, the
config key to change, and `--device cpu` as the escape.

`OPENJEV_UNSAFE_METAL_MAX_CTX` raises the constant for someone on a different GPU who has
measured their own ceiling. It is named "unsafe" because it is.

### Why not probe

A probe is the better design and it is not available. The failure is a **segfault inside
ggml**, not an error return — by the time we could observe it there is no process left to
report it. Probing would mean forking a child process per boot to find out whether the
parent may allocate, which costs a process launch and a model load on every start to guard
a value almost nobody changes.

### Why a hardcoded number is acceptable here despite being a guess

The real ceiling differs per GPU, so 28672 is certainly wrong for some device — too low on
a bigger one, possibly still too high on a smaller one. Three things make that tolerable:

* Erring **low** costs a config edit and prints why. Erring high costs a segfault. The
  directions are not symmetric, so the guess should be conservative and is.
* The workload does not want the context. NLI pairs are tens to hundreds of tokens; 28672
  is already ~50x the longest pair in the golden fixture.
* The override exists, so the constant is a default rather than a cap.

## Consequences

Raising `context` past 28672 on Metal is now a startup error rather than a crash. CPU is
untouched and still runs at 65536 — verified, same answer as at 8192.

The number will rot. When a device appears where 28672 is wrong in either direction, this
ADR is the place that says what it was measured on and why the direction of the error was
chosen; re-measure and move it rather than deleting the check.
