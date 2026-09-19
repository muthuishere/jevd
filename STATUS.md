# Status — 2026-09-19, overnight build

Four agents, one design phase, one build phase. Everything below was verified by
running it, not by reading it.

## Green

`task check` passes: `cargo fmt --check`, `cargo clippy --workspace --all-targets
-D warnings` (zero warnings), 139 Rust tests, and `go vet` / `go build` /
`go test ./...` across 7 packages, 0 failures.

29 commits. Remote is `muthuishere/herdr-jev`.

## What exists

| component | state |
| --- | --- |
| `crates/openjev-core` | Backend trait, llama.cpp backend, model registry, weight acquisition, tokeniser/template/pooling, the 3-label head, `predict`/`rerank`/`grade`/`latents` |
| `crates/openjev-cli` | full command tree, HTTP server, one error envelope, readyz phases, consent gate, flock + state file |
| `plugins/herdr-jev` | three-way dispatch, the asymmetric-bar policy, the short circuit, `doctor --probe`, the agent skill |

## The question that mattered, and its answer

**R2 — does llama.cpp reset the gated-DeltaNet state between sequences?** A leak
would be a confident wrong label, not a crash, so it was measured before any
product code was written.

On CPU the hidden state is **bit-identical** after a 1400-word distractor
(`relL2 0.000e0`). On Metal the deviation is `7.5e-4` and **flat across a 200×
change in preceding state** — identical when the "distractor" is a verbatim copy
of the target. Six orders of magnitude below the semantic control (`1-cos 7.0e-1`).
That is non-associative float reduction under different kernel tiling, not
contamination. Measured twice, independently: through `llama-embedding` and
through our own `Backend::forward`.

**Throughput:** 57 pairs/s at 64 tokens on the 0.8B.

## Three bugs found by the work, not by review

1. **All-zero hidden state under parallel context creation.** Five backends open
   at once left exactly one returning zeros — no error, no panic, no log. Through
   the head that is a uniform softmax whose argmax is always `contradiction`: a
   silent, deterministic wrong label. Fixed by serialising model creation and
   refusing all-zero states outright.
2. **The shape gate let a generative task through.** *"Is this slow, and can you
   profile it?"* passed because `profile` was missing from the verb list — the
   exact failure the gate exists to prevent.
3. **`CommandContext` does not bound `CombinedOutput`.** A killed shell's
   surviving grandchild holds the pipes open; a 300ms budget ran the full 30s.
   `cmd.WaitDelay` is what makes the deadline real.

## What is NOT proven

**No real forward pass has ever run.** Every test is weightless by design. The
engine is proven correct in structure, not against real weights.

- Our checkpoint is not converted to GGUF, so `models.toml` carries a floating
  `revision = "main"` with an empty digest. It warns at boot; it is not pinned.
- CUDA and Vulkan are wired but unexercised — only CPU and Metal are measured.
- `usage.tokens` is always 0; core's `Session` returns no token count.
- The tier→model tables are guesses against vendor CLIs. `doctor --probe` reports
  which the binaries actually accept; on this machine `claude --model` and
  `codex --model`/`-c` are confirmed, `sonnet`/`opus` appear in claude's help,
  **`haiku` does not**, gemini is not installed.

## Decisions taken while you slept

- **`doctor --probe` reads `--help` rather than trying the flag.** `<bin> --model x`
  starts an interactive agent for several kinds; a check that might launch an
  agent and spend tokens is a check nobody runs.
- **Build llama.cpp from source in v0.1.** `llama-cpp-sys-2` has no escape hatch
  for linking an existing libllama. `mochallama` declined the same trade: it
  compiles from a pinned tag in CI and ships the artefact. "The user never
  compiles" comes from building once in CI, not from linking someone else's libs.
  (ADR 0002.)
- **The repo is `muthuishere/herdr-jev`**, as asked, though it now holds the Rust
  engine and CLI too. `gh repo rename` is one command and GitHub redirects the
  old URL.

## Yours to decide

1. **Convert and publish the GGUF + head safetensors.** Nothing runs for real
   until this exists, and it is the only remaining blocker to a working
   `openjev serve`.
2. **Does the short circuit ship on or off?** It is off by default with a shadow
   mode, because something that silently answers instead of invoking your agent
   is a large behavioural claim. Run shadow mode on your own work, look at what
   it *would* have answered, then decide the bar.
