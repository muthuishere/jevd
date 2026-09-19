# 0009 — `--revision` cannot override the registry; it refuses

Status: **accepted** (openjev-cli, 2026-09-19). Divergence from design 02 §2.3/§2.5,
which lists `--revision` on `serve` and `model pull`.

## Context

`openjev-core`'s `BootOptions` has no revision field. A revision belongs to a model
*entry* in the registry, alongside the per-file sha256 pins. Core resolves weights,
tokenizer and head each against that entry.

## Decision

`--revision` is accepted and, when it disagrees with the resolved model's revision, is a
**hard error (exit 3)** naming `~/.config/openjev/models.toml` as the real lever.

## Why not thread it through

Because a revision applies to a set of files with pinned digests. Overriding it at the
CLI would fetch files from one revision and validate them against another's hashes — or,
worse, skip validation and load weights nobody pinned. That is the wrong-weights failure
the registry exists to prevent, and a confidently wrong label is this system's worst
outcome.

## Consequences

Pinning a different revision is a three-line models.toml entry, which is also the thing
that gets committed and reviewed. `models.toml` currently ships `revision = "main"` with
empty sha256s because the checkpoint is not published; that floating revision produces a
loud warning at boot, on `doctor`, and as a `warning` event on `/v1/events`.
