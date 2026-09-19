# 0006 — First-run consent: ask on a TTY, exit 77 off one

Status: **accepted** (openjev-cli, 2026-09-19).

## Context

`JevError::ConsentRequired` exists in core and core never raises it: TTY detection and
prompting are explicitly CLI-side. The first run fetches ~2.7 GB+ (more once the real
checkpoint is published).

Two real failures pull in opposite directions:

1. a tool that silently burns gigabytes of a metered connection;
2. a tool that hangs forever on a prompt nobody can see, under systemd or in CI.

## Decision

Before the first byte, the CLI computes what is missing and how large it is, then:

- **stdin and stderr are both TTYs** → print the size, the one-time-ness and the
  resumability, and ask. Default yes on empty input.
- **otherwise** → fail immediately with `JevError::ConsentRequired`, exit **77**, naming
  `--yes` / `OPENJEV_ASSUME_YES=1`.
- **below 64 MiB of missing files** → no prompt. Consent is about a wait, not about
  bytes; a tokenizer refresh asking permission is noise that trains people to type `y`.

Disk space is checked in the same place: free space must beat the download by 10 %, or
we abort with exit 4 *before* the first byte. ENOSPC at 94 % of a multi-GB download is
the cruellest failure available to us and it is entirely preventable here.

## Consequences

`openjev serve` on a fresh box under systemd will exit 77 the first time, with the fix in
the message. That is deliberate: the unit file should call `openjev model pull --yes` as
`ExecStartPre`, or set `OPENJEV_ASSUME_YES=1`. A silent multi-GB download from a service
start is a worse default than one loud failure with a one-line remedy.
