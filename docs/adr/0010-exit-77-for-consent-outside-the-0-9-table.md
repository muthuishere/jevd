# 0010 — Consent exits 77, outside design 02's 0–9 table

Status: **accepted** (openjev-cli, 2026-09-19). Divergence from design 02 §2.6.

## Context

Design 02 enumerates exit codes 0–9 plus 130 and has no slot for "a download needs
consent". `openjev-core`'s `JevError::exit_code()` already answers 77 for
`ConsentRequired` (sysexits' EX_NOPERM neighbourhood).

## Decision

The CLI translates every other core sysexit into the 0–9 table (78 → 3, 69 → 1, and so
on) and passes **77 through unchanged**.

Two numbers for one condition — 77 from the library, something else from the binary — is
worse than one number outside a table, because the library's number is the one that shows
up in an embedder's logs. The table is additive-only and 77 does not collide with
anything in it.

## Consequences

Scripts branch on 77 for "it wants permission to download". Everything else stays inside
0–9; 130 remains SIGINT.
