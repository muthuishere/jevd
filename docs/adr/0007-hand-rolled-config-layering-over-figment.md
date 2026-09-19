# 0007 — Flat-key config layering, hand-rolled, instead of figment

Status: **accepted** (openjev-cli, 2026-09-19). Divergence from design 02 §0's crate
table, which names `figment 0.10` for layered config with provenance.

## Decision

Config is a `BTreeMap<String, Entry>` keyed by dotted path (`server.port`), merged
defaults → file → env → flags, with each entry remembering the layer that last wrote it.
~200 lines, no dependency.

## Why

The only thing we wanted from figment was **provenance**, and provenance is a by-product
of the flat map rather than a feature to extract from it. The flat map also makes the
other three config commands one lookup each: `config get server.port`, `config set`, and
`config show --sources` are the same data structure read three ways. With a nested
deserialised struct, each of those is a separate tree walk plus a schema to keep in sync.

The defaults table doubles as the schema: `config set` refuses a key that is not in it,
and the type of the default is the type the value must parse as. That is how
`OPENJEV_SERVER__PORT=twenty` becomes exit 3 with a message naming the key instead of a
silent zero.

## Consequences

- Only scalar and string-array values are expressible. No nested tables beyond one level.
  That is the whole config file today, and a deeper schema would be a reason to revisit.
- Unknown keys are preserved and ignored, never fatal — a newer config must not break an
  older binary mid-rollout — and `config validate` reports them as warnings.
- One asymmetry, documented once: the short env aliases (`OPENJEV_PORT`, …) beat their
  `OPENJEV_SERVER__PORT` long forms.
