# 0011 — the template trims both fields, because the reference does

Status: **accepted** (openjev-core, 2026-09-20). **Reverses a decision made without the
reference in hand.**

## Context

`tokenize::render` substituted `{premise}` and `{hypothesis}` verbatim, and a test —
`template_does_not_trim_or_normalise` — asserted that it must. The stated reason was
"whitespace is the caller's problem; we must not silently change the input", which is a
good instinct applied to the wrong artefact.

The reference implementation does not agree. `modeling_openjev.py` renders

```python
texts = [self.template.format(premise=p.strip(), hypothesis=h.strip()) for p, h in pairs]
```

The `.strip()` is part of the model's input contract, not a convenience. The checkpoint
was trained and evaluated on trimmed fields.

## Decision

`render` trims both fields. Interior whitespace is untouched; only the ends.

## Consequences

An input with a leading newline on the hypothesis now produces the same token sequence —
and therefore the same label distribution — as the reference produces for it. Before this
change it produced a different one, silently, with no error and no warning: the classic
shape of a cross-encoder that is subtly not the model it claims to be.

The golden fixture carries two whitespace-bearing pairs for exactly this reason, so a
regression fails a test rather than degrading accuracy by an amount nobody measures.

We are not "not changing the input". We were changing it — by declining to apply a
transformation the model expects. There is no neutral option here; there is only matching
the reference or being a different model.
