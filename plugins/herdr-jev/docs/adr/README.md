# Architecture Decision Records

Each ADR records one decision, why it was made, and what it costs. Format:
Context / Decision / Consequences / Status. A decision that changes gets a new
ADR, or an explicit "supersedes" note, rather than a quiet edit.

`../../SPEC.md` is the binding build contract and wins over anything here.
The engine's own decisions live in `../../../../docs/adr/` — this directory is
the plugin's, and the two are deliberately separate: nothing in openjev knows
Herdr exists.

| # | Title | Status |
|---|---|---|
| [0001](0001-three-way-decision-answer-route-pass.md) | The decision is three-way: answer, route, pass through | Accepted |
| [0002](0002-port-the-asymmetric-confidence-policy.md) | Port the asymmetric-confidence policy rather than invent one | Accepted |
| [0003](0003-shape-before-confidence.md) | Shape is checked before confidence, and from the text alone | Accepted |
| [0004](0004-the-answer-bar-is-separate-and-higher.md) | The answer bar is separate from, and above, the routing bars | Accepted |
| [0005](0005-normalise-scores-into-a-confidence.md) | Normalise scores; a raw entailment score is not a confidence | Accepted |
| [0006](0006-short-circuit-off-by-default-with-shadow.md) | The short circuit ships off, with a shadow mode | Accepted |
| [0007](0007-route-at-agent-start-not-mid-session.md) | Route at agent start, because Herdr offers nothing else | Accepted |
