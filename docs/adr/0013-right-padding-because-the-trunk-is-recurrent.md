# 0013 — right padding, because the trunk is recurrent

Status: **accepted** (openjev-core, 2026-09-20). **Reverses design 01 §5's left-padding
rule.**

## Context

Design 01 §5 rule 2 says: *"Any padded batch (candle later): left-pad with 248044, so the
last position is always real text and pooling is `h[:, -1, :]` with no index arithmetic."*
The registry default and `models.toml` both said `left`.

That is the right advice for a dense causal decoder, and it is wrong for this one.

24 of these 32 layers are gated-DeltaNet linear attention. A recurrent layer has no mask.
Attention can be told to ignore a pad token; a recurrence cannot — the state is advanced by
every token it is fed, so `n` leading pad tokens change the hidden state at **every real
position that follows**. Left padding would make a pair's answer depend on the length of
the longest other pair in its batch.

The reference right-pads and gathers:

```python
self.tok.padding_side = "right"      # the head pools the last non-pad token
last = enc["attention_mask"].sum(1) - 1
```

## Decision

`padding_side` defaults to **right**, and `models.toml` says `right`. Pooling carries an
explicit per-row index — which `EncodedInput.pool_index` already did, so no pooling code
changed.

## Consequences

The "no index arithmetic" convenience that motivated left padding is given up. It was
buying tidiness and paying in correctness.

This is currently moot on the shipping path: the llama.cpp backend forwards one sequence
per call and pads nothing at all, so no pad token has ever reached the trunk. It stops
being moot the moment any batching backend lands, which is exactly when a wrong default
would be hardest to find — hence fixing it now, with a real mixed-length batch test on
real weights (`tests/golden.rs::a_mixed_length_batch_agrees_with_the_same_pairs_alone`)
standing guard.
