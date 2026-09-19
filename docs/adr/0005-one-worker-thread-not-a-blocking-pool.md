# 0005 — One inference worker thread, not a `spawn_blocking` pool

Status: **accepted** (openjev-cli, 2026-09-19). Implements design 02 §3.7 against the
reality of `openjev-core` as built.

## Context

Design 02 assumes an async `Engine` in core with a worker pool and batch coalescing.
That does not exist: core ships a **synchronous `Session`**, and ADR 0003 records that
creating llama.cpp contexts concurrently returns an all-zero hidden state with no error,
no log and no panic.

## Decision

The server owns **exactly one** OS thread that holds the `Session` for the process
lifetime. Async handlers reach it through a bounded `tokio::mpsc` channel and wait on a
`oneshot`. No `spawn_blocking`, no pool, no second context — ever.

Batching is opportunistic: when the worker wakes it drains what is *already* queued up to
`max_batch` pairs and runs one forward pass. It never sleeps to accumulate a batch.

## Why not `spawn_blocking`

A blocking pool is a pool: tokio grows it under load, and the first time it did we would
either need a second `Session` (ADR 0003's corruption, silently) or a mutex around one
(a pool that serialises — the cost of a pool for the throughput of a thread). Naming one
thread is the honest version of what the hardware can do: one model, one device.

## Consequences

- Concurrency is admission control, not parallelism. The queue bounds it (64) and a full
  queue is `429 queue_full` + `Retry-After: 1`, never an unbounded wait.
- A request already inside a forward pass cannot be cancelled. A *queued* one is dropped
  the moment the client disconnects, which is the win that matters under load.
- If core later grows a real async `Engine`, this module is the only thing that changes;
  the HTTP contract does not.
