# 02 — `openjev-cli`: CLI surface + HTTP API contract

Owner: API/CLI workstream. Scope: the `openjev` binary and the HTTP server it
runs. **Not in scope:** inference internals, model/device config *schema*, weight
download mechanics — those are `docs/design/01-inference-backend.md`, another
workstream. This file CONSUMES that schema and never redefines it.

Every decision below states the failure it is organised against. Where this file
and 01 disagree about anything inside `openjev-core`, 01 wins.

Binary: `openjev`   ·   Crate: `openjev-cli`   ·   Default port: **21131**

---

## 0. Verified facts (checked 2026-09-19, crates.io API — do not re-derive)

| crate | version | why |
|---|---|---|
| `clap` (+ `derive`, `env`) | **4.6.7** | 1.15B downloads. clap 5 is not stable. `derive` + `env` features are how flags/env unify. |
| `clap_complete` | 4.6.11 | shell completions, generated not hand-written |
| `axum` | **0.8.9** | 476M downloads, tower/tokio-native, `tower-http` gives us body limit + CORS + timeout + trace as middleware we do not write |
| `tokio` | 1.53.1 | non-negotiable base |
| `tower-http` | 0.7.1 | `RequestBodyLimitLayer`, `CorsLayer`, `TimeoutLayer`, `TraceLayer` |
| `tracing` / `tracing-subscriber` | 0.1.44 / 0.3.23 | structured logs, `RUST_LOG`, JSON layer |
| `figment` | 0.10.19 | layered config with **provenance** — it can say which layer won a key |
| `toml` | 1.1.6 | config file |
| `metrics-exporter-prometheus` | 0.18.3 | `/metrics` without hand-rolling exposition format |
| `fs4` | 1.1.0 | cross-platform advisory `flock` (macOS + Linux) for the pidfile |
| `indicatif` | 0.18.6 | download progress bar |
| `utoipa` | 5.5.0 | OpenAPI doc generated from the handlers, so it cannot drift |

**axum over actix-web.** Not performance — both are fast enough next to a 4B
forward pass. axum is `tower::Service`, so admission control, body limits,
timeouts and tracing are *layers we compose*, not framework-specific hooks. actix
brings its own actor runtime and a second mental model. Organised against: a
bespoke queue/limiter we have to debug ourselves.

Rust edition 2024, MSRV pinned to **1.95** (owner constraint). macOS arm64 +
Linux x86_64/aarch64 only.

---

## 1. Hard rules

1. **`openjev serve` with no arguments and no prior setup must work.** Download,
   load, serve. Any change that adds a prerequisite step is a regression.
2. **HTTP is the product.** herdr is one client among many. Nothing
   herdr-specific ships in the server — not a header, not an endpoint, not a
   config key.
3. **Bind `127.0.0.1` by default. A non-loopback bind without auth is refused at
   startup.** Not warned. Refused.
4. **The shape of stdout never depends on whether stdout is a TTY.** TTY controls
   colour, spinners and progress — nothing else. Organised against: the script
   that works in a terminal and silently emits a different shape under cron.
5. **A multi-GB download is never silent.** Neither on the CLI (progress bar on
   stderr) nor over HTTP (`/readyz` phase + `/v1/events` SSE).
6. Never truncate model input silently. Over-long input is a 422 unless the
   caller explicitly asked for truncation.
7. No secret literal in this repo. Tokens are generated at runtime, stored
   hashed, compared in constant time.

---

## 2. CLI surface

### 2.1 Tree

```
openjev
├── serve                         run the HTTP server (THE command)
├── predict  [PAIRS...]           one-shot NLI over premise/hypothesis pairs
├── rerank   <QUESTION> [OPTS...] one-shot rerank of options against a question
├── grade    <ANSWER> <REFERENCE> one-shot answer-vs-reference grade
├── model
│   ├── pull  [REF]               download weights, do not serve
│   ├── list                      what is in the cache, sizes, revisions
│   ├── rm    <REF>               delete a cached model
│   └── path  [REF]               print the on-disk path (scriptable)
├── status                        is a server running, where, which model
├── doctor                        environment + device + cache + connectivity
├── config
│   ├── path | show | get <K> | set <K> <V> | edit | validate
├── completions <SHELL>           bash|zsh|fish|powershell|elvish
└── version                       version, api version, build, device support
```

### 2.2 What is deliberately NOT here

- **No `openjev latents`.** It returns tensors. A CLI that prints a 2048-dim
  float array to a terminal is a papercut generator. Latents are HTTP-only, and
  off by default (§3.9). Organised against: scope creep into a numerics tool.
- **No `openjev daemon` / `openjev stop` / `openjev restart`.** Process
  supervision belongs to launchd/systemd (§4.6), which already do restart,
  logging and boot-start correctly. A hand-rolled supervisor is the bug we would
  spend a month on. `serve` runs in the foreground as the unit's main process.
- **No `openjev chat` / REPL.** This is a cross-encoder. There is no conversation.
- **No `openjev update` / self-update.** Package manager's job.
- **No `openjev bench`.** Folded into `doctor --bench`.
- **No plugin system.** The plugin surface *is* the HTTP API.

### 2.3 `openjev serve`

```
openjev serve [--host <IP>] [--port <N>] [--model <REF>] [--revision <REV>]
              [--device auto|cuda|metal|cpu] [--cache-dir <DIR>]
              [--token <TOKEN> | --token-file <F> | --no-auth]
              [--cors-origin <ORIGIN>]...  [--max-queue <N>] [--max-batch <N>]
              [--request-timeout <SECS>] [--shutdown-grace <SECS>]
              [--log-level <LVL>] [--log-format auto|text|json]
              [--state-file <PATH>] [--print-ready-json] [--fail-if-running]
              [--offline] [--preload | --no-preload]
```

- `--port 0` → kernel-assigned port; the real port lands in the state file and in
  the `--print-ready-json` line. This is how a client spawns its own server
  without fighting over a fixed port (§6.3).
- `--preload` (default **on**) blocks readiness until weights are resident.
  `--no-preload` binds the port immediately and loads lazily — `/readyz` stays
  503 until done. Default on because "the port is open but every request hangs
  for 90s" is a worse failure than a slow start.
- `--offline` forbids network access; a cache miss is a hard error, not a
  download. Organised against: an air-gapped box quietly reaching out.
- stdout: nothing, unless `--print-ready-json` (exactly one JSON line, then
  silence — see §6.3). All logs on **stderr**.
- Exit: 0 on clean SIGTERM/SIGINT drain; 0 also when another instance already
  holds the lock (idempotent start, §4.4) unless `--fail-if-running`.

### 2.4 One-shot commands: `predict` / `rerank` / `grade`

These exist so `openjev` composes in a pipeline without anyone standing up a
server or writing a client. They are the same code path as the HTTP handlers.

**`predict`** — the pair-oriented one.

```
openjev predict --premise "..." --hypothesis "..."     # single pair
openjev predict < pairs.ndjson                         # stdin, NDJSON
cat pairs.ndjson | openjev predict --json > out.ndjson
openjev predict --server http://127.0.0.1:21131        # use a running server
```

- **stdin contract:** NDJSON, one object per line,
  `{"premise":"...","hypothesis":"...","id":"optional"}`. One malformed line does
  not kill the run: it emits an error object on the same output line and sets
  exit 8 at the end. Organised against: line 90,000 of a batch job killing 89,999
  good results.
- **stdout contract:** one output line per input line, **in input order**, echoing
  `id` when given. Order is a promise — batching happens internally and must not
  reorder.
- `--server <URL>` sends to a running server instead of loading 4B params into a
  short-lived process. **If a server is discoverable (§6.1) and `--server` was
  not given, use it and say so on stderr.** Organised against: the user waiting
  90s for a one-line answer while a warm server sits idle on the same box.
  `--local` forces in-process.
- Human (TTY, no `--json`) output is an aligned table with the winning label
  bolded. `--json` gives NDJSON. The choice is the flag, never the TTY (rule 4).

**`rerank`**

```
openjev rerank "which is the fix?" --option "..." --option "..." [--top-k N]
openjev rerank "..." < options.txt        # one option per line, plain text
```
Output: ranked list `{rank, index, score, text}`. `--top-k` truncates.

**`grade`**

```
openjev grade --answer "..." --reference "..." [--threshold 0.5]
```
Output: `{label, scores{contradiction,entailment,neutral}, pass}` where `pass` is
`entailment >= threshold`. **`--threshold` also drives the exit code**: 0 = pass,
9 = fail. That makes `openjev grade` usable directly as a CI assertion, which is
the single highest-value non-server use of this model.

### 2.5 `status`, `doctor`, `model`, `config`

- `openjev status [--json]` — reads the state file (§4.5), probes `/readyz`,
  prints address, pid, uptime, model, revision, device, phase, queue depth,
  requests served. Exit 0 running-and-ready, 6 no server, 7 running-not-ready.
- `openjev doctor [--json] [--bench]` — one screen of ground truth: OS/arch, Rust
  build, detected accelerators and *why* a device was chosen, cache dir + free
  space vs model size, hub reachability, config file path and which layer won each
  key, port availability, running server. `--bench` runs 20 pairs and prints
  p50/p95. Exit 0 all-good, 1 any check failed. This is the first thing we ask
  for in a bug report, so it must be copy-pasteable and contain no secrets.
- `openjev model pull [REF]` — the escape hatch for "download now, on good wifi,
  before the demo". Same progress UI as first serve. `--revision`, `--force`.
- `openjev model list --json` → `[{ref, revision, size_bytes, path, last_used}]`.
- `openjev model rm <REF>` — refuses if a running server has it loaded, unless
  `--force`. `--dry-run` prints what would be freed.
- `openjev config show --sources` — prints the effective config **annotated with
  which layer set each key** (figment gives us provenance for free). Organised
  against the single most common support question: "why is it using CPU".

### 2.6 Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | generic failure |
| 2 | usage error (clap's own) |
| 3 | config invalid / unparseable |
| 4 | model unavailable (not cached + offline, or download failed) |
| 5 | requested device unavailable and `--device` was explicit |
| 6 | no server reachable |
| 7 | server reachable but not ready |
| 8 | bad input (malformed stdin, oversized field) |
| 9 | `grade` assertion failed (predicate, not an error) |
| 130 | SIGINT |

Codes are additive-only. Never renumber.

### 2.7 First-ever run — the exact transcript

This is the highest-stakes UI in the product. A user types one command and waits
several minutes. Every line below answers a question they are about to ask.

```
$ openjev serve
openjev 0.1.0  ·  api v1

  model    AlexWortega/openjev @ qwen3.5-4b-nli-v2
  device   metal (auto-detected: Apple M3 Max, 36 GB unified)
  cache    ~/.cache/openjev/models          (61.2 GB free)
  config   ~/.config/openjev/config.toml    (not found — using defaults)

This model is not cached. Downloading ~7.9 GB. One time only; later runs
start in seconds. Ctrl-C is safe — the download resumes.

  model-00001-of-00002.safetensors  ██████████░░░░░░░░  4.1/7.4 GB  58%  82 MB/s  ETA 00:40
  tokenizer.json                    ██████████████████  2.1/2.1 MB 100%

  ✓ downloaded 7.9 GB in 1m36s   verified sha256
  ✓ loaded onto metal in 11.2s   3.9 GB resident

  listening on http://127.0.0.1:21131   (loopback only, no auth required)
  ready  ·  try:  curl -s localhost:21131/v1/predict -d '{"pairs":[{"premise":"it rained","hypothesis":"the ground is wet"}]}'

  press Ctrl-C to stop
```

Non-negotiable properties of that transcript:

- **The size and the one-time-ness are stated before the wait starts**, not after.
- **Resumability is stated**, so Ctrl-C is not a 4 GB mistake.
- **The device choice is shown with its reason.** Silent CPU fallback on a
  CUDA box is the #1 "why is this slow" bug in every tool of this shape.
- **The free space is shown next to the download size.** ENOSPC at 94% is the
  cruellest failure available to us; `doctor`-grade checks run *before* the first
  byte and abort early with exit 4 if it will not fit.
- Progress goes to **stderr** and degrades to one line every 5s when stderr is
  not a TTY (organised against: 40 MB of `\r` spam in a systemd journal).
- The final line is a **copy-pasteable curl**, so success is self-evidently
  verifiable without opening docs.

---

## 3. HTTP API

Base path `/v1`. JSON in, JSON out, `Content-Type: application/json`. Every
response carries `X-OpenJEV-Api: 1` and `X-Request-Id`.

### 3.1 Endpoints

| method | path | auth | purpose |
|---|---|---|---|
| GET | `/healthz` | never | liveness. 200 whenever the process can answer. |
| GET | `/readyz` | never | readiness. 200 only when the model can serve a request. |
| GET | `/v1/info` | yes | server version, api version, capabilities, limits |
| GET | `/v1/model` | yes | loaded model ref, revision, device, dtype, labels |
| GET | `/v1/events` | yes | SSE: lifecycle phase + download/load progress |
| POST | `/v1/predict` | yes | pairs → label probabilities |
| POST | `/v1/rerank` | yes | question + options → ranked options |
| POST | `/v1/grade` | yes | answer + reference → label + pass |
| POST | `/v1/latents` | yes | opt-in, disabled by default (§3.9) |
| GET | `/metrics` | see §3.12 | Prometheus text exposition |
| GET | `/openapi.json` | yes | generated from handlers (`utoipa`) |

**`/healthz` and `/readyz` are different and the difference is the whole point.**
`/healthz` is "do not restart me" — 200 while downloading, while loading, while
draining. `/readyz` is "send me traffic" — 503 until weights are resident. A
supervisor that conflates them will kill the process 40 seconds into a 90-second
model load and loop forever. Both are unauthenticated: a probe that needs a
secret is a probe that will be misconfigured.

`GET /readyz` while not ready:

```json
{ "ready": false, "phase": "downloading",
  "detail": { "file": "model-00001-of-00002.safetensors",
              "bytes_done": 4402341888, "bytes_total": 7935819776,
              "eta_seconds": 40 },
  "since": "2026-09-19T08:14:02Z" }
```
`503` + `Retry-After: 5`. Phases: `starting` · `resolving` · `downloading` ·
`loading` · `ready` · `draining` · `failed`. `failed` returns 503 with a
terminal `error` object and **does not retry forever silently** — it retries with
backoff and says so.

### 3.2 `POST /v1/predict`

```json
{ "pairs": [ {"premise":"it rained","hypothesis":"the ground is wet","id":"a1"} ],
  "truncate": "error" }
```
`truncate`: `"error"` (default) | `"tail"`. Default is `error` because a silently
truncated premise produces a confident, wrong, unfalsifiable answer.

```json
{ "object": "predict",
  "model": "AlexWortega/openjev",
  "revision": "qwen3.5-4b-nli-v2",
  "results": [
    { "id": "a1", "index": 0,
      "label": "entailment",
      "scores": { "contradiction": 0.012, "entailment": 0.941, "neutral": 0.047 } }
  ],
  "usage": { "pairs": 1, "tokens": 14, "queue_ms": 3, "compute_ms": 71 } }
```
`results` is **always in request order**, `index` echoes the input position.
`scores` keys are the label set from `/v1/model` — never a bare array, because
an array's ordering becomes an undocumented tribal fact.

### 3.3 `POST /v1/rerank`

```json
{ "question": "which patch fixes the leak?",
  "options": ["...", "..."],
  "top_k": 5, "return_scores": true }
```
```json
{ "object": "rerank",
  "results": [ {"rank":0,"index":3,"score":0.87,"text":"..."} ],
  "usage": {...} }
```
`text` is echoed only when `return_documents: true` (default false) — the caller
already has the strings, and echoing 200 options doubles the payload.

### 3.4 `POST /v1/grade`

```json
{ "answer": "...", "reference": "...", "threshold": 0.5 }
```
```json
{ "object": "grade", "label": "entailment",
  "scores": {...}, "pass": true, "threshold": 0.5, "usage": {...} }
```

### 3.5 Error envelope

One shape, everywhere, including 500s:

```json
{ "error": {
    "code": "payload_too_large",
    "message": "request body is 4.2 MB, limit is 1.0 MB",
    "detail": { "limit_bytes": 1048576, "got_bytes": 4404019 },
    "request_id": "01JBX..." } }
```

| code | status | |
|---|---|---|
| `invalid_request` | 400 | malformed JSON / missing field |
| `unsupported_media_type` | 415 | |
| `unauthorized` | 401 | missing/bad bearer token |
| `not_found` | 404 | |
| `unprocessable` | 422 | input too long with `truncate:"error"`, empty options |
| `payload_too_large` | 413 | body limit or too many pairs |
| `queue_full` | 429 | admission rejected, `Retry-After` set |
| `timeout` | 504 | server-side deadline exceeded |
| `canceled` | 499 | client disconnected (logged, never sent) |
| `model_not_ready` | 503 | phase != ready, `Retry-After` set |
| `device_error` | 503 | OOM / accelerator fault; includes recovery state |
| `internal` | 500 | |

`message` is for humans and may change. **`code` is the contract** and is
additive-only. Clients branching on `message` are on their own.

### 3.6 Limits

| limit | default | config key |
|---|---|---|
| request body | 1 MiB | `server.max_body_bytes` |
| pairs per `/v1/predict` | 256 | `server.max_pairs` |
| options per `/v1/rerank` | 512 | `server.max_options` |
| chars per text field | 32768 | `server.max_field_chars` |
| queue depth | 64 | `server.max_queue` |
| inference micro-batch | 32 pairs | `server.max_batch` |
| request deadline | 60 s | `server.request_timeout_secs` |

Enforced by `tower_http::limit::RequestBodyLimitLayer` *before* the body is
buffered — a limit checked after reading 4 GB into memory is not a limit.
`/v1/info` publishes every one of these so a client can chunk correctly instead
of discovering the ceiling with a 413.

### 3.7 Concurrency, batching, admission

One model, one device, many callers. The whole design is: **queue explicitly,
reject early, never queue unboundedly.**

- **One inference worker** owns the model (`N` workers only if 01 says the backend
  is safely shardable; assume 1). Handlers are async, the worker is a dedicated
  blocking task fed by an `mpsc` channel.
- **Bounded queue, depth 64.** Full → `429 queue_full` + `Retry-After: 1`
  immediately. Organised against: the death spiral where every caller waits 40s,
  times out, retries, and the queue never drains.
- **FIFO. No priorities in v1.** Priority classes need a starvation story, and we
  do not have the traffic to justify one.
- **Opportunistic micro-batching, never a batching *delay*.** When the worker
  wakes, it drains whatever is already queued up to `max_batch` pairs and runs
  one forward pass. It never sleeps to accumulate a batch. Throughput of
  batching, zero added latency at low load. (Same pattern as `herdr-expose` B3.)
- **Timeouts.** Per-request deadline from `request_timeout_secs`, overridable
  downward per request by `X-OpenJEV-Timeout-Ms`, never upward.
- **Cancellation.** Client disconnect fires a `CancellationToken`: a *queued*
  request is dropped immediately (cheap, and the main win under load). A request
  already inside a forward pass **cannot be aborted** — we state that plainly
  rather than pretending. Its result is discarded.
- **Fairness against one fat caller:** the pair limit (256) bounds how much of a
  batch any one request can occupy. No per-client quotas in v1.

### 3.8 Streaming: no

`predict`/`rerank`/`grade` return a small fixed JSON after one forward pass.
There is nothing to stream — SSE here would add framing, partial-failure and
client-parsing surface to buy nothing. **The one exception is `/v1/events`**,
which streams *lifecycle*, not results: `phase`, download progress, load
progress, queue depth. That is what lets a dashboard or the herdr plugin show a
live download bar without polling `/readyz`. Heartbeat comment every 15 s so
proxies do not reap the connection.

### 3.9 `/v1/latents` — opt-in, off by default

Returns raw hidden-state vectors. Disabled unless `server.enable_latents = true`.
Reasons: response bodies jump from ~200 B to megabytes (blowing the body limit
assumptions in both directions), the tensor layout is an inference-internal
detail owned by 01 and will change, and it is the endpoint most likely to be
mistaken for a stable embeddings API. Behind a flag it is a research tool; on by
default it is a compatibility obligation we did not agree to.

### 3.10 No OpenAI-compatible shim

**Argued and rejected for v1.** The OpenAI surface has no honest slot for this
model. `/v1/chat/completions` would mean inventing a text rendering of a
3-way probability distribution; `/v1/embeddings` is a different computation
entirely (this is a *cross*-encoder — it has no per-text embedding); the closest
real fit is Cohere's rerank API, not OpenAI's.

The failure a shim is organised against is "my existing client works
out of the box". The failure a shim *causes* is worse: a caller points an OpenAI
SDK at us, gets a plausible-looking response, and silently consumes a mistranslated
score in production. A 404 is a better error than a wrong number.

What we ship instead: a generated `/openapi.json`, so clients are generated, not
hand-written. Revisit only with a concrete consumer that cannot be changed —
and then as `/compat/cohere/v1/rerank`, which at least maps honestly.

### 3.11 Auth and CORS

- **Loopback bind (default): no auth.** A token on `127.0.0.1` protects against
  nothing a local attacker cannot already do, and it makes the headline
  `openjev serve` one command instead of three.
- **Non-loopback bind: a token is mandatory.** `--host 0.0.0.0` without
  `--token`/`--token-file`/`config.auth.token` **refuses to start**, with an
  error that prints a freshly generated token to copy. Not a warning — the
  warning is the thing nobody reads before this ends up on a public IP. `--no-auth`
  exists to override deliberately and logs `WARN` on every startup, forever.
- Bearer only: `Authorization: Bearer <token>`. **Stored SHA-256-hashed** in the
  state dir (0600), never in the config file, never logged, never in
  `/v1/info`. `subtle::ConstantTimeCompare` on every comparison.
- 401s are rate-limited per source address.
- **CORS off by default.** `server.cors_origins` is an explicit allowlist;
  `"*"` is *refused* whenever auth is enabled (a wildcard plus a bearer token is
  a token-exfiltration invitation). Origin is validated before any CORS header is
  echoed. `X-Frame-Options: DENY`, `frame-ancestors 'none'`.

### 3.12 `/metrics`

Prometheus text exposition (`metrics-exporter-prometheus`). Unauthenticated on a
loopback bind; token-gated otherwise; `server.metrics = false` removes it.
Minimum set: `openjev_requests_total{endpoint,code}`,
`openjev_request_duration_seconds{endpoint}` (histogram),
`openjev_queue_depth`, `openjev_queue_wait_seconds`, `openjev_batch_size`,
`openjev_inference_duration_seconds`, `openjev_pairs_total`,
`openjev_model_load_seconds`, `openjev_ready` (0/1), `openjev_build_info`.
No text content in labels, ever — unbounded cardinality and a data leak in one
move.

### 3.13 Versioning

Path-versioned `/v1`. Inside v1 we are **additive only**: new endpoints, new
optional request fields, new response fields, new `error.code` values. A client
must ignore unknown response fields — stated here so it is a contract, not a
hope. Breaking changes mount `/v2` **alongside** `/v1`, and `/v1` survives at
least one minor release with a `Deprecation` header. `/v1/info` returns
`{"api_version": 1, "server_version": "0.3.1", ...}`; clients gate on
`api_version`, never on `server_version`.

---

## 4. Lifecycle

### 4.1 Startup sequence

1. Parse flags (clap) → merge config layers (figment, §5) → validate. Bad config
   exits **3** before anything expensive.
2. Resolve device. Log the choice **and the reason**.
3. Acquire the pidfile flock (§4.4). Held by another instance → §4.4.
4. Preflight: cache dir writable, free space ≥ model size × 1.1, hub reachable
   (unless `--offline`). Fail **4** here, not at 94% of a download.
5. Bind the listener. Bind failure exits 1 with the port and the owning pid if we
   can determine it.
6. Write the state file (§4.5) — **after** bind, so `bound_addr` is real.
7. Phase `downloading` → `loading` → `ready`. `/healthz` 200 throughout;
   `/readyz` 503 until `ready`.
8. On `ready`: log the address; emit the `--print-ready-json` line if asked.

Weight download and load are delegated to `openjev-core`; this workstream
consumes a progress callback and renders it (bar on a TTY, one INFO line per 5 s
or per 10 % otherwise, `phase` on `/readyz`, SSE on `/v1/events`).

### 4.2 Logged at startup, exactly

`INFO` (always): version + api version; model ref + revision; resolved device +
reason; cache dir; **config file path and whether it was found**; bind address;
auth mode (`none (loopback)` / `bearer`); limits (queue, batch, body, timeout);
each phase transition with its duration; final `ready` with total startup time.
`WARN`: non-loopback bind, `--no-auth`, `cors_origins` non-empty, device fallback
from an explicit request, low disk. **Never logged:** token values, any request
or response text body. Request logs carry `request_id`, endpoint, status,
pair count, queue_ms, compute_ms — counts and timings, never content.

### 4.3 Graceful shutdown

SIGTERM/SIGINT → phase `draining`: stop accepting connections, `/readyz` → 503
(load balancers drain first), in-flight requests finish, queued requests finish
within `shutdown_grace_secs` (default 20) and are otherwise failed with
`code: "shutting_down"`. Then unlink the state file, release the flock, exit 0.
**A second signal exits 130 immediately.** Organised against: the "graceful"
shutdown you cannot get out of.

### 4.4 A second `openjev serve` — flock on a pidfile (house pattern)

`$STATE_DIR/openjev.pid`, exclusive non-blocking `flock` (`fs4`). flock, not a
pid-in-a-file check: the kernel releases it on crash, so there is no stale-lock
recovery path to get wrong, and no pid-reuse race.

- **Lock free** → we are the server. Write pid, proceed.
- **Lock held** → another live instance owns it. Read its state file, print
  `openjev is already running on http://127.0.0.1:21131 (pid 40122)` and
  **exit 0**. A repeated start is harmless and idempotent — that is exactly what
  a launchd/systemd `KeepAlive` unit plus an impatient human will produce.
  `--fail-if-running` exits 1 instead, for scripts that need to know.
- **Lock free but port in use** → somebody else's process. Exit 1, name the port,
  suggest `--port 0`. We never kill a process we do not own.

### 4.5 State file — the discovery contract

`$STATE_DIR/server.json`, mode 0600, written after bind, unlinked on clean exit.
`$STATE_DIR` = `$XDG_STATE_HOME/openjev` → `~/.local/state/openjev` (Linux),
`~/Library/Application Support/openjev` (macOS). Overridable by `--state-file`.

```json
{ "schema": 1, "pid": 40122,
  "url": "http://127.0.0.1:21131",
  "bound_addr": "127.0.0.1:21131",
  "api_version": 1, "server_version": "0.3.1",
  "model": "AlexWortega/openjev", "revision": "qwen3.5-4b-nli-v2",
  "device": "metal", "auth": "none",
  "started_at": "2026-09-19T08:14:02Z" }
```

**`auth` names the mode, never the token.** A stale file (process dead, flock
free) is treated as absent; consumers must probe `/healthz` before trusting it.

### 4.6 systemd / launchd

Shipped units, not documentation to retype.

- Linux: `systemd --user`, `Type=exec`, `ExecStart=openjev serve`,
  `Restart=on-failure`, `RestartSec=5`, `TimeoutStartSec=0` (a 4 GB download on
  hotel wifi will exceed any finite value you pick), `WatchdogSec` unset,
  `StandardOutput=journal`, `Environment=OPENJEV_LOG_FORMAT=json`. Enable
  `loginctl enable-linger` for boot-start, matching the house pattern on prod.
- macOS: LaunchAgent, `KeepAlive.SuccessfulExit=false`, `RunAtLoad`, stderr to
  `~/Library/Logs/openjev/openjev.log`. **No `launchd` throttle below 10 s** —
  the flock makes a fast respawn harmless but not free.
- Both: an optional `openjev-download.service`/one-shot `ExecStartPre` calling
  `openjev model pull` so the first *service* start is not the first download.

### 4.7 Logs

`tracing` + `tracing-subscriber`. `--log-format auto|text|json`; `auto` = text
when stderr is a TTY, JSON lines otherwise. This is the **one** place TTY changes
a shape, and it is legitimate: logs are diagnostics on stderr, not data on
stdout (rule 4 governs stdout only). Level: `--log-level` > `OPENJEV_LOG` >
`RUST_LOG` > `info`. Level is hot-reloadable via SIGHUP.

---

## 5. Config

### 5.1 Precedence

`flags` > `env (OPENJEV_*)` > `config file` > `built-in defaults`.
figment, one `Figment` assembled once, with provenance retained so
`config show --sources` can attribute every key.

Env mapping is mechanical: `server.port` → `OPENJEV_SERVER__PORT`
(double underscore = nesting). Plus documented short aliases for the five people
actually use: `OPENJEV_PORT`, `OPENJEV_HOST`, `OPENJEV_MODEL`, `OPENJEV_DEVICE`,
`OPENJEV_CACHE_DIR`. Aliases win over their long forms; that asymmetry is
documented once here and nowhere else.

### 5.2 File

`~/.config/openjev/config.toml` **unconditionally**. Not `$XDG_CONFIG_HOME`-
chased, not app-dir-relative, not env-selected beyond a single explicit
`--config <PATH>`. One tool, one config path; the alternative is the bug
`herdr-expose` B6 documents — two different configs depending on how you started
it.

**Absent is normal.** No file is created on first run. `openjev config edit`
creates it, fully commented, from defaults.

```toml
# ~/.config/openjev/config.toml

[server]
host = "127.0.0.1"          # non-loopback requires auth.token or it refuses to start
port = 21131
max_queue = 64
max_batch = 32
max_body_bytes = 1048576
max_pairs = 256
request_timeout_secs = 60
shutdown_grace_secs = 20
cors_origins = []           # explicit allowlist; "*" refused when auth is on
metrics = true
enable_latents = false

[auth]
token = ""                  # blank + loopback => no auth. Prefer token_file.
token_file = ""             # path to a 0600 file; preferred over inline

[log]
level = "info"
format = "auto"             # auto | text | json

# [model] and [device] are defined by docs/design/01-inference-backend.md.
# This workstream reads them and does not define their keys.
[model]
[device]
```

Unknown keys are **preserved and ignored**, never an error — a newer config must
not break an older binary during a staged rollout. `config validate` reports them
as warnings.

### 5.3 Hot reload (SIGHUP)

Reloadable, because nothing touches loaded weights: `log.level`,
`log.format`, `server.cors_origins`, `server.max_queue`, `server.max_batch`,
`server.request_timeout_secs`, `server.max_body_bytes`, `auth.token`/`token_file`
(existing connections keep their authorisation; new requests use the new set).

**Not reloadable — refused with a clear message naming the key, not silently
ignored:** `server.host`, `server.port`, `[model]`, `[device]`,
`server.enable_latents`. Changing those is a restart. Silently-ignored reloads
are how you spend an afternoon wondering why the device did not change.

### 5.4 `openjev config` subcommands

`path` (print the file path, whether or not it exists) · `show [--json]
[--effective] [--sources]` · `get <KEY>` (single value, bare on stdout,
scriptable) · `set <KEY> <VALUE>` (validates, writes preserving comments,
refuses to write a token inline unless `--allow-inline-token`) · `edit`
(`$EDITOR`, validate before saving, never leave a broken file behind) ·
`validate [--file F]` (exit 0/3).

---

## 6. Client contract (what `herdr-jev` and every other client can rely on)

The herdr plugin is **a client of the public HTTP API and nothing more**. If it
needs something the API does not offer, the API gains it — for everyone.

### 6.1 Discovery — ordered, and it stops at the first hit

1. Explicit URL from the client's own config.
2. `OPENJEV_URL` environment variable.
3. **State file** `~/.local/state/openjev/server.json` (§4.5) → `url`. This is
   the answer to "fixed port or not": **the port is not fixed, the state file
   is.** It survives `--port 0`, a user moving the port, and two projects
   wanting different ports.
4. Probe `http://127.0.0.1:21131` (the documented default).
5. Nothing found → spawn its own (§6.3) or report clearly. Never guess further.

At every step, confirm with `GET /healthz` before believing it. A state file can
outlive its process.

### 6.2 Version negotiation

`GET /v1/info` → `{ "api_version": 1, "server_version": "...", "capabilities":
["predict","rerank","grade","events","metrics"], "limits": {...},
"model": {...} }`.

- Client requires `api_version` **equal** to the major it was built for.
  Mismatch → refuse with a real message naming both versions. No best-effort
  degradation across a major: a client guessing at v2 semantics is worse than a
  client that stops.
- Within a major: **feature-detect via `capabilities`, never by version
  comparison.** `enable_latents=false` is a capability difference at the same
  version, so version arithmetic cannot express it.
- Read `limits` and chunk to them. Do not hardcode 256.
- Ignore unknown fields (§3.13).

### 6.3 Point-at-one or spawn-your-own — both, identically

A client must be able to do either without changing any other code.

- **Attach:** discover (§6.1), use it. Never take ownership; never send it
  SIGTERM; never delete its state file.
- **Spawn:**
  ```
  openjev serve --port 0 --state-file <client-private-path> --print-ready-json
  ```
  The server writes **exactly one line of JSON to stdout** when it is ready —
  `{"url":"http://127.0.0.1:53211","pid":40122,"api_version":1,"model":"...","revision":"..."}`
  — and nothing else on stdout, ever. The client blocks on that line with its own
  timeout (default generous: a first-run download is minutes, and it should
  surface `/v1/events` progress rather than time out). Everything else is stderr.
  `--port 0` + a private state file means a spawned server never collides with a
  user's own.
- A spawned server is an ordinary server. Same API, same auth rules, same
  everything. **There is no embedded, client-private mode**, because that is how
  a second, divergent, undocumented API gets built.
- Ownership: whoever spawned it kills it (SIGTERM, then the grace window). The
  flock (§4.4) means a client racing to spawn one while another already runs
  simply gets exit 0 and finds it via §6.1.

---

## 7. Definition of done

- `openjev serve` on a clean machine with no config downloads, loads, serves, and
  prints §2.7's transcript. `curl /v1/predict` returns the documented shape.
- `/readyz` is 503 with a live `phase` and `bytes_done` during the download, and
  `/healthz` is 200 throughout.
- Two concurrent `openjev serve` → one server, the second exits 0 quietly.
- `--host 0.0.0.0` with no token **refuses to start**.
- `echo '{"premise":"a","hypothesis":"b"}' | openjev predict --json` works with
  no server, and reuses a running server when one exists.
- `openjev grade --answer x --reference y --threshold 0.9` exits 9 on failure.
- `openjev doctor` output is copy-pasteable into an issue and contains no secret.
- Queue saturation returns 429 with `Retry-After`, never an unbounded wait.
- SIGTERM drains in-flight work and unlinks the state file; SIGTERM twice exits.
- `/openapi.json` generates a working client with no access to this source.

## 8. ADRs to write

| id | decision |
|---|---|
| `0001-axum-over-actix.md` | tower layers as the concurrency/limits substrate |
| `0002-no-openai-compat-shim.md` | §3.10 — a wrong number beats a 404 is false |
| `0003-loopback-default-auth-mandatory-off-loopback.md` | refuse, do not warn |
| `0004-flock-pidfile-idempotent-start.md` | house pattern, no stale-lock path |
| `0005-state-file-discovery.md` | the port is not the contract, the file is |
| `0006-health-vs-ready-split.md` | the multi-GB load is the reason |
| `0007-bounded-queue-opportunistic-batching.md` | 429 over unbounded latency |
| `0008-tty-never-changes-stdout-shape.md` | cron-vs-terminal divergence |
