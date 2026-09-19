# 0004 — Weight acquisition talks to the HF resolve endpoint directly, rather than through `hf-hub`

Status: **accepted**. Amends `docs/design/01-inference-backend.md` §4, which names
`hf-hub` as the crate.

## Why the design said `hf-hub`

It already speaks HF's resolve URLs, revisions and auth, and it is the obvious default.
Nothing about that was wrong.

## Why this crate does not use it

§4 asks for five properties, and they are the actual requirements:

1. **Resumable** — `Range` on retry, because a 3 GB download that dies at 2.9 GB must not
   start over.
2. **Streamed integrity** — sha256 computed *during* the download and checked *before* the
   atomic rename, so a corrupt file is never momentarily readable as a model.
3. **Progress as a callback**, not a print. Core does not own stderr.
4. **Offline** that fails naming the exact path it wanted.
5. **A `.ok` stamp** so a 3 GB file is not re-hashed on every boot.

Implementing (1)+(2) together requires the raw byte stream: the hash must cover the whole
file, so a resumed download re-reads the bytes already on disk before appending. Wrapping
a downloader that owns its own retry, its own file placement and its own cache layout, in
order to reach underneath it for the byte stream, is more code and more coupling than
issuing the request. The endpoint itself is trivial and stable:

```
GET {HF_ENDPOINT}/{repo}/resolve/{revision}/{file}
```

plus `Authorization: Bearer $HF_TOKEN`. That is the whole protocol we need. `hf-hub` would
be carried for URL formatting.

## Decision

`hub.rs` uses `ureq` (rustls, no OpenSSL) against the resolve endpoint directly, and owns:
`.partial` + `Range` resume, streamed sha256, atomic rename after `fsync`, a `.ok` digest
stamp, `--verify` for a forced re-hash, offline mode, and a `ProgressFn` callback.

**Cache compatibility is kept deliberately**: `HF_HUB_CACHE`, then `HF_HOME/hub`, then
`OPENJEV_CACHE`, then `$XDG_CACHE_HOME/openjev`, then `~/.cache/openjev`. `HF_ENDPOINT`
and `HF_TOKEN` are honoured. A developer with a 200 GB HF cache does not get a second copy.
Paths are revision-addressed (`models/<repo>/<revision>/<file>`) so two pinned revisions
coexist rather than overwrite.

## What this costs

- We own HTTP error handling, redirects and retry semantics. `ureq` handles redirects;
  retry is currently one attempt plus resume-on-next-run, which is weaker than a
  purpose-built downloader and is the known gap.
- We do not get HF's xet/dedup transfer acceleration.
- If HF changes the resolve URL shape, we change one function instead of bumping a crate.

## Notes

One behaviour worth stating because it is easy to get wrong: if a `Range` request comes
back **200 instead of 206**, the server ignored the range, and the bytes already on disk
are worthless. Appending to them would produce a corrupt file that hashes wrong — or, if
the digest is unpinned, one that hashes not at all and is simply broken. The code treats
206-with-offset as the only resume case and truncates otherwise.
