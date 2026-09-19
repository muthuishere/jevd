//! Weight acquisition: download-on-first-run, resumable, integrity-checked,
//! revision-pinned, offline-capable.
//!
//! Progress is a **callback**, never a print. Core does not own stderr — the CLI renders
//! a bar, a herdr plugin renders a log line every 10 %, a test renders nothing. A library
//! that prints is a library you cannot embed.
//!
//! Organised against three specific failures:
//!   * a truncated 3 GB download that is readable as a model (→ `.partial` + atomic rename),
//!   * a corrupt download that is kept with a warning (→ mismatch deletes and fails),
//!   * re-hashing 3 GB on every boot (→ a `.ok` stamp carrying the digest).

use crate::error::{JevError, Result};
use crate::registry::HubFile;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const ENDPOINT: &str = "https://huggingface.co";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    Start { total: Option<u64> },
    Advance { done: u64, total: Option<u64> },
    Done { bytes: u64 },
}

/// What the caller renders. `&dyn` rather than a generic so `Hub` stays object-safe and
/// the CLI can swap renderers at runtime.
pub type ProgressFn<'a> = &'a mut dyn FnMut(&str, Progress);

fn noop(_: &str, _: Progress) {}

#[derive(Debug, Clone)]
pub struct Hub {
    root: PathBuf,
    offline: bool,
    token: Option<String>,
    endpoint: String,
}

impl Hub {
    /// Respect `HF_HOME` / `HF_HUB_CACHE` when set — a developer with a 200 GB HF cache
    /// should not get a second copy. Unset ⇒ our own dir.
    pub fn from_env() -> Self {
        let root = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
            .or_else(|| std::env::var_os("OPENJEV_CACHE").map(PathBuf::from))
            .or_else(|| {
                std::env::var_os("XDG_CACHE_HOME").map(|x| PathBuf::from(x).join("openjev"))
            })
            // ~/.cache/openjev, not ~/Library/Caches — CLI convention beats Apple
            // convention, same reasoning as the config dir.
            .or_else(|| dirs::home_dir().map(|h| h.join(".cache").join("openjev")))
            .unwrap_or_else(|| PathBuf::from(".openjev-cache"));

        let offline = truthy("OPENJEV_OFFLINE") || truthy("HF_HUB_OFFLINE");

        Self {
            root,
            offline,
            token: std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
            endpoint: std::env::var("HF_ENDPOINT").unwrap_or_else(|_| ENDPOINT.to_string()),
        }
    }

    pub fn with_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.root = root.into();
        self
    }

    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Revision-addressed, so two pinned revisions coexist rather than overwrite.
    pub fn path_for(&self, f: &HubFile, default_revision: &str) -> PathBuf {
        let rev = f.revision.as_deref().unwrap_or(default_revision);
        self.root
            .join("models")
            .join(f.repo.replace('/', "--"))
            .join(rev)
            .join(&f.file)
    }

    fn url_for(&self, f: &HubFile, default_revision: &str) -> String {
        let rev = f.revision.as_deref().unwrap_or(default_revision);
        format!("{}/{}/resolve/{}/{}", self.endpoint, f.repo, rev, f.file)
    }

    /// Cached path, downloading if necessary.
    pub fn get(&self, f: &HubFile, default_revision: &str) -> Result<PathBuf> {
        self.get_with_progress(f, default_revision, &mut noop)
    }

    pub fn get_with_progress(
        &self,
        f: &HubFile,
        default_revision: &str,
        progress: ProgressFn<'_>,
    ) -> Result<PathBuf> {
        let dest = self.path_for(f, default_revision);
        if self.is_complete(&dest, f)? {
            return Ok(dest);
        }
        if self.offline {
            return Err(JevError::Offline {
                what: format!("{}/{}", f.repo, f.file),
                path: dest,
            });
        }
        self.download(&self.url_for(f, default_revision), &dest, f, progress)?;
        Ok(dest)
    }

    /// Present and trusted. Trust is the `.ok` stamp carrying the digest, not a re-hash:
    /// re-hashing 3 GB every boot is a tax nobody pays twice.
    pub fn is_complete(&self, dest: &Path, f: &HubFile) -> Result<bool> {
        if !dest.is_file() {
            return Ok(false);
        }
        if f.sha256.is_empty() {
            return Ok(true);
        }
        let stamp = stamp_path(dest);
        match std::fs::read_to_string(&stamp) {
            Ok(s) if s.trim().eq_ignore_ascii_case(&f.sha256) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Force a full re-hash of a cached file (`--verify`).
    pub fn verify(&self, dest: &Path, f: &HubFile) -> Result<()> {
        if f.sha256.is_empty() {
            return Ok(());
        }
        let mut file = std::fs::File::open(dest).map_err(|e| JevError::io(dest, e))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = file.read(&mut buf).map_err(|e| JevError::io(dest, e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let actual = hex(&hasher.finalize());
        if !actual.eq_ignore_ascii_case(&f.sha256) {
            return Err(JevError::Integrity {
                file: dest.to_path_buf(),
                expected: f.sha256.clone(),
                actual,
            });
        }
        write_stamp(dest, &actual)?;
        Ok(())
    }

    fn download(
        &self,
        url: &str,
        dest: &Path,
        f: &HubFile,
        progress: ProgressFn<'_>,
    ) -> Result<()> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| JevError::io(parent, e))?;
        }
        let partial = partial_path(dest);

        // Resume. A 3 GB download that dies at 2.9 GB must not start over.
        let have = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

        let mut req = ureq::get(url);
        if let Some(t) = &self.token {
            req = req.header("Authorization", &format!("Bearer {t}"));
        }
        if have > 0 {
            req = req.header("Range", &format!("bytes={have}-"));
        }

        let resp = req.call().map_err(|e| JevError::Download {
            url: url.to_string(),
            message: e.to_string(),
        })?;

        let status = resp.status().as_u16();
        // 206 means the server honoured our Range; 200 means it did not, so the bytes we
        // already have are worthless and appending them would corrupt the file silently.
        let resuming = status == 206 && have > 0;
        if status != 200 && status != 206 {
            return Err(JevError::Download {
                url: url.to_string(),
                message: format!("HTTP {status}"),
            });
        }

        let remaining: Option<u64> = resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok());
        let total = match (remaining, resuming) {
            (Some(r), true) => Some(r + have),
            (Some(r), false) => Some(r),
            (None, _) => f.size_bytes,
        };

        let mut file = if resuming {
            std::fs::OpenOptions::new()
                .append(true)
                .open(&partial)
                .map_err(|e| JevError::io(&partial, e))?
        } else {
            std::fs::File::create(&partial).map_err(|e| JevError::io(&partial, e))?
        };

        // The hash must cover the whole file, so a resumed download re-reads what is
        // already on disk rather than hashing only the tail.
        let mut hasher = Sha256::new();
        if resuming {
            let mut pre = std::fs::File::open(&partial).map_err(|e| JevError::io(&partial, e))?;
            let mut buf = vec![0u8; 1 << 20];
            let mut left = have;
            while left > 0 {
                let want = buf.len().min(left as usize);
                let n = pre
                    .read(&mut buf[..want])
                    .map_err(|e| JevError::io(&partial, e))?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                left -= n as u64;
            }
        }

        progress(&f.file, Progress::Start { total });
        let mut done = if resuming { have } else { 0 };
        let mut body = resp.into_body().into_reader();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = body.read(&mut buf).map_err(|e| JevError::Download {
                url: url.to_string(),
                message: e.to_string(),
            })?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            file.write_all(&buf[..n])
                .map_err(|e| JevError::io(&partial, e))?;
            done += n as u64;
            progress(&f.file, Progress::Advance { done, total });
        }
        file.flush().map_err(|e| JevError::io(&partial, e))?;
        // fsync before the rename: the rename is what makes the file visible as a model,
        // and it must not become visible before its bytes are durable.
        file.sync_all().map_err(|e| JevError::io(&partial, e))?;
        drop(file);

        let actual = hex(&hasher.finalize());
        if !f.sha256.is_empty() && !actual.eq_ignore_ascii_case(&f.sha256) {
            // Delete and fail. Never keep-and-warn: a corrupt GGUF that loads is worse
            // than no GGUF.
            let _ = std::fs::remove_file(&partial);
            return Err(JevError::Integrity {
                file: dest.to_path_buf(),
                expected: f.sha256.clone(),
                actual,
            });
        }

        std::fs::rename(&partial, dest).map_err(|e| JevError::io(dest, e))?;
        if !f.sha256.is_empty() {
            write_stamp(dest, &actual)?;
        }
        progress(&f.file, Progress::Done { bytes: done });
        Ok(())
    }
}

fn partial_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".partial");
    PathBuf::from(s)
}

fn stamp_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".ok");
    PathBuf::from(s)
}

fn write_stamp(dest: &Path, digest: &str) -> Result<()> {
    let p = stamp_path(dest);
    std::fs::write(&p, digest).map_err(|e| JevError::io(&p, e))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn truthy(var: &str) -> bool {
    matches!(
        std::env::var(var).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "openjev-hub-test-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&p).expect("tmpdir");
        p
    }

    fn file(sha: &str) -> HubFile {
        HubFile {
            repo: "acme/weights".into(),
            file: "sub/model.gguf".into(),
            revision: None,
            sha256: sha.into(),
            size_bytes: None,
        }
    }

    #[test]
    fn cache_path_is_revision_addressed_so_two_pins_coexist() {
        let hub = Hub::from_env().with_root("/cache");
        let a = hub.path_for(&file(""), "aaa");
        let mut f = file("");
        f.revision = Some("bbb".into());
        let b = hub.path_for(&f, "aaa");
        assert_ne!(a, b);
        assert!(a.ends_with("models/acme--weights/aaa/sub/model.gguf"));
        assert!(b.ends_with("models/acme--weights/bbb/sub/model.gguf"));
    }

    #[test]
    fn offline_miss_names_the_exact_path_it_wanted() {
        let hub = Hub::from_env().with_root(tmp()).offline(true);
        let err = hub.get(&file(""), "rev").unwrap_err();
        match &err {
            JevError::Offline { path, .. } => assert!(path.ends_with("model.gguf")),
            other => panic!("wrong variant: {other}"),
        }
        assert!(err.to_string().contains("not in the cache"));
    }

    #[test]
    fn an_unpinned_cached_file_is_trusted_and_a_pinned_one_needs_its_stamp() {
        let root = tmp();
        let hub = Hub::from_env().with_root(&root).offline(true);
        let f = file("");
        let dest = hub.path_for(&f, "rev");
        std::fs::create_dir_all(dest.parent().expect("parent")).expect("mkdir");
        std::fs::write(&dest, b"hello").expect("write");

        assert!(
            hub.is_complete(&dest, &f).unwrap(),
            "unpinned: present is enough"
        );

        // sha256("hello")
        let sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        let pinned = file(sha);
        assert!(
            !hub.is_complete(&dest, &pinned).unwrap(),
            "pinned without a stamp must not be trusted"
        );

        hub.verify(&dest, &pinned).expect("verify");
        assert!(
            hub.is_complete(&dest, &pinned).unwrap(),
            "stamp written by verify"
        );
    }

    #[test]
    fn a_digest_mismatch_is_an_error_not_a_warning() {
        let root = tmp();
        let hub = Hub::from_env().with_root(&root);
        let f = file("0000000000000000000000000000000000000000000000000000000000000000");
        let dest = hub.path_for(&f, "rev");
        std::fs::create_dir_all(dest.parent().expect("parent")).expect("mkdir");
        std::fs::write(&dest, b"hello").expect("write");
        let e = hub.verify(&dest, &f).unwrap_err();
        assert!(matches!(e, JevError::Integrity { .. }), "{e}");
    }

    #[test]
    fn partial_and_stamp_paths_sit_beside_the_file() {
        let p = Path::new("/c/model.gguf");
        assert_eq!(partial_path(p), Path::new("/c/model.gguf.partial"));
        assert_eq!(stamp_path(p), Path::new("/c/model.gguf.ok"));
    }
}
