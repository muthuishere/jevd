//! The discovery contract: a pidfile flock and a state file (design 02 §4.4, §4.5).
//!
//! flock, not a pid-in-a-file check: the kernel drops it when the process dies, so there
//! is no stale-lock recovery path to get wrong and no pid-reuse race. The state file is
//! advisory — it can outlive its process, so every consumer probes `/healthz` before
//! believing it.

use crate::exit::{CliError, CliResult};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerState {
    pub schema: u32,
    pub pid: u32,
    pub url: String,
    pub bound_addr: String,
    pub api_version: u32,
    pub server_version: String,
    pub model: String,
    pub revision: String,
    pub device: String,
    /// The mode — `none` or `bearer`. **Never the token.**
    pub auth: String,
    pub started_at: String,
}

pub const SCHEMA: u32 = 1;
pub const API_VERSION: u32 = 1;

/// `$XDG_STATE_HOME/openjev`, else `~/.local/state/openjev`. macOS gets the same path as
/// Linux on purpose: one tool, one location, so a script does not need an OS branch.
pub fn state_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_STATE_HOME") {
        return PathBuf::from(x).join("openjev");
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local")
        .join("state")
        .join("openjev")
}

pub fn default_state_file() -> PathBuf {
    state_dir().join("server.json")
}

/// The pidfile lives beside whatever state file we were given, so a client-private
/// `--state-file` gets a client-private lock and does not collide with the user's server.
pub fn pid_file_for(state_file: &Path) -> PathBuf {
    state_file.with_file_name("openjev.pid")
}

pub fn write_state(path: &Path, state: &ServerState) -> CliResult<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let body = serde_json::to_string_pretty(state).map_err(|e| CliError::other(e.to_string()))?;
    write_private(path, &body)
}

fn write_private(path: &Path, body: &str) -> CliResult<()> {
    use std::io::Write;
    let mut opts = OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    f.write_all(body.as_bytes())?;
    Ok(())
}

pub fn read_state(path: &Path) -> Option<ServerState> {
    let body = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&body).ok()
}

pub fn remove_state(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Held for the process lifetime. Dropping it releases the lock.
pub struct PidLock {
    _file: File,
    path: PathBuf,
}

pub enum Lock {
    Acquired(PidLock),
    /// Someone else holds it. Not an error: a repeated start is idempotent (§4.4).
    Held,
}

pub fn acquire(pid_file: &Path) -> CliResult<Lock> {
    if let Some(dir) = pid_file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(pid_file)?;
    // std's own advisory lock since 1.89 — one fewer crate in the trust path for the
    // one primitive that decides whether a second server starts.
    match file.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Ok(Lock::Held),
        Err(std::fs::TryLockError::Error(e)) => return Err(e.into()),
    }
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = &file;
        f.set_len(0)?;
        f.seek(SeekFrom::Start(0))?;
        write!(f, "{}", std::process::id())?;
        f.flush()?;
    }
    Ok(Lock::Acquired(PidLock {
        _file: file,
        path: pid_file.to_path_buf(),
    }))
}

impl Drop for PidLock {
    fn drop(&mut self) {
        // Best effort. The flock is what matters; the file is a convenience for humans.
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(url: &str) -> ServerState {
        ServerState {
            schema: SCHEMA,
            pid: 1,
            url: url.into(),
            bound_addr: "127.0.0.1:21131".into(),
            api_version: API_VERSION,
            server_version: "0.1.0".into(),
            model: "m".into(),
            revision: "main".into(),
            device: "cpu".into(),
            auth: "none".into(),
            started_at: crate::util::rfc3339_now(),
        }
    }

    #[test]
    fn a_state_file_round_trips_and_never_carries_a_token() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("server.json");
        write_state(&p, &sample("http://127.0.0.1:21131")).unwrap();
        let body = std::fs::read_to_string(&p).unwrap();
        assert!(!body.contains("token"), "{body}");
        let back = read_state(&p).expect("parse");
        assert_eq!(back.url, "http://127.0.0.1:21131");
        assert_eq!(back.schema, SCHEMA);
    }

    #[test]
    fn a_corrupt_state_file_reads_as_absent_rather_than_exploding() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("server.json");
        std::fs::write(&p, "{ not json").unwrap();
        assert!(read_state(&p).is_none());
    }

    #[test]
    fn the_second_holder_is_told_it_is_held_not_handed_an_error() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("openjev.pid");
        let first = acquire(&p).unwrap();
        assert!(matches!(first, Lock::Acquired(_)));
        assert!(matches!(acquire(&p).unwrap(), Lock::Held));
        drop(first);
        assert!(matches!(acquire(&p).unwrap(), Lock::Acquired(_)));
    }

    #[cfg(unix)]
    #[test]
    fn the_state_file_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("server.json");
        write_state(&p, &sample("http://127.0.0.1:1")).unwrap();
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode {mode:o}");
    }
}
