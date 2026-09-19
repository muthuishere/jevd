//! Exit codes, and the one place a `JevError` becomes one.
//!
//! Codes are the scriptable contract (design 02 §2.6) and are **additive only** — never
//! renumbered. Organised against: a CI job that cannot tell "the model is missing" from
//! "your answer was wrong", because both were exit 1.

use openjev_core::JevError;

pub const OK: i32 = 0;
pub const FAILURE: i32 = 1;
/// clap owns 2 and returns it itself; it is named here so the table is complete and so
/// nothing else ever claims the number.
#[allow(dead_code)]
pub const USAGE: i32 = 2;
pub const CONFIG: i32 = 3;
pub const MODEL_UNAVAILABLE: i32 = 4;
pub const DEVICE_UNAVAILABLE: i32 = 5;
pub const NO_SERVER: i32 = 6;
pub const NOT_READY: i32 = 7;
pub const BAD_INPUT: i32 = 8;
pub const ASSERTION_FAILED: i32 = 9;
/// Consent for a multi-GB download was required and not given. Outside the 0–9 table on
/// purpose: it is `openjev-core`'s own `JevError::exit_code()` value (sysexits' EX_NOPERM
/// neighbourhood), and two different numbers for one condition is worse than one number
/// outside a table. See docs/adr/0009.
pub const CONSENT_REQUIRED: i32 = 77;
pub const INTERRUPTED: i32 = 130;

/// The CLI's own failures, as distinct from the library's.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    BadInput(String),
    #[error("no openjev server is reachable{0}")]
    NoServer(String),
    #[error("{0}")]
    Other(String),
    #[error(transparent)]
    Jev(#[from] JevError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl CliError {
    pub fn other(m: impl Into<String>) -> Self {
        CliError::Other(m.into())
    }
    pub fn config(m: impl Into<String>) -> Self {
        CliError::Config(m.into())
    }
    pub fn bad_input(m: impl Into<String>) -> Self {
        CliError::BadInput(m.into())
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::Config(_) => CONFIG,
            CliError::BadInput(_) => BAD_INPUT,
            CliError::NoServer(_) => NO_SERVER,
            CliError::Other(_) | CliError::Io(_) => FAILURE,
            CliError::Jev(e) => jev_exit_code(e),
        }
    }
}

/// `JevError::exit_code()` speaks sysexits (78/77/69); the CLI speaks design 02's table.
/// Translating here, once, is why no command has to know both.
pub fn jev_exit_code(e: &JevError) -> i32 {
    match e {
        JevError::ModelNotFound { .. } | JevError::Config { .. } | JevError::Model(_) => CONFIG,
        JevError::BackendUnavailable { .. } | JevError::Device(_) => DEVICE_UNAVAILABLE,
        JevError::Download { .. } | JevError::Integrity { .. } | JevError::Offline { .. } => {
            MODEL_UNAVAILABLE
        }
        JevError::ConsentRequired { .. } => CONSENT_REQUIRED,
        JevError::ContextOverflow { .. } => BAD_INPUT,
        _ => FAILURE,
    }
}

pub type CliResult<T> = std::result::Result<T, CliError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_sysexits_become_cli_codes() {
        assert_eq!(
            jev_exit_code(&JevError::Model("x".into())),
            CONFIG,
            "a bad model spec is a config error, not a generic failure"
        );
        assert_eq!(
            jev_exit_code(&JevError::Offline {
                what: "weights".into(),
                path: "/tmp/x".into()
            }),
            MODEL_UNAVAILABLE
        );
        assert_eq!(
            jev_exit_code(&JevError::ConsentRequired {
                what: "weights".into(),
                bytes: 1
            }),
            CONSENT_REQUIRED
        );
        assert_eq!(
            jev_exit_code(&JevError::Device("no metal".into())),
            DEVICE_UNAVAILABLE
        );
    }

    #[test]
    fn codes_are_the_documented_ones() {
        // Renumbering these silently breaks every CI job that branches on them.
        assert_eq!(
            [
                OK,
                FAILURE,
                USAGE,
                CONFIG,
                MODEL_UNAVAILABLE,
                DEVICE_UNAVAILABLE,
                NO_SERVER,
                NOT_READY,
                BAD_INPUT,
                ASSERTION_FAILED
            ],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(INTERRUPTED, 130);
    }
}
