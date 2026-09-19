//! `tracing` init. Diagnostics go to **stderr**; stdout carries data only.
//!
//! This is the one place a TTY changes a shape (text vs JSON lines), and it is legitimate
//! precisely because logs are not stdout.

use crate::cli::LogFormat;
use std::sync::OnceLock;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::reload;

type ReloadHandle = reload::Handle<EnvFilter, tracing_subscriber::Registry>;
static RELOAD: OnceLock<ReloadHandle> = OnceLock::new();

/// Level precedence: `--log-level` > `OPENJEV_LOG` > `RUST_LOG` > `info` (§4.7).
pub fn filter_directive(flag: Option<&str>) -> String {
    flag.map(str::to_string)
        .or_else(|| std::env::var("OPENJEV_LOG").ok())
        .or_else(|| std::env::var("RUST_LOG").ok())
        .unwrap_or_else(|| "info".to_string())
}

pub fn init(level: Option<&str>, format: LogFormat) {
    use tracing_subscriber::prelude::*;

    let directive = filter_directive(level);
    let filter = EnvFilter::try_new(&directive).unwrap_or_else(|_| EnvFilter::new("info"));
    let (filter, handle) = reload::Layer::new(filter);
    let _ = RELOAD.set(handle);

    let json = match format {
        LogFormat::Json => true,
        LogFormat::Text => false,
        LogFormat::Auto => !crate::util::stderr_is_tty(),
    };

    let registry = tracing_subscriber::registry().with(filter);
    if json {
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .with_writer(std::io::stderr);
        let _ = registry.with(layer).try_init();
    } else {
        let layer = tracing_subscriber::fmt::layer()
            .with_target(false)
            .with_writer(std::io::stderr);
        let _ = registry.with(layer).try_init();
    }
}

/// SIGHUP path. A reload that silently does nothing is how you spend an afternoon
/// wondering why the level did not change, so failures are returned, not swallowed.
pub fn set_level(directive: &str) -> Result<(), String> {
    let handle = RELOAD.get().ok_or("logging is not initialised")?;
    let filter = EnvFilter::try_new(directive).map_err(|e| e.to_string())?;
    handle.reload(filter).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_beats_the_environment() {
        unsafe { std::env::set_var("OPENJEV_LOG", "warn") };
        assert_eq!(filter_directive(Some("debug")), "debug");
        assert_eq!(filter_directive(None), "warn");
        unsafe { std::env::remove_var("OPENJEV_LOG") };
    }
}
