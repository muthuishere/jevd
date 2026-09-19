//! One module per command. Shared rule: **stdout carries data, stderr carries
//! everything else**, and `--json` — never the TTY — decides stdout's shape.

pub mod configcmd;
pub mod doctor;
pub mod model;
pub mod oneshot;
pub mod status;

use crate::exit::CliResult;
use crate::runtime::{ConsentOptions, LoadSpec, ProgressRenderer, ensure_consent, plan};
use openjev_core::Session;

/// Load the model into this process. Used by the one-shot commands and `doctor --bench`.
pub fn local_session(load: &LoadSpec, assume_yes: bool) -> CliResult<Session> {
    let reg = crate::runtime::registry()?;
    let spec = crate::runtime::resolve_spec(&reg, load)?;
    let p = plan(&spec, load.cache_dir.as_ref())?;
    if let Some((need, free)) = p.space_shortfall() {
        return Err(crate::exit::CliError::Jev(
            openjev_core::JevError::Download {
                url: spec.repo.clone(),
                message: format!(
                    "needs {} free, only {} available in {}",
                    crate::util::human_bytes(need),
                    crate::util::human_bytes(free),
                    p.cache_root.display()
                ),
            },
        ));
    }
    ensure_consent(
        &p,
        &ConsentOptions {
            assume_yes,
            interactive: crate::util::stdin_is_tty() && crate::util::stderr_is_tty(),
        },
    )?;
    if spec.revision_is_floating() {
        eprintln!(
            "warning: model '{}' pins revision '{}', which is not a commit sha — \
             the weights behind it can change",
            spec.id, spec.revision
        );
    }
    let mut renderer = ProgressRenderer::new(crate::util::stderr_is_tty());
    let (session, report) = crate::runtime::load_session(&reg, load, &spec, &mut renderer)?;
    if let Some(line) = renderer.summary() {
        eprintln!("{line}");
    }
    for w in report.warnings.iter().chain(report.demotions.iter()) {
        eprintln!("warning: {w}");
    }
    eprintln!("{}", session.banner());
    Ok(session)
}
