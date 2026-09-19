//! Turning config into a loaded `Session`: the plan, the consent gate, the progress
//! rendering, and the boot itself.
//!
//! `openjev-core` never prints and never asks. Everything a human sees during a
//! multi-GB first run is here.

use crate::cli::DeviceArg;
use crate::config::Layered;
use crate::exit::{CliError, CliResult};
use crate::util::{human_bytes, human_secs};
use openjev_core::hub::{Hub, Progress};
use openjev_core::registry::HubFile;
use openjev_core::{BootOptions, BootReport, DeviceRequest, ModelSpec, Registry, Session};
use std::path::PathBuf;
use std::time::Instant;

/// What a first run is about to cost, computed **before the first byte**.
#[derive(Debug, Clone)]
pub struct LoadPlan {
    pub spec: ModelSpec,
    pub cache_root: PathBuf,
    pub missing: Vec<(String, Option<u64>)>,
    pub known_bytes: u64,
    /// True when the registry does not pin a size for something we must fetch, so the
    /// number we quote is a floor, not a total. Saying so beats quoting a wrong total.
    pub sizes_incomplete: bool,
    pub free_bytes: Option<u64>,
}

impl LoadPlan {
    pub fn needs_download(&self) -> bool {
        !self.missing.is_empty()
    }

    /// Consent is about a *wait*, not about bytes on disk. Below this a download is a
    /// blip and a prompt is noise; above it the user deserves to be asked.
    pub const CONSENT_FLOOR: u64 = 64 * 1024 * 1024;

    pub fn needs_consent(&self) -> bool {
        self.needs_download() && (self.sizes_incomplete || self.known_bytes >= Self::CONSENT_FLOOR)
    }

    /// Free space must beat the download by 10 %: ENOSPC at 94 % of 7 GB is the cruellest
    /// failure available to us, and it is entirely preventable here.
    pub fn space_shortfall(&self) -> Option<(u64, u64)> {
        let need = (self.known_bytes as f64 * 1.1) as u64;
        match self.free_bytes {
            Some(free) if need > 0 && free < need => Some((need, free)),
            _ => None,
        }
    }
}

fn files_of(spec: &ModelSpec) -> Vec<HubFile> {
    let mut out = Vec::new();
    for (_, b) in spec.ordered_backends() {
        out.push(b.weights.clone());
    }
    out.push(HubFile {
        repo: spec.tokenizer.repo.clone(),
        file: spec.tokenizer.file.clone(),
        revision: spec.tokenizer.revision.clone(),
        sha256: String::new(),
        size_bytes: None,
    });
    out.push(spec.head.file());
    out
}

pub fn hub_for(cache_dir: Option<&PathBuf>) -> Hub {
    let hub = Hub::from_env();
    match cache_dir {
        Some(d) => hub.with_root(d.clone()),
        None => hub,
    }
}

pub fn plan(spec: &ModelSpec, cache_dir: Option<&PathBuf>) -> CliResult<LoadPlan> {
    let hub = hub_for(cache_dir);
    let mut missing = Vec::new();
    let mut known = 0u64;
    let mut incomplete = false;
    for f in files_of(spec) {
        let dest = hub.path_for(&f, &spec.revision);
        let complete = hub.is_complete(&dest, &f).unwrap_or(false);
        if !complete {
            match f.size_bytes {
                Some(n) => known += n,
                None => incomplete = true,
            }
            missing.push((f.file.clone(), f.size_bytes));
        }
    }
    let free = fs4::available_space(existing_ancestor(hub.root())).ok();
    Ok(LoadPlan {
        spec: spec.clone(),
        cache_root: hub.root().to_path_buf(),
        missing,
        known_bytes: known,
        sizes_incomplete: incomplete,
        free_bytes: free,
    })
}

/// `statvfs` needs a path that exists; the cache dir usually does not yet.
fn existing_ancestor(p: &std::path::Path) -> PathBuf {
    let mut cur = p.to_path_buf();
    while !cur.exists() {
        match cur.parent() {
            Some(parent) => cur = parent.to_path_buf(),
            None => return PathBuf::from("/"),
        }
    }
    cur
}

pub struct ConsentOptions {
    /// `--yes` / `OPENJEV_ASSUME_YES`.
    pub assume_yes: bool,
    /// False for a server started by a supervisor, where there is nobody to ask.
    pub interactive: bool,
}

/// The gate `openjev-core` defines (`JevError::ConsentRequired`) and deliberately never
/// raises. Two failures are in tension and both are real: a tool that silently burns
/// 8 GB of a metered connection, and a tool that hangs forever on a prompt nobody can
/// see under systemd. So: ask on a TTY, fail loudly with the fix off one.
pub fn ensure_consent(plan: &LoadPlan, opts: &ConsentOptions) -> CliResult<()> {
    if !plan.needs_consent() || opts.assume_yes {
        return Ok(());
    }
    let size = if plan.sizes_incomplete {
        format!("at least {}", human_bytes(plan.known_bytes))
    } else {
        format!("~{}", human_bytes(plan.known_bytes))
    };
    if !opts.interactive {
        return Err(CliError::Jev(openjev_core::JevError::ConsentRequired {
            what: format!("{} weights ({size})", plan.spec.id),
            bytes: plan.known_bytes,
        }));
    }
    eprintln!();
    eprintln!(
        "This model is not cached. Downloading {size}. One time only; later runs\n\
         start in seconds. Ctrl-C is safe — the download resumes."
    );
    eprint!("Proceed? [Y/n] ");
    use std::io::Write;
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| CliError::other(format!("reading consent: {e}")))?;
    let answer = line.trim().to_ascii_lowercase();
    if answer.is_empty() || answer == "y" || answer == "yes" {
        Ok(())
    } else {
        Err(CliError::Jev(openjev_core::JevError::ConsentRequired {
            what: plan.spec.id.clone(),
            bytes: plan.known_bytes,
        }))
    }
}

/// A second consumer of hub progress — `/readyz` and `/v1/events` watch the same bytes
/// the bar draws.
pub type Observer = Box<dyn FnMut(&str, Progress) + Send>;

/// Renders hub progress. On a TTY: one bar per file. Off one: a line every 5 s or every
/// 10 %, because 40 MB of `\r` in a systemd journal helps nobody.
pub struct ProgressRenderer {
    tty: bool,
    bar: Option<indicatif::ProgressBar>,
    current: String,
    last_log: Option<Instant>,
    last_pct: u64,
    started: Instant,
    total_done: u64,
    /// Fan-out for `/v1/events` and `/readyz`.
    pub observer: Option<Observer>,
}

impl ProgressRenderer {
    pub fn new(tty: bool) -> Self {
        Self {
            tty,
            bar: None,
            current: String::new(),
            last_log: None,
            last_pct: 0,
            started: Instant::now(),
            total_done: 0,
            observer: None,
        }
    }

    pub fn elapsed(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn on(&mut self, file: &str, p: Progress) {
        if let Some(obs) = self.observer.as_mut() {
            obs(file, p);
        }
        match p {
            Progress::Start { total } => {
                self.current = file.to_string();
                self.last_pct = 0;
                self.last_log = None;
                if self.tty {
                    let bar = match total {
                        Some(t) => indicatif::ProgressBar::new(t),
                        None => indicatif::ProgressBar::new_spinner(),
                    };
                    bar.set_style(
                        indicatif::ProgressStyle::with_template(
                            "  {msg:34} {bar:18} {bytes:>10}/{total_bytes:<10} {percent:>3}% {bytes_per_sec:>10} ETA {eta}",
                        )
                        .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                        .progress_chars("██░"),
                    );
                    bar.set_message(short_name(file));
                    self.bar = Some(bar);
                } else {
                    tracing::info!(file, total_bytes = total, "download started");
                }
            }
            Progress::Advance { done, total } => {
                if let Some(bar) = &self.bar {
                    bar.set_position(done);
                } else {
                    let pct = total.map(|t| done * 100 / t.max(1)).unwrap_or(0);
                    let due_time = self
                        .last_log
                        .map(|t| t.elapsed().as_secs() >= 5)
                        .unwrap_or(true);
                    if due_time || pct >= self.last_pct + 10 {
                        self.last_log = Some(Instant::now());
                        self.last_pct = pct;
                        tracing::info!(file, done, total, pct, "downloading");
                    }
                }
            }
            Progress::Done { bytes } => {
                self.total_done += bytes;
                if let Some(bar) = self.bar.take() {
                    bar.finish_and_clear();
                    eprintln!("  ✓ {:34} {}", short_name(file), human_bytes(bytes));
                } else {
                    tracing::info!(file, bytes, "downloaded");
                }
            }
        }
    }

    pub fn summary(&self) -> Option<String> {
        (self.total_done > 0).then(|| {
            format!(
                "  ✓ downloaded {} in {}   verified sha256",
                human_bytes(self.total_done),
                human_secs(self.elapsed())
            )
        })
    }
}

fn short_name(file: &str) -> String {
    file.rsplit('/').next().unwrap_or(file).to_string()
}

pub fn device_request(d: Option<DeviceArg>) -> DeviceRequest {
    match d {
        None | Some(DeviceArg::Auto) => DeviceRequest::Auto,
        Some(DeviceArg::Cuda) => DeviceRequest::Explicit(openjev_core::Device::Cuda(0)),
        Some(DeviceArg::Metal) => DeviceRequest::Explicit(openjev_core::Device::Metal),
        Some(DeviceArg::Vulkan) => DeviceRequest::Explicit(openjev_core::Device::Vulkan),
        Some(DeviceArg::Cpu) => DeviceRequest::Explicit(openjev_core::Device::Cpu),
    }
}

/// Everything boot needs, already resolved from flags + config.
#[derive(Debug, Clone, Default)]
pub struct LoadSpec {
    pub model: Option<String>,
    pub revision: Option<String>,
    pub device: Option<DeviceArg>,
    pub cache_dir: Option<PathBuf>,
    pub offline: bool,
}

impl LoadSpec {
    pub fn from_config(cfg: &Layered) -> Self {
        Self {
            model: cfg.opt_string("model.id"),
            revision: cfg.opt_string("model.revision"),
            device: match cfg.string("device.kind").as_str() {
                "cuda" => Some(DeviceArg::Cuda),
                "metal" => Some(DeviceArg::Metal),
                "vulkan" => Some(DeviceArg::Vulkan),
                "cpu" => Some(DeviceArg::Cpu),
                _ => None,
            },
            cache_dir: cfg.opt_string("model.cache_dir").map(PathBuf::from),
            offline: false,
        }
    }
}

pub fn registry() -> CliResult<Registry> {
    Ok(Registry::load(None)?)
}

pub fn resolve_spec(reg: &Registry, load: &LoadSpec) -> CliResult<ModelSpec> {
    let spec = reg.resolve(load.model.as_deref())?.clone();
    if let Some(rev) = &load.revision
        && rev != &spec.revision
    {
        // core's `BootOptions` has no revision field: a revision is part of the model
        // entry, not a boot parameter. Overriding it here would fetch one revision and
        // load another, which is exactly the wrong-weights failure the registry exists
        // to prevent. So: refuse, and name the real lever.
        return Err(CliError::config(format!(
            "model '{}' pins revision '{}'; --revision '{rev}' cannot override it. \
             Add an entry to ~/.config/openjev/models.toml with the revision you want.",
            spec.id, spec.revision
        )));
    }
    Ok(spec)
}

/// Boot, with our progress renderer wired into core's callback.
pub fn load_session(
    reg: &Registry,
    load: &LoadSpec,
    spec: &ModelSpec,
    renderer: &mut ProgressRenderer,
) -> CliResult<(Session, BootReport)> {
    // core's `Hub::from_env()` is constructed inside `boot`, so a cache dir can only
    // reach it through the environment it reads. Setting it here keeps one cache-root
    // resolution rather than two that can disagree.
    if let Some(dir) = &load.cache_dir {
        unsafe { std::env::set_var("OPENJEV_CACHE", dir) };
    }
    let opts = BootOptions {
        model: Some(spec.id.clone()),
        device: device_request(load.device),
        dtype: None,
        context: None,
        n_threads: None,
        offline: load.offline,
    };
    let mut cb = |file: &str, p: Progress| renderer.on(file, p);
    let out = openjev_core::boot(reg, &opts, &mut cb)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_with(known: u64, incomplete: bool, missing: bool) -> LoadPlan {
        let spec = Registry::builtin()
            .unwrap()
            .default_model()
            .unwrap()
            .clone();
        LoadPlan {
            spec,
            cache_root: PathBuf::from("/tmp"),
            missing: if missing {
                vec![("w.gguf".into(), Some(known))]
            } else {
                vec![]
            },
            known_bytes: known,
            sizes_incomplete: incomplete,
            free_bytes: Some(100 * 1024 * 1024 * 1024),
        }
    }

    #[test]
    fn a_multi_gb_first_run_needs_consent_and_a_tokenizer_refresh_does_not() {
        assert!(plan_with(8 << 30, false, true).needs_consent());
        assert!(!plan_with(2 << 20, false, true).needs_consent());
        assert!(!plan_with(8 << 30, false, false).needs_consent());
        // Unknown size is not permission to skip the question.
        assert!(plan_with(0, true, true).needs_consent());
    }

    #[test]
    fn without_a_tty_consent_fails_loudly_instead_of_hanging_on_a_prompt() {
        let p = plan_with(8 << 30, false, true);
        let err = ensure_consent(
            &p,
            &ConsentOptions {
                assume_yes: false,
                interactive: false,
            },
        )
        .unwrap_err();
        assert_eq!(err.exit_code(), crate::exit::CONSENT_REQUIRED);
        assert!(err.to_string().contains("--yes"), "{err}");
    }

    #[test]
    fn assume_yes_skips_the_gate_entirely() {
        let p = plan_with(8 << 30, false, true);
        assert!(
            ensure_consent(
                &p,
                &ConsentOptions {
                    assume_yes: true,
                    interactive: false
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn a_disk_that_cannot_hold_the_download_is_caught_before_the_first_byte() {
        let mut p = plan_with(8 << 30, false, true);
        p.free_bytes = Some(4 << 30);
        let (need, free) = p.space_shortfall().expect("shortfall");
        assert!(need > free);
        p.free_bytes = Some(64 << 30);
        assert!(p.space_shortfall().is_none());
    }

    #[test]
    fn a_revision_that_contradicts_the_registry_is_refused_not_quietly_ignored() {
        let reg = Registry::builtin().unwrap();
        let load = LoadSpec {
            revision: Some("deadbeef".into()),
            ..Default::default()
        };
        let err = resolve_spec(&reg, &load).unwrap_err();
        assert_eq!(err.exit_code(), crate::exit::CONFIG);
        assert!(err.to_string().contains("models.toml"), "{err}");
    }
}
