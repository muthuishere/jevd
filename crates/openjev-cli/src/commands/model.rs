//! `openjev model {pull,list,rm,path}` — the cache, as a thing you can inspect and
//! empty without knowing our directory layout.

use crate::cli::ModelCmd;
use crate::exit::{self, CliError, CliResult};
use crate::runtime::{LoadSpec, ProgressRenderer, hub_for};
use openjev_core::registry::HubFile;
use openjev_core::{ModelSpec, Registry};
use std::path::PathBuf;

fn files_of(spec: &ModelSpec) -> Vec<HubFile> {
    let mut v: Vec<HubFile> = spec
        .ordered_backends()
        .into_iter()
        .map(|(_, b)| b.weights.clone())
        .collect();
    v.push(HubFile {
        repo: spec.tokenizer.repo.clone(),
        file: spec.tokenizer.file.clone(),
        revision: spec.tokenizer.revision.clone(),
        sha256: String::new(),
        size_bytes: None,
    });
    v.push(spec.head.file());
    v
}

pub fn run(
    cmd: &ModelCmd,
    cfg: &crate::config::Layered,
    json: bool,
    assume_yes: bool,
) -> CliResult<i32> {
    let reg = Registry::load(None)?;
    let base = LoadSpec::from_config(cfg);
    match cmd {
        ModelCmd::Pull {
            model,
            revision,
            force,
            offline,
        } => {
            let load = LoadSpec {
                model: model.clone().or(base.model.clone()),
                revision: revision.clone(),
                offline: *offline,
                ..base.clone()
            };
            let spec = crate::runtime::resolve_spec(&reg, &load)?;
            let hub = hub_for(load.cache_dir.as_ref());
            if *force {
                for f in files_of(&spec) {
                    let p = hub.path_for(&f, &spec.revision);
                    let _ = std::fs::remove_file(&p);
                    let _ = std::fs::remove_file(p.with_extension("ok"));
                }
            }
            let plan = crate::runtime::plan(&spec, load.cache_dir.as_ref())?;
            if let Some((need, free)) = plan.space_shortfall() {
                return Err(CliError::other(format!(
                    "not enough space: need {}, have {}",
                    crate::util::human_bytes(need),
                    crate::util::human_bytes(free)
                )));
            }
            if !plan.needs_download() {
                eprintln!("already cached: {}@{}", spec.id, spec.revision);
                return Ok(exit::OK);
            }
            crate::runtime::ensure_consent(
                &plan,
                &crate::runtime::ConsentOptions {
                    assume_yes,
                    interactive: crate::util::stdin_is_tty() && crate::util::stderr_is_tty(),
                },
            )?;
            let mut renderer = ProgressRenderer::new(crate::util::stderr_is_tty());
            let hub = hub.offline(load.offline);
            for f in files_of(&spec) {
                let mut cb = |file: &str, p: openjev_core::hub::Progress| renderer.on(file, p);
                hub.get_with_progress(&f, &spec.revision, &mut cb)?;
            }
            if let Some(line) = renderer.summary() {
                eprintln!("{line}");
            }
            // Weights only. Loading is `serve`'s job — "download now, on good wifi,
            // before the demo" must not also need a GPU to be free.
            Ok(exit::OK)
        }

        ModelCmd::List => {
            let hub = hub_for(base.cache_dir.as_ref());
            let mut rows = Vec::new();
            for spec in reg.iter() {
                let mut bytes = 0u64;
                let mut present = 0usize;
                let mut total = 0usize;
                let mut last_used = None;
                let mut path = None;
                for f in files_of(spec) {
                    total += 1;
                    let p = hub.path_for(&f, &spec.revision);
                    if let Ok(md) = std::fs::metadata(&p) {
                        present += 1;
                        bytes += md.len();
                        path.get_or_insert_with(|| {
                            p.parent().map(PathBuf::from).unwrap_or(p.clone())
                        });
                        if let Ok(t) = md.modified()
                            && let Ok(d) = t.duration_since(std::time::UNIX_EPOCH)
                        {
                            last_used = Some(crate::util::rfc3339(d.as_secs()));
                        }
                    }
                }
                rows.push(serde_json::json!({
                    "ref": spec.id,
                    "revision": spec.revision,
                    "size_bytes": bytes,
                    "files_present": present,
                    "files_total": total,
                    "path": path.map(|p| p.display().to_string()),
                    "last_used": last_used,
                }));
            }
            if json {
                println!("{}", serde_json::Value::Array(rows));
            } else if rows.is_empty() {
                println!("no models in the registry");
            } else {
                for r in rows {
                    println!(
                        "{:<24} {:<10} {:>10}  {}/{} files  {}",
                        r["ref"].as_str().unwrap_or(""),
                        short(r["revision"].as_str().unwrap_or("")),
                        crate::util::human_bytes(r["size_bytes"].as_u64().unwrap_or(0)),
                        r["files_present"],
                        r["files_total"],
                        r["path"].as_str().unwrap_or("(not cached)")
                    );
                }
            }
            Ok(exit::OK)
        }

        ModelCmd::Rm {
            model,
            force,
            dry_run,
        } => {
            let spec = reg.get(model)?;
            // Never delete weights out from under a live server: the forward pass would
            // fail in a way that looks like a model bug.
            if !force
                && let Some((url, _)) = crate::client::discover(None, None)
                && let Ok(info) = crate::client::Client::new(&url).info()
                && info.model.as_ref().is_some_and(|m| m.model == spec.id)
            {
                return Err(CliError::other(format!(
                    "{} is loaded by the server at {url}; stop it or pass --force",
                    spec.id
                )));
            }
            let hub = hub_for(base.cache_dir.as_ref());
            let mut freed = 0u64;
            let mut removed = Vec::new();
            for f in files_of(spec) {
                let p = hub.path_for(&f, &spec.revision);
                if let Ok(md) = std::fs::metadata(&p) {
                    freed += md.len();
                    removed.push(p.display().to_string());
                    if !dry_run {
                        std::fs::remove_file(&p).map_err(CliError::Io)?;
                        let _ = std::fs::remove_file(p.with_extension("ok"));
                    }
                }
            }
            if json {
                println!(
                    "{}",
                    serde_json::json!({"removed": removed, "freed_bytes": freed, "dry_run": dry_run})
                );
            } else {
                println!(
                    "{} {} ({} files)",
                    if *dry_run { "would free" } else { "freed" },
                    crate::util::human_bytes(freed),
                    removed.len()
                );
            }
            Ok(exit::OK)
        }

        ModelCmd::Path { model } => {
            let load = LoadSpec {
                model: model.clone().or(base.model.clone()),
                ..base.clone()
            };
            let spec = crate::runtime::resolve_spec(&reg, &load)?;
            let hub = hub_for(load.cache_dir.as_ref());
            let first = files_of(&spec)
                .first()
                .map(|f| hub.path_for(f, &spec.revision))
                .ok_or_else(|| CliError::other("model declares no files"))?;
            let dir = first.parent().unwrap_or(&first);
            // Bare path on stdout: this exists to be substituted into a shell command.
            println!("{}", dir.display());
            Ok(exit::OK)
        }
    }
}

fn short(rev: &str) -> &str {
    if rev.len() >= 40 { &rev[..8] } else { rev }
}
