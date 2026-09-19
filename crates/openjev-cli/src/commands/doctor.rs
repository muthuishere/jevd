//! `openjev doctor` — one screen of ground truth, copy-pasteable into an issue, with no
//! secret in it. This is the first thing we ask for in a bug report.

use crate::cli::DoctorArgs;
use crate::exit::{self, CliResult};
use crate::runtime::LoadSpec;
use openjev_core::device::{Device, DeviceProbe, HostProbe};
use serde_json::json;

struct Check {
    name: String,
    ok: bool,
    detail: String,
}

pub fn run(
    args: &DoctorArgs,
    cfg: &crate::config::Layered,
    json_out: bool,
    assume_yes: bool,
) -> CliResult<i32> {
    let mut checks: Vec<Check> = Vec::new();
    let mut push = |name: &str, ok: bool, detail: String| {
        checks.push(Check {
            name: name.into(),
            ok,
            detail,
        })
    };

    push(
        "build",
        true,
        format!(
            "openjev {} · {} {} · backends compiled in: {}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            backends_compiled_in()
        ),
    );

    // Devices, and *why* — silent CPU fallback on a CUDA box is the #1 "why is this slow"
    // bug in every tool of this shape.
    let probe = HostProbe;
    let present: Vec<&str> = [Device::Cuda(0), Device::Metal, Device::Vulkan, Device::Cpu]
        .into_iter()
        .filter(|d| DeviceProbe::present(&probe, *d))
        .map(|d| d.kind())
        .collect();
    push(
        "devices",
        !present.is_empty(),
        format!(
            "present: {}  ·  requested: {} (from {})",
            present.join(", "),
            cfg.string("device.kind"),
            cfg.source_of("device.kind")
        ),
    );

    let load = LoadSpec::from_config(cfg);
    let reg = crate::runtime::registry()?;
    match crate::runtime::resolve_spec(&reg, &load) {
        Ok(spec) => {
            push(
                "model",
                true,
                format!(
                    "{}@{}{}",
                    spec.id,
                    spec.revision,
                    if spec.revision_is_floating() {
                        "  (WARNING: not a commit sha — the weights can change)"
                    } else {
                        ""
                    }
                ),
            );
            let plan = crate::runtime::plan(&spec, load.cache_dir.as_ref())?;
            let free = plan
                .free_bytes
                .map(crate::util::human_bytes)
                .unwrap_or_else(|| "unknown".into());
            let shortfall = plan.space_shortfall();
            push(
                "cache",
                shortfall.is_none(),
                format!(
                    "{}  ·  {free} free  ·  {} file(s) to fetch ({})",
                    plan.cache_root.display(),
                    plan.missing.len(),
                    crate::util::human_bytes(plan.known_bytes)
                ),
            );
        }
        Err(e) => push("model", false, e.to_string()),
    }

    let hub_ok = crate::client::reachable("https://huggingface.co");
    push(
        "hub",
        true,
        format!(
            "huggingface.co {}",
            if hub_ok {
                "reachable"
            } else {
                "unreachable (fine if the model is cached; --offline makes it explicit)"
            }
        ),
    );

    let cfg_path = crate::config::default_config_path();
    push(
        "config",
        true,
        format!(
            "{} ({})",
            cfg_path.display(),
            if cfg_path.is_file() {
                "found"
            } else {
                "not found — using defaults"
            }
        ),
    );

    match crate::client::discover(None, None) {
        Some((url, how)) => push("server", true, format!("{url} (via {})", how.describe())),
        None => push("server", true, "none running".into()),
    }

    let mut bench = None;
    if args.bench {
        let session = crate::commands::local_session(&load, assume_yes)?;
        let pairs: Vec<(&str, &str)> = (0..20)
            .map(|_| ("it rained", "the ground is wet"))
            .collect();
        let mut times = Vec::new();
        for p in &pairs {
            let t = std::time::Instant::now();
            session.predict(std::slice::from_ref(p))?;
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        let p50 = times[times.len() / 2];
        let p95 = times[(times.len() * 95 / 100).min(times.len() - 1)];
        push(
            "bench",
            true,
            format!("20 pairs · p50 {p50:.0} ms · p95 {p95:.0} ms"),
        );
        bench = Some((p50, p95));
    }

    let all_ok = checks.iter().all(|c| c.ok);
    if json_out {
        println!(
            "{}",
            json!({
                "ok": all_ok,
                "checks": checks.iter().map(|c| json!({"name": c.name, "ok": c.ok, "detail": c.detail})).collect::<Vec<_>>(),
                "bench": bench.map(|(a, b)| json!({"p50_ms": a, "p95_ms": b})),
            })
        );
    } else {
        for c in &checks {
            println!(
                "{}  {:<9} {}",
                if c.ok { "✓" } else { "✗" },
                c.name,
                c.detail
            );
        }
    }
    Ok(if all_ok { exit::OK } else { exit::FAILURE })
}

fn backends_compiled_in() -> String {
    let names: Vec<&str> = openjev_core::backend::factories()
        .iter()
        .map(|f| f.name())
        .collect();
    if names.is_empty() {
        "none (build with --features backend-llamacpp, or metal/cuda/vulkan)".to_string()
    } else {
        names.join(", ")
    }
}
