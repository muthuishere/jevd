//! `openjev status` — is a server running, where, which model, how far along.
//!
//! Exit 0 running-and-ready, 6 no server, 7 running-but-not-ready. Three codes because a
//! script that starts a server needs to tell "not there" from "not yet".

use crate::cli::StatusArgs;
use crate::client::{Client, discover};
use crate::exit::{self, CliResult};

pub fn run(args: &StatusArgs, json: bool) -> CliResult<i32> {
    let Some((url, how)) = discover(None, args.state_file.as_deref()) else {
        if json {
            println!("{}", serde_json::json!({"running": false}));
        } else {
            println!("no openjev server is running");
        }
        return Ok(exit::NO_SERVER);
    };
    let client = Client::new(&url);
    let ready = client.readyz()?;
    let info = client.info().ok();
    let state = crate::state::read_state(
        &args
            .state_file
            .clone()
            .unwrap_or_else(crate::state::default_state_file),
    );

    if json {
        println!(
            "{}",
            serde_json::json!({
                "running": true,
                "url": url,
                "discovered_via": how.describe(),
                "ready": ready.ready,
                "phase": ready.phase,
                "detail": ready.detail,
                "pid": state.as_ref().map(|s| s.pid),
                "model": info.as_ref().and_then(|i| i.model.as_ref().map(|m| m.model.clone())),
                "revision": info.as_ref().and_then(|i| i.model.as_ref().map(|m| m.revision.clone())),
                "device": info.as_ref().and_then(|i| i.model.as_ref().map(|m| m.device.clone())),
                "api_version": info.as_ref().map(|i| i.api_version),
                "server_version": info.as_ref().map(|i| i.server_version.clone()),
            })
        );
    } else {
        println!("url       {url}   (via {})", how.describe());
        if let Some(s) = &state {
            println!("pid       {}   started {}", s.pid, s.started_at);
        }
        println!(
            "phase     {}{}",
            ready.phase,
            if ready.ready { "" } else { "  (not ready)" }
        );
        if let Some(d) = &ready.detail
            && let (Some(done), Some(total)) = (d.bytes_done, d.bytes_total)
        {
            println!(
                "progress  {} / {}  {}",
                crate::util::human_bytes(done),
                crate::util::human_bytes(total),
                d.file.clone().unwrap_or_default()
            );
        }
        if let Some(m) = info.as_ref().and_then(|i| i.model.clone()) {
            println!("model     {}@{}  on {}", m.model, m.revision, m.device);
        }
    }
    Ok(if ready.ready {
        exit::OK
    } else {
        exit::NOT_READY
    })
}
