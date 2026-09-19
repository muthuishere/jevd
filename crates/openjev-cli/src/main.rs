//! The binary. Everything it does lives in the library beside it, so the router, the
//! error envelope and the config layering can be tested without spawning a process.

use clap::Parser;
use openjev_cli::cli::{Cli, Command, LogFormat};
use openjev_cli::exit::CliResult;
use openjev_cli::{api, commands, config, exit, logging, server};
use std::collections::BTreeMap;

fn main() {
    let cli = Cli::parse();
    logging::init(
        cli.log_level.as_deref(),
        cli.log_format.unwrap_or(LogFormat::Auto),
    );
    match dispatch(&cli) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            // Errors are diagnostics: stderr, always, so a `--json` stdout stays parseable
            // even on the failure path.
            eprintln!("error: {e}");
            std::process::exit(e.exit_code());
        }
    }
}

fn dispatch(cli: &Cli) -> CliResult<i32> {
    let env: BTreeMap<String, String> = std::env::vars().collect();
    let mut cfg = config::Layered::load(cli.config.as_deref(), &env)?;

    match &cli.command {
        Command::Serve(args) => {
            let server_cfg = server::build_config(args, &mut cfg)?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(exit::CliError::Io)?;
            rt.block_on(server::run(server_cfg, cli.yes))
        }
        Command::Predict(args) => commands::oneshot::predict(args, &cfg, cli.json, cli.yes),
        Command::Rerank(args) => commands::oneshot::rerank(args, &cfg, cli.json, cli.yes),
        Command::Grade(args) => commands::oneshot::grade(args, &cfg, cli.json, cli.yes),
        Command::Model(cmd) => commands::model::run(cmd, &cfg, cli.json, cli.yes),
        Command::Status(args) => commands::status::run(args, cli.json),
        Command::Doctor(args) => commands::doctor::run(args, &cfg, cli.json, cli.yes),
        Command::Config(cmd) => commands::configcmd::run(cmd, &cfg, cli.config.as_ref(), cli.json),
        Command::Completions { shell } => {
            use clap::CommandFactory;
            clap_complete::generate(
                *shell,
                &mut Cli::command(),
                "openjev",
                &mut std::io::stdout(),
            );
            Ok(exit::OK)
        }
        Command::Version => {
            let devices = openjev_core::backend::factories()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>()
                .join(",");
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "version": env!("CARGO_PKG_VERSION"),
                        "api_version": api::API_VERSION,
                        "backends": openjev_core::backend::factories().iter().map(|f| f.name()).collect::<Vec<_>>(),
                        "os": std::env::consts::OS,
                        "arch": std::env::consts::ARCH,
                    })
                );
            } else {
                println!(
                    "openjev {}  ·  api v{}  ·  {} {}  ·  backends: {}",
                    env!("CARGO_PKG_VERSION"),
                    api::API_VERSION,
                    std::env::consts::OS,
                    std::env::consts::ARCH,
                    if devices.is_empty() {
                        "none compiled in"
                    } else {
                        &devices
                    }
                );
            }
            Ok(exit::OK)
        }
    }
}
