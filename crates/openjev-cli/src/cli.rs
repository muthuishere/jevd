//! The command tree (design 02 §2.1), as clap derive.
//!
//! Every flag that also exists in the config file is an `Option<T>` here and stays
//! `None` when unset. That is load-bearing: `Some(default)` from clap would make the
//! flag layer win over the config file for a value the user never typed, which is the
//! classic "why is my config file ignored" bug.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

pub const DEFAULT_PORT: u16 = 21131;

#[derive(Debug, Parser)]
#[command(
    name = "openjev",
    version,
    about = "NLI inference: predict, rerank, grade — over HTTP or in a pipeline.",
    disable_help_subcommand = true
)]
pub struct Cli {
    /// Config file to use instead of ~/.config/openjev/config.toml.
    #[arg(long, global = true, value_name = "PATH", env = "OPENJEV_CONFIG")]
    pub config: Option<PathBuf>,

    /// Log level. Overrides OPENJEV_LOG and RUST_LOG.
    #[arg(long, global = true, value_name = "LVL")]
    pub log_level: Option<String>,

    /// Log format. `auto` = text on a TTY, JSON lines otherwise.
    #[arg(long, global = true, value_enum)]
    pub log_format: Option<LogFormat>,

    /// Machine-readable output on stdout. Never inferred from the TTY.
    #[arg(long, global = true)]
    pub json: bool,

    /// Assume yes to the first-run download consent prompt.
    #[arg(short = 'y', long, global = true, env = "OPENJEV_ASSUME_YES")]
    pub yes: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    Auto,
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DeviceArg {
    Auto,
    Cuda,
    Metal,
    Vulkan,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TruncateArg {
    /// Refuse over-long input. Default: a silently truncated premise gives a confident,
    /// wrong, unfalsifiable answer.
    Error,
    Tail,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the HTTP server.
    Serve(Box<ServeArgs>),
    /// One-shot NLI over premise/hypothesis pairs.
    Predict(PredictArgs),
    /// One-shot rerank of options against a question.
    Rerank(RerankArgs),
    /// One-shot answer-vs-reference grade. Exits 9 when the assertion fails.
    Grade(GradeArgs),
    /// Weights in the cache.
    #[command(subcommand)]
    Model(ModelCmd),
    /// Is a server running, where, which model.
    Status(StatusArgs),
    /// Environment, device, cache and connectivity — the thing to paste into a bug report.
    Doctor(DoctorArgs),
    /// Read and write the config file.
    #[command(subcommand)]
    Config(ConfigCmd),
    /// Shell completions, generated not hand-written.
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
    /// Version, api version, build, device support.
    Version,
}

#[derive(Debug, clap::Args)]
pub struct ServeArgs {
    #[arg(long, value_name = "IP")]
    pub host: Option<String>,
    /// 0 asks the kernel for a port; the real one lands in the state file.
    #[arg(long, value_name = "N")]
    pub port: Option<u16>,
    #[arg(long, value_name = "REF")]
    pub model: Option<String>,
    #[arg(long, value_name = "REV")]
    pub revision: Option<String>,
    #[arg(long, value_enum)]
    pub device: Option<DeviceArg>,
    #[arg(long, value_name = "DIR")]
    pub cache_dir: Option<PathBuf>,
    #[arg(long, value_name = "TOKEN", env = "OPENJEV_TOKEN", conflicts_with_all = ["token_file", "no_auth"])]
    pub token: Option<String>,
    #[arg(long, value_name = "F", conflicts_with = "no_auth")]
    pub token_file: Option<PathBuf>,
    /// Serve a non-loopback address with no auth. Logs WARN on every startup, forever.
    #[arg(long)]
    pub no_auth: bool,
    #[arg(long = "cors-origin", value_name = "ORIGIN")]
    pub cors_origin: Vec<String>,
    #[arg(long, value_name = "N")]
    pub max_queue: Option<usize>,
    #[arg(long, value_name = "N")]
    pub max_batch: Option<usize>,
    #[arg(long, value_name = "SECS")]
    pub request_timeout: Option<u64>,
    #[arg(long, value_name = "SECS")]
    pub shutdown_grace: Option<u64>,
    #[arg(long, value_name = "PATH")]
    pub state_file: Option<PathBuf>,
    /// Print exactly one line of JSON to stdout when ready, and nothing else, ever.
    #[arg(long)]
    pub print_ready_json: bool,
    /// Exit 1 instead of 0 when another instance already holds the lock.
    #[arg(long)]
    pub fail_if_running: bool,
    #[arg(long)]
    pub offline: bool,
    /// Block readiness until weights are resident (default on).
    #[arg(long, overrides_with = "no_preload")]
    pub preload: bool,
    #[arg(long = "no-preload")]
    pub no_preload: bool,
}

impl ServeArgs {
    pub fn preload(&self) -> bool {
        !self.no_preload
    }
}

/// Shared by the one-shot commands: where the work runs.
#[derive(Debug, clap::Args, Clone)]
pub struct Target {
    /// Send to a running server instead of loading the model in this process.
    #[arg(
        long,
        value_name = "URL",
        env = "OPENJEV_URL",
        conflicts_with = "local"
    )]
    pub server: Option<String>,
    /// Force in-process inference even when a server is discoverable.
    #[arg(long)]
    pub local: bool,
    #[arg(long, value_name = "REF")]
    pub model: Option<String>,
    #[arg(long, value_enum)]
    pub device: Option<DeviceArg>,
    #[arg(long)]
    pub offline: bool,
}

#[derive(Debug, clap::Args)]
pub struct PredictArgs {
    #[arg(long, requires = "hypothesis")]
    pub premise: Option<String>,
    #[arg(long, requires = "premise")]
    pub hypothesis: Option<String>,
    #[arg(long, value_enum, default_value_t = TruncateArg::Error)]
    pub truncate: TruncateArg,
    #[command(flatten)]
    pub target: Target,
}

#[derive(Debug, clap::Args)]
pub struct RerankArgs {
    pub question: String,
    /// Repeatable. Omit entirely to read one option per line from stdin.
    #[arg(long = "option", value_name = "TEXT")]
    pub options: Vec<String>,
    #[arg(long, value_name = "N")]
    pub top_k: Option<usize>,
    /// Echo each option's text in the output.
    #[arg(long)]
    pub return_documents: bool,
    #[command(flatten)]
    pub target: Target,
}

#[derive(Debug, clap::Args)]
pub struct GradeArgs {
    #[arg(long)]
    pub answer: String,
    #[arg(long)]
    pub reference: String,
    /// pass = P(entailment) >= threshold. Also drives the exit code: 0 pass, 9 fail.
    #[arg(long, default_value_t = 0.5)]
    pub threshold: f32,
    #[command(flatten)]
    pub target: Target,
}

#[derive(Debug, Subcommand)]
pub enum ModelCmd {
    /// Download weights, do not serve.
    Pull {
        model: Option<String>,
        #[arg(long)]
        revision: Option<String>,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        offline: bool,
    },
    /// What is in the cache, with sizes and revisions.
    List,
    /// Delete a cached model.
    Rm {
        model: String,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        dry_run: bool,
    },
    /// Print the on-disk path. Scriptable: bare path on stdout.
    Path { model: Option<String> },
}

#[derive(Debug, clap::Args)]
pub struct StatusArgs {
    #[arg(long, value_name = "PATH")]
    pub state_file: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub struct DoctorArgs {
    /// Run 20 pairs and print p50/p95. Loads the model.
    #[arg(long)]
    pub bench: bool,
}

#[derive(Debug, Subcommand)]
pub enum ConfigCmd {
    /// Print the config file path, whether or not it exists.
    Path,
    Show {
        /// Annotate every key with the layer that set it.
        #[arg(long)]
        sources: bool,
    },
    /// Print one value, bare, for scripts.
    Get { key: String },
    Set {
        key: String,
        value: String,
        /// Permit writing a token into the config file in plaintext.
        #[arg(long)]
        allow_inline_token: bool,
    },
    /// $EDITOR, validated before it replaces the file.
    Edit,
    Validate {
        #[arg(long, value_name = "F")]
        file: Option<PathBuf>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).expect("parse")
    }

    #[test]
    fn the_tree_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn serve_defaults_are_all_none_so_the_config_file_can_win() {
        let cli = parse(&["openjev", "serve"]);
        let Command::Serve(s) = cli.command else {
            panic!("serve")
        };
        assert!(s.host.is_none() && s.port.is_none() && s.max_queue.is_none());
        assert!(s.preload(), "preload is on by default");
    }

    #[test]
    fn no_preload_wins_when_given() {
        let cli = parse(&["openjev", "serve", "--no-preload"]);
        let Command::Serve(s) = cli.command else {
            panic!("serve")
        };
        assert!(!s.preload());
    }

    #[test]
    fn token_and_no_auth_are_mutually_exclusive() {
        assert!(
            Cli::try_parse_from(["openjev", "serve", "--token", "x", "--no-auth"]).is_err(),
            "asking for auth and no auth at once is a usage error, not a precedence puzzle"
        );
    }

    #[test]
    fn predict_pair_flags_come_as_a_pair() {
        assert!(Cli::try_parse_from(["openjev", "predict", "--premise", "a"]).is_err());
        assert!(
            Cli::try_parse_from(["openjev", "predict", "--premise", "a", "--hypothesis", "b"])
                .is_ok()
        );
    }

    #[test]
    fn json_is_global_and_never_inferred() {
        let cli = parse(&["openjev", "--json", "status"]);
        assert!(cli.json);
        let cli = parse(&["openjev", "status"]);
        assert!(!cli.json);
    }

    #[test]
    fn grade_threshold_defaults_to_half() {
        let cli = parse(&["openjev", "grade", "--answer", "a", "--reference", "b"]);
        let Command::Grade(g) = cli.command else {
            panic!("grade")
        };
        assert_eq!(g.threshold, 0.5);
    }

    #[test]
    fn server_and_local_cannot_both_be_asked_for() {
        assert!(
            Cli::try_parse_from(["openjev", "rerank", "q", "--server", "http://x", "--local"])
                .is_err()
        );
    }
}
