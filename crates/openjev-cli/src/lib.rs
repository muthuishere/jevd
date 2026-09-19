//! `openjev-cli` — NLI inference over HTTP, and the one-shot commands that compose in a
//! pipeline.
//!
//! main is the only place that unwraps, prints an error and picks an exit code. Every
//! other function returns `CliResult`, so no command can exit out from under a caller.

pub mod api;
pub mod cli;
pub mod client;
pub mod commands;
pub mod config;
pub mod engine;
pub mod exit;
pub mod logging;
pub mod openapi;
pub mod runtime;
pub mod server;
pub mod state;
pub mod util;
