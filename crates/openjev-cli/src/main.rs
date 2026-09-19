mod api;
mod cli;
mod config;
mod engine;
mod exit;
mod logging;
mod runtime;
mod state;
mod util;

fn main() {
    println!("openjev {}", env!("CARGO_PKG_VERSION"));
}
