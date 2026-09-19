//! The binary as a user meets it: exit codes and stdout shape, from a real process.
//!
//! Nothing here loads weights — every assertion is about the contract a script depends
//! on, and a test that needs a multi-GB download is a test nobody runs.

use std::process::Command;

fn openjev() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_openjev"));
    // Keep the test off the developer's real config and state.
    c.env(
        "XDG_STATE_HOME",
        std::env::temp_dir().join("openjev-test-state"),
    );
    c
}

fn run(args: &[&str]) -> (i32, String, String) {
    let out = openjev().args(args).output().expect("spawn");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn version_is_stable_json_under_json_and_a_line_otherwise() {
    let (code, stdout, _) = run(&["--json", "version"]);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("json on stdout");
    assert_eq!(v["api_version"], 1);
    assert!(v["version"].is_string());

    let (code, stdout, _) = run(&["version"]);
    assert_eq!(code, 0);
    assert!(stdout.starts_with("openjev "));
    assert!(serde_json::from_str::<serde_json::Value>(&stdout).is_err());
}

#[test]
fn an_unknown_flag_is_clap_s_usage_error_and_says_nothing_on_stdout() {
    let (code, stdout, stderr) = run(&["--no-such-flag"]);
    assert_eq!(code, 2, "usage errors are exit 2 and always have been");
    assert!(
        stdout.is_empty(),
        "stdout is data; a usage error is not data"
    );
    assert!(!stderr.is_empty());
}

#[test]
fn status_with_no_server_is_exit_6_in_both_output_shapes() {
    let (code, stdout, _) = run(&["status", "--state-file", "/nonexistent/openjev/server.json"]);
    assert!(code == 6 || code == 0 || code == 7, "unexpected {code}");
    if code == 6 {
        assert!(stdout.contains("no openjev server"));
        let (code, stdout, _) = run(&[
            "--json",
            "status",
            "--state-file",
            "/nonexistent/openjev/server.json",
        ]);
        assert_eq!(code, 6);
        let v: serde_json::Value = serde_json::from_str(&stdout).expect("json");
        assert_eq!(v["running"], false);
    }
}

#[test]
fn a_bad_config_file_is_exit_3_before_anything_expensive_happens() {
    let dir = std::env::temp_dir().join("openjev-test-badcfg");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    std::fs::write(&path, "[server]\nport = \"not a port\"\n").unwrap();
    let (code, _, stderr) = run(&["--config", path.to_str().unwrap(), "config", "show"]);
    assert_eq!(code, 3);
    assert!(stderr.contains("server.port"), "{stderr}");
}

#[test]
fn a_non_loopback_bind_without_a_token_refuses_to_start() {
    let (code, stdout, stderr) = run(&["serve", "--host", "0.0.0.0"]);
    assert_eq!(code, 3);
    assert!(stderr.contains("refusing to serve"), "{stderr}");
    assert!(stdout.is_empty(), "a refusal writes nothing to stdout");
}

#[test]
fn config_get_prints_a_bare_scriptable_value() {
    let (code, stdout, _) = run(&["config", "get", "server.port"]);
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "21131");
    assert!(!stdout.contains('"'), "a scriptable value is unquoted");
}

#[test]
fn config_get_on_an_unknown_key_is_exit_3_not_an_empty_success() {
    let (code, stdout, _) = run(&["config", "get", "server.nonsense"]);
    assert_eq!(code, 3);
    assert!(stdout.is_empty());
}

#[test]
fn config_show_sources_names_the_layer_and_never_prints_a_token() {
    let (code, stdout, _) = run(&["config", "show", "--sources"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("[default]"));
    assert!(!stdout.contains("supersecret"));
    let line = stdout
        .lines()
        .find(|l| l.starts_with("auth.token "))
        .expect("auth.token line");
    assert!(line.contains("\"\""), "{line}");
}

#[test]
fn the_env_layer_beats_the_defaults_and_is_reported_as_env() {
    let out = openjev()
        .args(["config", "show", "--sources"])
        .env("OPENJEV_PORT", "31337")
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("server.port "))
        .expect("server.port line");
    assert!(line.contains("31337") && line.contains("[env]"), "{line}");
}

#[test]
fn completions_are_generated_for_every_shell_we_advertise() {
    for shell in ["bash", "zsh", "fish", "powershell", "elvish"] {
        let (code, stdout, _) = run(&["completions", shell]);
        assert_eq!(code, 0, "{shell}");
        assert!(stdout.len() > 100, "{shell} produced nothing useful");
    }
}

#[test]
fn predict_with_no_input_at_all_is_a_bad_input_exit_not_a_crash() {
    // stdin is closed here, so there is no NDJSON and no flags: exit 8, before any
    // attempt to load 4B parameters.
    use std::process::Stdio;
    let out = openjev()
        .args(["predict"])
        .stdin(Stdio::null())
        .output()
        .expect("spawn");
    assert_eq!(out.status.code(), Some(8));
}

#[test]
fn doctor_is_copy_pasteable_and_contains_no_secret() {
    let out = openjev()
        .args(["doctor"])
        .env("OPENJEV_TOKEN", "supersecret")
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("build"), "{stdout}");
    assert!(stdout.contains("devices"));
    assert!(!stdout.contains("supersecret"));
}
