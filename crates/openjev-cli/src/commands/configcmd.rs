//! `openjev config {path,show,get,set,edit,validate}`.
//!
//! `show --sources` is the point of the whole config module: it names the layer that set
//! each key, which is the only honest answer to "why is it using CPU".

use crate::cli::ConfigCmd;
use crate::config::{Layered, commented_defaults, default_config_path, defaults};
use crate::exit::{self, CliError, CliResult};
use std::path::PathBuf;

pub fn run(
    cmd: &ConfigCmd,
    cfg: &Layered,
    explicit: Option<&PathBuf>,
    json: bool,
) -> CliResult<i32> {
    // The path the layered config actually used, so `config path` cannot disagree with
    // the file `config show` read.
    let path = explicit.cloned().unwrap_or_else(|| {
        let p = cfg.file_path.clone();
        if p.as_os_str().is_empty() {
            default_config_path()
        } else {
            p
        }
    });
    match cmd {
        ConfigCmd::Path => {
            println!("{}", path.display());
            Ok(exit::OK)
        }

        ConfigCmd::Show { sources } => {
            if json {
                let map: serde_json::Map<String, serde_json::Value> = cfg
                    .iter()
                    .map(|(k, e)| {
                        let v = serde_json::to_value(&e.value).unwrap_or(serde_json::Value::Null);
                        (
                            k.clone(),
                            if *sources {
                                serde_json::json!({"value": v, "source": e.source.to_string()})
                            } else {
                                v
                            },
                        )
                    })
                    .collect();
                println!("{}", serde_json::Value::Object(map));
            } else {
                for (k, e) in cfg.iter() {
                    let v = redact(k, &e.value);
                    if *sources {
                        println!("{k:<32} {v:<24} [{}]", e.source);
                    } else {
                        println!("{k:<32} {v}");
                    }
                }
            }
            Ok(exit::OK)
        }

        ConfigCmd::Get { key } => {
            let e = cfg
                .get(key)
                .ok_or_else(|| CliError::config(format!("unknown config key '{key}'")))?;
            // Bare, unquoted, one line: this is for `$(openjev config get server.port)`.
            match &e.value {
                toml::Value::String(s) => println!("{s}"),
                other => println!("{other}"),
            }
            Ok(exit::OK)
        }

        ConfigCmd::Set {
            key,
            value,
            allow_inline_token,
        } => {
            if key == "auth.token" && !allow_inline_token {
                return Err(CliError::config(
                    "refusing to write a token into the config file. Use auth.token_file, or \
                     pass --allow-inline-token if you accept a plaintext secret on disk.",
                ));
            }
            let v = crate::config::coerce(key, value, &defaults())?;
            let mut doc = read_or_seed(&path)?;
            set_in_toml(&mut doc, key, v)?;
            // Validate the whole file before it replaces the old one: never leave a
            // broken config behind.
            let text = doc.to_string();
            validate_text(&text, &path)?;
            write_file(&path, &text)?;
            Ok(exit::OK)
        }

        ConfigCmd::Edit => {
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());
            if !path.exists() {
                write_file(&path, &commented_defaults())?;
            }
            let tmp = path.with_extension("toml.editing");
            std::fs::copy(&path, &tmp)?;
            let status = std::process::Command::new(&editor)
                .arg(&tmp)
                .status()
                .map_err(|e| CliError::other(format!("{editor}: {e}")))?;
            if !status.success() {
                let _ = std::fs::remove_file(&tmp);
                return Err(CliError::other(format!("{editor} exited non-zero")));
            }
            let text = std::fs::read_to_string(&tmp)?;
            match validate_text(&text, &tmp) {
                Ok(_) => {
                    std::fs::rename(&tmp, &path)?;
                    Ok(exit::OK)
                }
                Err(e) => {
                    eprintln!("{e}");
                    eprintln!("your edit is kept at {}", tmp.display());
                    Ok(exit::CONFIG)
                }
            }
        }

        ConfigCmd::Validate { file } => {
            let target = file.clone().unwrap_or(path);
            if !target.exists() {
                println!("{}: absent — defaults are in force", target.display());
                return Ok(exit::OK);
            }
            let text = std::fs::read_to_string(&target)?;
            let unknown = validate_text(&text, &target)?;
            for k in &unknown {
                // Preserved and ignored, never fatal: a newer config must not break an
                // older binary mid-rollout.
                println!("warning: unknown key '{k}' (ignored)");
            }
            println!("{}: ok", target.display());
            Ok(exit::OK)
        }
    }
}

fn redact(key: &str, v: &toml::Value) -> String {
    if key == "auth.token" {
        return match v {
            toml::Value::String(s) if s.is_empty() => "\"\"".into(),
            _ => "\"<set>\"".into(),
        };
    }
    v.to_string()
}

fn validate_text(text: &str, origin: &std::path::Path) -> CliResult<Vec<String>> {
    let mut probe = Layered::default();
    for (k, v) in defaults() {
        probe.set_flag(&k, Some(v));
    }
    let mut fresh = Layered::default();
    fresh.merge_toml(text, origin)?;
    let _ = probe;
    Ok(fresh.unknown)
}

fn read_or_seed(path: &std::path::Path) -> CliResult<toml::Table> {
    if path.exists() {
        let text = std::fs::read_to_string(path)?;
        toml::from_str(&text).map_err(|e| CliError::config(format!("{}: {e}", path.display())))
    } else {
        Ok(toml::Table::new())
    }
}

fn set_in_toml(doc: &mut toml::Table, key: &str, value: toml::Value) -> CliResult<()> {
    let (section, leaf) = key
        .split_once('.')
        .ok_or_else(|| CliError::config(format!("'{key}' is not a section.key")))?;
    let entry = doc
        .entry(section.to_string())
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    match entry {
        toml::Value::Table(t) => {
            t.insert(leaf.to_string(), value);
            Ok(())
        }
        _ => Err(CliError::config(format!("[{section}] is not a table"))),
    }
}

fn write_file(path: &std::path::Path, text: &str) -> CliResult<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text).map_err(CliError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_is_never_printed_even_when_it_is_set() {
        let v = toml::Value::String("supersecret".into());
        assert_eq!(redact("auth.token", &v), "\"<set>\"");
        assert!(!redact("auth.token", &v).contains("supersecret"));
    }

    #[test]
    fn setting_a_key_creates_its_section_and_keeps_the_others() {
        let mut doc: toml::Table = toml::from_str("[log]\nlevel = \"debug\"\n").unwrap();
        set_in_toml(&mut doc, "server.port", toml::Value::Integer(2)).unwrap();
        let text = doc.to_string();
        assert!(text.contains("port = 2"), "{text}");
        assert!(text.contains("level = \"debug\""), "{text}");
    }

    #[test]
    fn validation_reports_unknown_keys_as_warnings_not_failures() {
        let unknown = validate_text(
            "[server]\nport = 1\nnew_thing = true\n",
            std::path::Path::new("t"),
        )
        .expect("valid");
        assert_eq!(unknown, vec!["server.new_thing"]);
    }

    #[test]
    fn validation_fails_on_a_wrong_type() {
        let e =
            validate_text("[server]\nport = \"nope\"\n", std::path::Path::new("t")).unwrap_err();
        assert_eq!(e.exit_code(), exit::CONFIG);
    }
}
