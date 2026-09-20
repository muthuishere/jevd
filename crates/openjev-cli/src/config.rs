//! Layered config with provenance: flags > env > file > defaults (design 02 §5.1).
//!
//! Config is a flat map of dotted keys to TOML values, merged layer by layer, remembering
//! **which layer last wrote each key**. That flatness is the whole trick: `config get
//! server.port`, `config set`, and `config show --sources` are then one lookup each
//! instead of three tree walks, and provenance is a by-product rather than a feature.
//!
//! Organised against the single most common support question — "why is it using CPU" —
//! which is unanswerable unless the tool can name the layer that decided.

use crate::exit::{CliError, CliResult};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    Default,
    File,
    Env,
    Flag,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Source::Default => "default",
            Source::File => "file",
            Source::Env => "env",
            Source::Flag => "flag",
        })
    }
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub value: toml::Value,
    pub source: Source,
}

/// The merged config, still in key/value form.
#[derive(Debug, Clone, Default)]
pub struct Layered {
    map: BTreeMap<String, Entry>,
    /// Keys present in the file that the defaults do not know. Preserved and ignored — a
    /// newer config must not break an older binary mid-rollout — and reported by
    /// `config validate`.
    pub unknown: Vec<String>,
    pub file_path: PathBuf,
    pub file_found: bool,
}

pub fn default_config_path() -> PathBuf {
    // ~/.config/openjev/config.toml unconditionally (§5.2). Not XDG-chased: one tool, one
    // path, or you get two configs depending on how the process was started.
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("openjev")
        .join("config.toml")
}

/// Every key the binary knows, with its built-in value. This table *is* the schema:
/// `config set` refuses keys that are not here, and the type of the default is the type
/// the value must parse as.
pub fn defaults() -> BTreeMap<String, toml::Value> {
    use toml::Value as V;
    let mut m = BTreeMap::new();
    let mut set = |k: &str, v: V| {
        m.insert(k.to_string(), v);
    };
    set("server.host", V::String("127.0.0.1".into()));
    set("server.port", V::Integer(crate::cli::DEFAULT_PORT as i64));
    set("server.max_queue", V::Integer(64));
    set("server.max_batch", V::Integer(32));
    // Sequences the backend decodes in one graph. 1 keeps the shipping behaviour: one
    // sequence per forward, the full `model.context` available to every pair. Raising it
    // is worth roughly 3x at short NLI lengths and divides the context budget by the same
    // number — which is why it is opt-in rather than inherited from `max_batch`.
    set("server.max_seqs", V::Integer(1));
    set("server.max_body_bytes", V::Integer(1_048_576));
    set("server.max_pairs", V::Integer(256));
    set("server.max_options", V::Integer(512));
    set("server.max_field_chars", V::Integer(32_768));
    // /v1/systemone. 255 criteria is TypeSafe's own documented cap on a choice, so a
    // request they accept is a request we accept. The question cap is ours: there is no
    // documented one upstream, and an unbounded question map is an unbounded number of
    // forward passes behind a single admission slot.
    set("server.max_questions", V::Integer(32));
    set("server.max_criteria", V::Integer(255));
    set("server.request_timeout_secs", V::Integer(60));
    set("server.shutdown_grace_secs", V::Integer(20));
    set("server.cors_origins", V::Array(vec![]));
    set("server.metrics", V::Boolean(true));
    set("server.enable_latents", V::Boolean(false));
    set("auth.token", V::String(String::new()));
    set("auth.token_file", V::String(String::new()));
    set("log.level", V::String("info".into()));
    set("log.format", V::String("auto".into()));
    // [model] and [device] are 01's schema. We carry the keys we must pass through and
    // define none of their semantics.
    set("model.id", V::String(String::new()));
    set("model.revision", V::String(String::new()));
    set("model.cache_dir", V::String(String::new()));
    set("device.kind", V::String("auto".into()));
    m
}

/// `OPENJEV_PORT` and friends. Aliases win over their `__` long forms; that asymmetry is
/// documented in design 02 §5.1 and nowhere else.
const ENV_ALIASES: &[(&str, &str)] = &[
    ("OPENJEV_PORT", "server.port"),
    ("OPENJEV_HOST", "server.host"),
    ("OPENJEV_MODEL", "model.id"),
    ("OPENJEV_DEVICE", "device.kind"),
    ("OPENJEV_CACHE_DIR", "model.cache_dir"),
];

impl Layered {
    pub fn load(explicit_file: Option<&Path>, env: &BTreeMap<String, String>) -> CliResult<Self> {
        let path = explicit_file
            .map(Path::to_path_buf)
            .unwrap_or_else(default_config_path);
        let mut cfg = Layered {
            file_path: path.clone(),
            ..Default::default()
        };
        for (k, v) in defaults() {
            cfg.map.insert(
                k,
                Entry {
                    value: v,
                    source: Source::Default,
                },
            );
        }

        if path.is_file() {
            cfg.file_found = true;
            let text = std::fs::read_to_string(&path)
                .map_err(|e| CliError::config(format!("{}: {e}", path.display())))?;
            cfg.merge_toml(&text, &path)?;
        } else if explicit_file.is_some() {
            return Err(CliError::config(format!(
                "config file {} does not exist",
                path.display()
            )));
        }

        cfg.merge_env(env)?;
        Ok(cfg)
    }

    pub fn merge_toml(&mut self, text: &str, origin: &Path) -> CliResult<()> {
        let parsed: toml::Table = toml::from_str(text)
            .map_err(|e| CliError::config(format!("{}: {e}", origin.display())))?;
        let mut flat = BTreeMap::new();
        flatten("", &toml::Value::Table(parsed), &mut flat);
        let known = defaults();
        for (k, v) in flat {
            if !known.contains_key(&k) {
                self.unknown.push(k);
                continue;
            }
            self.put(&k, v, Source::File)?;
        }
        Ok(())
    }

    fn merge_env(&mut self, env: &BTreeMap<String, String>) -> CliResult<()> {
        let known = defaults();
        for (name, raw) in env {
            let Some(rest) = name.strip_prefix("OPENJEV_") else {
                continue;
            };
            if !rest.contains("__") {
                continue;
            }
            let key = rest.to_ascii_lowercase().replace("__", ".");
            if !known.contains_key(&key) {
                self.unknown.push(format!("{name} (env)"));
                continue;
            }
            let v = coerce(&key, raw, &known)?;
            self.put(&key, v, Source::Env)?;
        }
        // Aliases last: they win over the long form.
        for (name, key) in ENV_ALIASES {
            if let Some(raw) = env.get(*name) {
                let v = coerce(key, raw, &known)?;
                self.put(key, v, Source::Env)?;
            }
        }
        Ok(())
    }

    /// Flag layer. `None` means "the user did not type it" and must not overwrite.
    pub fn set_flag(&mut self, key: &str, value: Option<toml::Value>) {
        if let Some(v) = value {
            let _ = self.put(key, v, Source::Flag);
        }
    }

    fn put(&mut self, key: &str, value: toml::Value, source: Source) -> CliResult<()> {
        if let Some(expected) = defaults().get(key)
            && std::mem::discriminant(expected) != std::mem::discriminant(&value)
        {
            return Err(CliError::config(format!(
                "{key}: expected {}, got {}",
                expected.type_str(),
                value.type_str()
            )));
        }
        self.map.insert(key.to_string(), Entry { value, source });
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&Entry> {
        self.map.get(key)
    }

    pub fn source_of(&self, key: &str) -> Source {
        self.map.get(key).map_or(Source::Default, |e| e.source)
    }

    pub fn string(&self, key: &str) -> String {
        match self.map.get(key).map(|e| &e.value) {
            Some(toml::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => String::new(),
        }
    }

    pub fn int(&self, key: &str) -> i64 {
        match self.map.get(key).map(|e| &e.value) {
            Some(toml::Value::Integer(i)) => *i,
            _ => 0,
        }
    }

    pub fn usize(&self, key: &str) -> usize {
        self.int(key).max(0) as usize
    }

    pub fn bool(&self, key: &str) -> bool {
        matches!(
            self.map.get(key).map(|e| &e.value),
            Some(toml::Value::Boolean(true))
        )
    }

    pub fn strings(&self, key: &str) -> Vec<String> {
        match self.map.get(key).map(|e| &e.value) {
            Some(toml::Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Entry)> {
        self.map.iter()
    }

    /// Non-empty string, or None. Blank in the file means "unset", not "empty value".
    pub fn opt_string(&self, key: &str) -> Option<String> {
        let s = self.string(key);
        (!s.is_empty()).then_some(s)
    }
}

fn flatten(prefix: &str, v: &toml::Value, out: &mut BTreeMap<String, toml::Value>) {
    match v {
        toml::Value::Table(t) => {
            for (k, v) in t {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten(&key, v, out);
            }
        }
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

/// Env and CLI both hand us strings; the default's type says what to parse into.
pub fn coerce(
    key: &str,
    raw: &str,
    known: &BTreeMap<String, toml::Value>,
) -> CliResult<toml::Value> {
    let want = known
        .get(key)
        .ok_or_else(|| CliError::config(format!("unknown config key '{key}'")))?;
    Ok(match want {
        toml::Value::Integer(_) => toml::Value::Integer(
            raw.parse()
                .map_err(|_| CliError::config(format!("{key}: '{raw}' is not an integer")))?,
        ),
        toml::Value::Boolean(_) => toml::Value::Boolean(match raw {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => return Err(CliError::config(format!("{key}: '{raw}' is not a boolean"))),
        }),
        toml::Value::Array(_) => toml::Value::Array(
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| toml::Value::String(s.to_string()))
                .collect(),
        ),
        _ => toml::Value::String(raw.to_string()),
    })
}

/// The commented file `config edit` creates. Generated from the defaults so it cannot
/// drift from them.
pub fn commented_defaults() -> String {
    let d = defaults();
    let mut out = String::from("# ~/.config/openjev/config.toml\n");
    let mut section = "";
    for (k, v) in &d {
        let (sec, leaf) = k.split_once('.').unwrap_or(("", k.as_str()));
        if sec != section {
            out.push_str(&format!("\n[{sec}]\n"));
            section = sec;
        }
        out.push_str(&format!("{leaf} = {v}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn write(dir: &tempfile::TempDir, body: &str) -> PathBuf {
        let p = dir.path().join("config.toml");
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn precedence_is_flags_over_env_over_file_over_defaults() {
        let d = tempfile::tempdir().unwrap();
        let p = write(&d, "[server]\nport = 1111\n");

        let c = Layered::load(Some(&p), &BTreeMap::new()).unwrap();
        assert_eq!(c.int("server.port"), 1111);
        assert_eq!(c.source_of("server.port"), Source::File);

        let c = Layered::load(Some(&p), &env(&[("OPENJEV_SERVER__PORT", "2222")])).unwrap();
        assert_eq!(c.int("server.port"), 2222);
        assert_eq!(c.source_of("server.port"), Source::Env);

        let mut c = Layered::load(Some(&p), &env(&[("OPENJEV_SERVER__PORT", "2222")])).unwrap();
        c.set_flag("server.port", Some(toml::Value::Integer(3333)));
        assert_eq!(c.int("server.port"), 3333);
        assert_eq!(c.source_of("server.port"), Source::Flag);
    }

    #[test]
    fn an_unset_flag_does_not_overwrite_the_file() {
        let d = tempfile::tempdir().unwrap();
        let p = write(&d, "[server]\nport = 1111\n");
        let mut c = Layered::load(Some(&p), &BTreeMap::new()).unwrap();
        c.set_flag("server.port", None);
        assert_eq!(c.int("server.port"), 1111);
        assert_eq!(c.source_of("server.port"), Source::File);
    }

    #[test]
    fn the_short_alias_beats_the_long_env_form() {
        let c = Layered::load(
            None,
            &env(&[("OPENJEV_SERVER__PORT", "2222"), ("OPENJEV_PORT", "4444")]),
        )
        .unwrap();
        assert_eq!(c.int("server.port"), 4444);
    }

    #[test]
    fn unknown_keys_are_preserved_and_ignored_not_fatal() {
        let d = tempfile::tempdir().unwrap();
        let p = write(&d, "[server]\nport = 1111\nfrom_the_future = 9\n");
        let c = Layered::load(Some(&p), &BTreeMap::new()).unwrap();
        assert_eq!(c.unknown, vec!["server.from_the_future"]);
        assert_eq!(c.int("server.port"), 1111);
    }

    #[test]
    fn a_wrongly_typed_key_is_a_config_error_not_a_coercion() {
        let d = tempfile::tempdir().unwrap();
        let p = write(&d, "[server]\nport = \"twenty\"\n");
        let e = Layered::load(Some(&p), &BTreeMap::new()).unwrap_err();
        assert_eq!(e.exit_code(), crate::exit::CONFIG);
    }

    #[test]
    fn a_missing_explicit_config_file_is_an_error_but_a_missing_default_one_is_not() {
        let missing = PathBuf::from("/nonexistent/openjev/config.toml");
        assert!(Layered::load(Some(&missing), &BTreeMap::new()).is_err());
        let c = Layered::load(None, &BTreeMap::new()).unwrap();
        assert_eq!(c.int("server.port"), crate::cli::DEFAULT_PORT as i64);
    }

    #[test]
    fn the_generated_file_parses_back_into_the_same_defaults() {
        let d = tempfile::tempdir().unwrap();
        let p = write(&d, &commented_defaults());
        let c = Layered::load(Some(&p), &BTreeMap::new()).unwrap();
        assert!(c.unknown.is_empty(), "{:?}", c.unknown);
        assert_eq!(c.int("server.max_queue"), 64);
    }
}
