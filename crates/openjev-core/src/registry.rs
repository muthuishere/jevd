//! D2 — a model is a config entry, not code.
//!
//! An embedded TOML registry of known-good models, merged with
//! `$XDG_CONFIG_HOME/openjev/models.toml`. Arch, template, label map, tokenizer, head
//! shape and per-backend weights are all data. A 2-label entailment model, a different
//! prompt shape or a 7-label taxonomy is a TOML edit. The only thing that is code is a
//! new *head kind*.
//!
//! Organised against: the model id appearing in forty places, so that supporting a second
//! checkpoint means a release rather than a config line.

use crate::error::{BackendRejection, JevError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const BUILTIN: &str = include_str!("models.toml");

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HubFile {
    pub repo: String,
    pub file: String,
    #[serde(default)]
    pub revision: Option<String>,
    /// Empty means unpinned. Unpinned is allowed (so a local build can move fast) and
    /// warned about at boot, because "we shipped an unverified 3 GB blob" must not be
    /// silent.
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct TokenizerSpec {
    pub repo: String,
    pub file: String,
    #[serde(default)]
    pub revision: Option<String>,
    pub pad_token_id: u32,
    #[serde(default = "default_padding_side")]
    pub padding_side: PaddingSide,
}

fn default_padding_side() -> PaddingSide {
    PaddingSide::Left
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PaddingSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase", tag = "kind")]
pub enum HeadSpec {
    /// `Linear(in -> out)`, optionally with bias. The whole classification head.
    Linear {
        in_features: usize,
        out_features: usize,
        #[serde(default)]
        bias: bool,
        tensor: String,
        repo: String,
        file: String,
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        sha256: String,
    },
}

impl HeadSpec {
    pub fn out_features(&self) -> usize {
        match self {
            HeadSpec::Linear { out_features, .. } => *out_features,
        }
    }
    pub fn in_features(&self) -> usize {
        match self {
            HeadSpec::Linear { in_features, .. } => *in_features,
        }
    }
    pub fn file(&self) -> HubFile {
        match self {
            HeadSpec::Linear {
                repo,
                file,
                revision,
                sha256,
                ..
            } => HubFile {
                repo: repo.clone(),
                file: file.clone(),
                revision: revision.clone(),
                sha256: sha256.clone(),
                size_bytes: None,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct BackendSpec {
    pub weights: HubFile,
    /// Capability tokens the backend must advertise. Strings so a new architecture is a
    /// registry edit, not a core release.
    #[serde(default)]
    pub requires: Vec<String>,
    /// Label for the trunk dtype on this path (the quantisation, for GGUF). Informational
    /// — it goes in the banner.
    #[serde(default)]
    pub dtype_label: Option<String>,
    /// Pinned native runtime build, if this backend loads one.
    #[serde(default)]
    pub native_release: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelSpec {
    #[serde(skip)]
    pub id: String,
    #[serde(default)]
    pub default: bool,
    pub repo: String,
    #[serde(default = "default_revision")]
    pub revision: String,
    #[serde(default)]
    pub subfolder: Option<String>,
    pub arch: String,
    #[serde(default)]
    pub task: Option<String>,
    pub template: String,
    pub labels: Vec<String>,
    /// Which label means "the hypothesis follows". `rerank` and `grade` are defined in
    /// terms of this, so there is one place to be wrong instead of three.
    pub entailment_label: String,
    pub context: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub modalities: Vec<String>,
    pub tokenizer: TokenizerSpec,
    pub head: HeadSpec,
    /// Declaration order is preference order. TOML tables do not preserve order, so this
    /// is a map plus an explicit `order`.
    pub backends: BTreeMap<String, BackendSpec>,
    #[serde(default)]
    pub backend_order: Vec<String>,
}

fn default_revision() -> String {
    "main".into()
}

impl ModelSpec {
    /// Index of the entailment label in `labels`.
    pub fn entailment_index(&self) -> Result<usize> {
        self.labels
            .iter()
            .position(|l| l == &self.entailment_label)
            .ok_or_else(|| {
                JevError::Model(format!(
                    "model '{}': entailment_label '{}' is not in labels {:?}",
                    self.id, self.entailment_label, self.labels
                ))
            })
    }

    /// Preference order over backends: `backend_order` first (for the entries it names),
    /// then everything else alphabetically, so a registry that forgets `backend_order`
    /// is still deterministic.
    pub fn ordered_backends(&self) -> Vec<(&str, &BackendSpec)> {
        let mut out: Vec<(&str, &BackendSpec)> = Vec::new();
        for name in &self.backend_order {
            if let Some((k, v)) = self.backends.get_key_value(name.as_str()) {
                out.push((k.as_str(), v));
            }
        }
        for (k, v) in &self.backends {
            if !out.iter().any(|(n, _)| *n == k.as_str()) {
                out.push((k.as_str(), v));
            }
        }
        out
    }

    pub fn validate(&self) -> Result<()> {
        if self.labels.is_empty() {
            return Err(JevError::Model(format!("model '{}': no labels", self.id)));
        }
        if self.labels.len() != self.head.out_features() {
            return Err(JevError::Model(format!(
                "model '{}': {} labels but the head has {} outputs — the label map and the \
                 head must agree or every answer is mislabelled",
                self.id,
                self.labels.len(),
                self.head.out_features()
            )));
        }
        if self.head.in_features() != self.hidden_size {
            return Err(JevError::Model(format!(
                "model '{}': head expects {}-d input but hidden_size is {}",
                self.id,
                self.head.in_features(),
                self.hidden_size
            )));
        }
        self.entailment_index()?;
        if !self.template.contains("{premise}") || !self.template.contains("{hypothesis}") {
            return Err(JevError::Model(format!(
                "model '{}': template must contain {{premise}} and {{hypothesis}}",
                self.id
            )));
        }
        if self.backends.is_empty() {
            return Err(JevError::Model(format!(
                "model '{}': no backends declared",
                self.id
            )));
        }
        if self.modalities.iter().any(|m| m != "text") {
            return Err(JevError::Model(format!(
                "model '{}': v0.1 is text-only; images are a NotSupported capability, not a \
                 silent wrong answer",
                self.id
            )));
        }
        Ok(())
    }

    /// True if the revision is a branch rather than a commit sha. Reproducibility is the
    /// point of pinning, so this earns a warning at boot.
    pub fn revision_is_floating(&self) -> bool {
        !(self.revision.len() == 40 && self.revision.chars().all(|c| c.is_ascii_hexdigit()))
    }
}

#[derive(Debug, Deserialize)]
struct RegistryFile {
    #[serde(default)]
    models: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Clone, Default)]
pub struct Registry {
    models: BTreeMap<String, ModelSpec>,
}

impl Registry {
    /// Built-in registry only. Never fails in a shipped binary — a broken embedded TOML
    /// is a build-time bug, so it is a test, not a runtime branch.
    pub fn builtin() -> Result<Self> {
        let mut r = Registry::default();
        r.merge_str(BUILTIN, Path::new("<builtin>"))?;
        Ok(r)
    }

    /// Built-in, then the user file if it exists. User wins, whole entry at a time.
    pub fn load(user_file: Option<&Path>) -> Result<Self> {
        let mut r = Registry::builtin()?;
        let path = match user_file {
            Some(p) => Some(p.to_path_buf()),
            None => default_user_file(),
        };
        if let Some(p) = path.filter(|p| p.is_file()) {
            let text = std::fs::read_to_string(&p).map_err(|e| JevError::io(&p, e))?;
            r.merge_str(&text, &p)?;
        }
        Ok(r)
    }

    pub fn merge_str(&mut self, text: &str, origin: &Path) -> Result<()> {
        let parsed: RegistryFile = toml::from_str(text).map_err(|e| JevError::Config {
            path: origin.to_path_buf(),
            message: e.to_string(),
        })?;
        for (id, value) in parsed.models {
            let mut spec: ModelSpec = value.clone().try_into().map_err(|e| JevError::Config {
                path: origin.to_path_buf(),
                message: format!("models.\"{id}\": {e}"),
            })?;
            spec.id = id.clone();
            spec.validate()?;
            self.models.insert(id, spec);
        }
        Ok(())
    }

    pub fn ids(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ModelSpec> {
        self.models.values()
    }

    pub fn get(&self, id: &str) -> Result<&ModelSpec> {
        self.models.get(id).ok_or_else(|| JevError::ModelNotFound {
            id: id.to_string(),
            known: self.ids(),
        })
    }

    /// The entry marked `default = true`, or the only entry if there is exactly one.
    pub fn default_model(&self) -> Result<&ModelSpec> {
        if let Some(m) = self.models.values().find(|m| m.default) {
            return Ok(m);
        }
        if self.models.len() == 1 {
            return Ok(self.models.values().next().expect("len == 1"));
        }
        Err(JevError::Model(
            "no model is marked default = true and there is more than one; pass --model".into(),
        ))
    }

    pub fn resolve(&self, id: Option<&str>) -> Result<&ModelSpec> {
        match id {
            Some(id) => self.get(id),
            None => self.default_model(),
        }
    }
}

pub fn default_user_file() -> Option<PathBuf> {
    // CLI convention beats Apple convention: ~/.config/openjev, not
    // ~/Library/Application Support. Organised against a Mac user having to learn a
    // second location for the same file they already edit on Linux.
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(x).join("openjev").join("models.toml"));
    }
    dirs::home_dir().map(|h| h.join(".config").join("openjev").join("models.toml"))
}

/// Walk a model's backends in preference order and keep the first that is compiled in,
/// advertises everything the entry requires, and can use the resolved device.
///
/// Nothing matches => a hard error naming the exact reason per candidate. Falling back
/// silently would make `predict` return confident wrong labels from the wrong weights.
pub fn select_backend<'a>(
    spec: &'a ModelSpec,
    available: &[(&str, &'static [&'static str])],
    device_ok: &dyn Fn(&str) -> bool,
) -> Result<(&'a str, &'a BackendSpec)> {
    let mut rejections = Vec::new();
    for (name, bspec) in spec.ordered_backends() {
        let Some((_, provides)) = available.iter().find(|(n, _)| *n == name) else {
            rejections.push(BackendRejection {
                backend: name.to_string(),
                reason: format!("not compiled in (build with --features backend-{name})"),
            });
            continue;
        };
        if let Some(missing) = bspec
            .requires
            .iter()
            .find(|r| !provides.iter().any(|p| p == &r.as_str()))
        {
            rejections.push(BackendRejection {
                backend: name.to_string(),
                reason: format!("present, but lacks capability '{missing}'"),
            });
            continue;
        }
        if !device_ok(name) {
            rejections.push(BackendRejection {
                backend: name.to_string(),
                reason: "present, but supports no device this host has".into(),
            });
            continue;
        }
        return Ok((name, bspec));
    }
    Err(JevError::BackendUnavailable {
        model: spec.id.clone(),
        candidates: rejections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg() -> Registry {
        Registry::builtin().expect("embedded models.toml must parse and validate")
    }

    #[test]
    fn builtin_registry_parses_and_validates() {
        let r = reg();
        let m = r.default_model().unwrap();
        assert_eq!(m.id, "openjev-4b-nli-v2");
        assert_eq!(m.labels, ["contradiction", "entailment", "neutral"]);
        assert_eq!(m.entailment_index().unwrap(), 1);
        assert_eq!(m.tokenizer.pad_token_id, 248044);
        assert_eq!(m.tokenizer.padding_side, PaddingSide::Left);
        assert_eq!(m.hidden_size, 2560);
        assert!(m.backends.contains_key("llamacpp"));
    }

    #[test]
    fn template_newline_is_a_real_newline() {
        // The whole label distribution moves if this becomes a literal backslash-n.
        let r = reg();
        let m = r.default_model().unwrap();
        assert_eq!(m.template, "Premise: {premise}\nHypothesis: {hypothesis}");
        assert!(!m.template.contains(r"\n"));
    }

    #[test]
    fn user_entry_replaces_the_builtin_entry_wholesale() {
        let mut r = reg();
        let user = r#"
[models."openjev-4b-nli-v2"]
repo = "someone/else"
revision = "0123456789abcdef0123456789abcdef01234567"
arch = "qwen3_5"
template = "P: {premise} / H: {hypothesis}"
labels = ["no", "yes"]
entailment_label = "yes"
context = 4096
hidden_size = 8
tokenizer = { repo = "someone/else", file = "tokenizer.json", pad_token_id = 0 }
head = { kind = "linear", in_features = 8, out_features = 2, tensor = "score.weight", repo = "r", file = "f" }
[models."openjev-4b-nli-v2".backends.llamacpp]
weights = { repo = "r", file = "w.gguf" }
"#;
        r.merge_str(user, Path::new("user.toml")).unwrap();
        let m = r.get("openjev-4b-nli-v2").unwrap();
        assert_eq!(m.repo, "someone/else");
        assert_eq!(m.labels.len(), 2);
        assert!(!m.revision_is_floating());
    }

    #[test]
    fn label_count_must_match_the_head() {
        let mut r = Registry::default();
        let bad = r#"
[models.x]
repo = "a/b"
arch = "q"
template = "{premise}{hypothesis}"
labels = ["a", "b", "c"]
entailment_label = "a"
context = 8
hidden_size = 4
tokenizer = { repo = "a/b", file = "t.json", pad_token_id = 0 }
head = { kind = "linear", in_features = 4, out_features = 2, tensor = "score.weight", repo = "r", file = "f" }
[models.x.backends.llamacpp]
weights = { repo = "r", file = "w" }
"#;
        let e = r.merge_str(bad, Path::new("t.toml")).unwrap_err();
        assert!(e.to_string().contains("3 labels but the head has 2"));
    }

    #[test]
    fn entailment_label_must_exist() {
        let mut r = Registry::default();
        let bad = r#"
[models.x]
repo = "a/b"
arch = "q"
template = "{premise}{hypothesis}"
labels = ["a", "b"]
entailment_label = "yes"
context = 8
hidden_size = 4
tokenizer = { repo = "a/b", file = "t.json", pad_token_id = 0 }
head = { kind = "linear", in_features = 4, out_features = 2, tensor = "score.weight", repo = "r", file = "f" }
[models.x.backends.llamacpp]
weights = { repo = "r", file = "w" }
"#;
        assert!(r.merge_str(bad, Path::new("t.toml")).is_err());
    }

    #[test]
    fn missing_backend_fails_loudly_and_names_the_cargo_feature() {
        let r = reg();
        let m = r.default_model().unwrap();
        let err = select_backend(m, &[], &|_| true).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--features backend-llamacpp"), "{msg}");
        assert_eq!(err.exit_code(), 78);
    }

    #[test]
    fn present_backend_missing_a_capability_is_named_precisely() {
        let r = reg();
        let m = r.default_model().unwrap();
        let err = select_backend(m, &[("llamacpp", &[])], &|_| true).unwrap_err();
        assert!(
            err.to_string()
                .contains("lacks capability 'qwen3_5-hybrid'"),
            "{err}"
        );
    }

    #[test]
    fn a_capable_backend_is_selected() {
        let r = reg();
        let m = r.default_model().unwrap();
        let (name, _) = select_backend(m, &[("llamacpp", &["qwen3_5-hybrid"])], &|_| true).unwrap();
        assert_eq!(name, "llamacpp");
    }

    #[test]
    fn unknown_model_lists_what_is_known() {
        let err = reg().get("nope").unwrap_err();
        assert!(err.to_string().contains("openjev-4b-nli-v2"));
    }

    #[test]
    fn backend_order_is_deterministic_without_an_explicit_order() {
        let r = reg();
        let m = r.default_model().unwrap();
        let a: Vec<_> = m.ordered_backends().iter().map(|(n, _)| *n).collect();
        let b: Vec<_> = m.ordered_backends().iter().map(|(n, _)| *n).collect();
        assert_eq!(a, b);
    }
}
