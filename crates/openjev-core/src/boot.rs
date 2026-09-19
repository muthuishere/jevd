//! Boot: registry -> backend selection -> device resolution -> weights -> session.
//!
//! Every failure here is loud and names the fix. The one thing this module must never do
//! is succeed with different weights than were asked for.

use crate::backend::{OpenRequest, factories};
use crate::device::{self, Device, DeviceRequest, Dtype, HostProbe};
use crate::error::Result;
use crate::head::Head;
use crate::hub::{Hub, ProgressFn};
use crate::ops::Session;
use crate::registry::{Registry, select_backend};
use crate::tokenize::Encoder;

#[derive(Debug, Clone)]
pub struct BootOptions {
    pub model: Option<String>,
    pub device: DeviceRequest,
    pub dtype: Option<Dtype>,
    pub context: Option<usize>,
    pub n_threads: Option<usize>,
    pub offline: bool,
}

impl Default for BootOptions {
    fn default() -> Self {
        Self {
            model: None,
            device: DeviceRequest::Auto,
            dtype: None,
            context: None,
            n_threads: None,
            offline: false,
        }
    }
}

/// Warnings the caller logs. Core does not print.
#[derive(Debug, Clone, Default)]
pub struct BootReport {
    pub warnings: Vec<String>,
    pub demotions: Vec<String>,
}

pub fn boot(
    registry: &Registry,
    opts: &BootOptions,
    progress: ProgressFn<'_>,
) -> Result<(Session, BootReport)> {
    let mut report = BootReport::default();
    let spec = registry.resolve(opts.model.as_deref())?.clone();

    if spec.revision_is_floating() {
        report.warnings.push(format!(
            "model '{}' pins revision '{}', which is not a commit sha — reproducibility is \
             the point of pinning",
            spec.id, spec.revision
        ));
    }

    let available: Vec<(&str, &'static [&'static str])> = factories()
        .iter()
        .map(|f| (f.name(), f.provides()))
        .collect();

    // A backend is only a candidate if it can serve *some* device this host has. The
    // precise device is resolved after selection, against that backend's support.
    let probe = HostProbe;
    let device_ok = |name: &str| {
        factories()
            .iter()
            .find(|f| f.name() == name)
            .is_some_and(|f| {
                [Device::Cuda(0), Device::Metal, Device::Vulkan, Device::Cpu]
                    .into_iter()
                    .any(|d| device::DeviceProbe::present(&probe, d) && f.supports_device(d))
            })
    };
    let (backend_name, backend_spec) = select_backend(&spec, &available, &device_ok)?;

    let factory = crate::backend::factory_named(backend_name).ok_or_else(|| {
        crate::error::JevError::Model(format!("backend '{backend_name}' vanished"))
    })?;

    let resolved = device::resolve(
        opts.device,
        &|d| factory.supports_device(d),
        &probe,
        opts.dtype
            .or_else(|| backend_spec.dtype_label.as_deref().and_then(label_to_dtype)),
    )?;
    report.demotions = resolved.demotions.clone();

    let hub = Hub::from_env().offline(opts.offline);
    let weights = hub.get_with_progress(&backend_spec.weights, &spec.revision, progress)?;
    let tokenizer_file = crate::registry::HubFile {
        repo: spec.tokenizer.repo.clone(),
        file: spec.tokenizer.file.clone(),
        revision: spec.tokenizer.revision.clone(),
        sha256: String::new(),
        size_bytes: None,
    };
    let tokenizer_path = hub.get_with_progress(&tokenizer_file, &spec.revision, progress)?;
    let head_path = hub.get_with_progress(&spec.head.file(), &spec.revision, progress)?;

    let encoder = Encoder::from_file(&spec, &tokenizer_path)?;
    let head = Head::load(&spec.head, &head_path)?;

    let req = OpenRequest {
        spec: spec.clone(),
        weights,
        device: resolved.device,
        dtype: resolved.dtype,
        context: opts.context.unwrap_or(spec.context),
        n_threads: opts.n_threads,
    };
    let backend = factory.open(&req)?;

    Ok((Session::new(spec, encoder, head, backend)?, report))
}

fn label_to_dtype(label: &str) -> Option<Dtype> {
    Some(match label {
        "f32" => Dtype::F32,
        "f16" => Dtype::F16,
        "bf16" => Dtype::Bf16,
        "q4_k_m" => Dtype::Quant("q4_k_m"),
        "q5_k_m" => Dtype::Quant("q5_k_m"),
        "q8_0" => Dtype::Quant("q8_0"),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_binary_with_no_backends_refuses_at_boot_naming_the_feature() {
        // With no backend feature compiled in, `factories()` is empty and boot must fail
        // with exit 78 rather than reach the network.
        if !factories().is_empty() {
            return;
        }
        let reg = Registry::builtin().expect("builtin");
        let mut noop = |_: &str, _: crate::hub::Progress| {};
        let Err(err) = boot(&reg, &BootOptions::default(), &mut noop) else {
            panic!("boot must not succeed with no backends compiled in");
        };
        assert_eq!(err.exit_code(), 78);
        assert!(
            err.to_string().contains("--features backend-llamacpp"),
            "{err}"
        );
    }

    #[test]
    fn dtype_labels_round_trip() {
        assert_eq!(label_to_dtype("q5_k_m"), Some(Dtype::Quant("q5_k_m")));
        assert_eq!(label_to_dtype("f16"), Some(Dtype::F16));
        assert_eq!(label_to_dtype("nonsense"), None);
    }
}
