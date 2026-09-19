//! Device and dtype resolution.
//!
//! D3 — device is detected, not configured; configuration only overrides.
//!
//! Organised against two distinct failures:
//!   * a silent CPU run at 40x the latency when the user typed `--device cuda`, and
//!   * a hard failure on a laptop with no GPU when the user asked for nothing at all.
//!
//! So: `auto` demotes down a logged chain, an explicit device never demotes.

use crate::error::{JevError, Result};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Metal,
    Cuda(u32),
    Vulkan,
}

impl Device {
    /// Family name, no ordinal. What the registry and `supports_device` match on.
    pub fn kind(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
            Device::Cuda(_) => "cuda",
            Device::Vulkan => "vulkan",
        }
    }

    pub fn is_gpu(self) -> bool {
        !matches!(self, Device::Cpu)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cuda(n) => write!(f, "cuda:{n}"),
            other => f.write_str(other.kind()),
        }
    }
}

/// `auto` is a *request*, not a device, so it is not a `Device` variant — that keeps
/// "unresolved" unrepresentable once resolution has run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    Auto,
    Explicit(Device),
}

impl FromStr for DeviceRequest {
    type Err = JevError;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        Ok(match s.as_str() {
            "auto" => DeviceRequest::Auto,
            "cpu" => DeviceRequest::Explicit(Device::Cpu),
            "metal" => DeviceRequest::Explicit(Device::Metal),
            "vulkan" => DeviceRequest::Explicit(Device::Vulkan),
            "cuda" => DeviceRequest::Explicit(Device::Cuda(0)),
            other => {
                let n = other
                    .strip_prefix("cuda:")
                    .and_then(|n| n.parse::<u32>().ok())
                    .ok_or_else(|| {
                        JevError::Device(format!(
                            "unknown device '{other}' (want auto | cpu | metal | cuda[:N] | vulkan)"
                        ))
                    })?;
                DeviceRequest::Explicit(Device::Cuda(n))
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    /// The trunk dtype on the GGUF path is the quantisation; it is data, not a choice we
    /// make at runtime, so it is carried as the label from the registry.
    Quant(&'static str),
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
            Dtype::Quant(q) => q,
        })
    }
}

/// Default dtype for a device.
///
/// Metal gets **f16, not bf16**: bf16 coverage on Metal is uneven across ops and macOS
/// versions, and the failure mode is wrong numbers rather than a missing-kernel error.
/// bf16 on Metal is opt-in only.
pub fn default_dtype(device: Device) -> Dtype {
    match device {
        Device::Cpu => Dtype::F32,
        Device::Metal => Dtype::F16,
        Device::Cuda(_) => Dtype::Bf16,
        Device::Vulkan => Dtype::F16,
    }
}

/// What the host actually has. A *probe*, not a compile-time feature: a binary built with
/// CUDA support on a machine with no GPU must resolve to cpu, not die.
pub trait DeviceProbe: Send + Sync {
    fn present(&self, device: Device) -> bool;
}

/// Host probe. Deliberately conservative — it answers "could this plausibly work", and
/// the backend's own `open` is the real oracle.
pub struct HostProbe;

impl DeviceProbe for HostProbe {
    fn present(&self, device: Device) -> bool {
        match device {
            Device::Cpu => true,
            Device::Metal => cfg!(all(target_os = "macos", target_arch = "aarch64")),
            Device::Cuda(_) => {
                cfg!(any(target_os = "linux", target_os = "windows"))
                    && (std::path::Path::new("/dev/nvidiactl").exists()
                        || std::env::var_os("CUDA_PATH").is_some())
            }
            Device::Vulkan => cfg!(any(target_os = "linux", target_os = "windows")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Resolved {
    pub device: Device,
    pub dtype: Dtype,
    /// Every demotion, in order, with its reason. The caller logs these at warn — core
    /// never prints.
    pub demotions: Vec<String>,
}

/// `auto` probes cuda -> metal -> vulkan -> cpu and keeps the first that the host has and
/// the backend supports. An explicit device that fails is a hard error.
pub fn resolve(
    request: DeviceRequest,
    supported: &dyn Fn(Device) -> bool,
    probe: &dyn DeviceProbe,
    dtype_override: Option<Dtype>,
) -> Result<Resolved> {
    let mut demotions = Vec::new();

    let device = match request {
        DeviceRequest::Explicit(d) => {
            if !probe.present(d) {
                return Err(JevError::Device(format!(
                    "device '{d}' was requested explicitly but this host does not have it"
                )));
            }
            if !supported(d) {
                return Err(JevError::Device(format!(
                    "device '{d}' was requested explicitly but the selected backend cannot use it"
                )));
            }
            d
        }
        DeviceRequest::Auto => {
            const CHAIN: [Device; 4] =
                [Device::Cuda(0), Device::Metal, Device::Vulkan, Device::Cpu];
            let mut chosen = None;
            for d in CHAIN {
                if !probe.present(d) {
                    demotions.push(format!("{d}: not present on this host"));
                    continue;
                }
                if !supported(d) {
                    demotions.push(format!(
                        "{d}: present, but the backend has no kernel for it"
                    ));
                    continue;
                }
                chosen = Some(d);
                break;
            }
            chosen.ok_or_else(|| {
                JevError::Device(
                    "no usable device, not even cpu — the backend supports nothing this host has"
                        .into(),
                )
            })?
        }
    };

    Ok(Resolved {
        device,
        dtype: dtype_override.unwrap_or_else(|| default_dtype(device)),
        demotions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(Vec<Device>);
    impl DeviceProbe for Fake {
        fn present(&self, d: Device) -> bool {
            self.0.iter().any(|x| x.kind() == d.kind())
        }
    }

    fn all(_: Device) -> bool {
        true
    }

    #[test]
    fn parses_device_strings() {
        assert_eq!(
            "auto".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Auto
        );
        assert_eq!(
            "CUDA:3".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Explicit(Device::Cuda(3))
        );
        assert_eq!(
            "cuda".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Explicit(Device::Cuda(0))
        );
        assert!("tpu".parse::<DeviceRequest>().is_err());
    }

    #[test]
    fn auto_demotes_down_the_chain_and_records_why() {
        let probe = Fake(vec![Device::Cpu]);
        let r = resolve(DeviceRequest::Auto, &all, &probe, None).unwrap();
        assert_eq!(r.device, Device::Cpu);
        assert_eq!(r.dtype, Dtype::F32);
        assert_eq!(r.demotions.len(), 3, "cuda, metal and vulkan each logged");
        assert!(r.demotions[0].starts_with("cuda:0: not present"));
    }

    #[test]
    fn auto_prefers_metal_over_cpu_and_picks_f16() {
        let probe = Fake(vec![Device::Metal, Device::Cpu]);
        let r = resolve(DeviceRequest::Auto, &all, &probe, None).unwrap();
        assert_eq!(r.device, Device::Metal);
        assert_eq!(
            r.dtype,
            Dtype::F16,
            "bf16 on Metal is opt-in, never default"
        );
    }

    #[test]
    fn auto_skips_a_device_the_backend_cannot_use() {
        let probe = Fake(vec![Device::Metal, Device::Cpu]);
        let no_metal = |d: Device| d != Device::Metal;
        let r = resolve(DeviceRequest::Auto, &no_metal, &probe, None).unwrap();
        assert_eq!(r.device, Device::Cpu);
        assert!(r.demotions.iter().any(|d| d.contains("no kernel")));
    }

    #[test]
    fn explicit_device_never_demotes() {
        let probe = Fake(vec![Device::Cpu]);
        let err =
            resolve(DeviceRequest::Explicit(Device::Cuda(0)), &all, &probe, None).unwrap_err();
        assert!(matches!(err, JevError::Device(_)));

        let probe = Fake(vec![Device::Metal, Device::Cpu]);
        let no_metal = |d: Device| d != Device::Metal;
        assert!(
            resolve(
                DeviceRequest::Explicit(Device::Metal),
                &no_metal,
                &probe,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn dtype_override_wins() {
        let probe = Fake(vec![Device::Metal, Device::Cpu]);
        let r = resolve(DeviceRequest::Auto, &all, &probe, Some(Dtype::Bf16)).unwrap();
        assert_eq!(r.dtype, Dtype::Bf16);
    }
}
