//! Throughput and latency on real weights. Gated exactly like the other real-weights
//! tests, and **asserts nothing** — it reports. A benchmark that fails the build on a
//! noisy laptop is a benchmark people delete.
//!
//! Methodology, stated so the numbers mean something:
//!   * **Warm, never cold.** The model is loaded, then `WARMUP` forwards are run and
//!     discarded before any timing. A cold first forward on Metal pays shader compilation
//!     and buffer allocation and is not what a served request costs.
//!   * **Synthetic token ids** of an exact length, so "128 tokens" is 128 tokens rather
//!     than whatever a sentence happened to tokenise to. NLI pairs in practice are short;
//!     32 / 128 / 512 brackets the realistic range.
//!   * **Per-forward wall clock**, sorted, p50 and p95 reported from the sorted samples.
//!     Throughput is pairs divided by total measured time, so it includes the per-call
//!     overhead a caller actually pays.
//!   * One process, one backend, one thread driving it — which is what `openjev serve`
//!     does today (ADR 0005).
//!
//! ```text
//! OPENJEV_TEST_GGUF=.../openjev-4b-nli-v2-Q8_0.gguf \
//! OPENJEV_BENCH_CTX=1024 OPENJEV_TEST_DEVICE=metal \
//!   cargo test --release -p openjev-core --features real-weights,backend-llamacpp \
//!   --test bench -- --nocapture
//! ```

#![cfg(all(feature = "real-weights", feature = "backend-llamacpp"))]

use openjev_core::backend::{Backend, EncodedInput, OpenRequest};
use openjev_core::backends::llamacpp;
use openjev_core::device::{Device, Dtype};
use openjev_core::registry::Registry;
use std::path::PathBuf;
use std::time::Instant;

const WARMUP: usize = 5;

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn ids(seed: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| 1000 + (seed * 97 + i as u32 * 31) % 40000)
        .collect()
}

fn open() -> Option<(Box<dyn Backend>, String)> {
    let path = std::env::var_os("OPENJEV_TEST_GGUF")
        .map(PathBuf::from)
        .filter(|p| p.is_file())?;

    let ctx = env_usize("OPENJEV_BENCH_CTX", 8192);
    let hidden = env_usize("OPENJEV_TEST_HIDDEN", 2560);
    let threads = std::env::var("OPENJEV_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    let mut reg = Registry::default();
    reg.merge_str(
        &format!(
            r#"
[models.bench]
repo = "local/bench"
arch = "qwen3_5"
template = "Premise: {{premise}} Hypothesis: {{hypothesis}}"
labels = ["contradiction", "entailment", "neutral"]
entailment_label = "entailment"
context = {ctx}
hidden_size = {hidden}
tokenizer = {{ repo = "local/bench", file = "tokenizer.json", pad_token_id = 248044 }}
head = {{ kind = "linear", in_features = {hidden}, out_features = 3, tensor = "score.weight", repo = "r", file = "f" }}
[models.bench.backends.llamacpp]
requires = ["qwen3_5-hybrid"]
weights = {{ repo = "r", file = "w.gguf" }}
"#
        ),
        std::path::Path::new("<bench>"),
    )
    .expect("bench registry entry must be valid");
    let spec = reg.get("bench").expect("spec").clone();

    let device = match std::env::var("OPENJEV_TEST_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ if llamacpp::factory().supports_device(Device::Metal) => Device::Metal,
        _ => Device::Cpu,
    };
    let dtype_label: &'static str = Box::leak(
        std::env::var("OPENJEV_TEST_DTYPE")
            .unwrap_or_else(|_| "q8_0".into())
            .into_boxed_str(),
    );

    let load = Instant::now();
    let backend = llamacpp::factory()
        .open(&OpenRequest {
            spec,
            weights: path.clone(),
            device,
            dtype: Dtype::Quant(dtype_label),
            context: ctx,
            n_threads: threads,
        })
        .expect("backend must open");
    let label = format!(
        "{}  device={device}  dtype={dtype_label}  n_ctx={ctx}  threads={}  load={:.1}s",
        path.file_name().unwrap_or_default().to_string_lossy(),
        threads
            .map(|t| t.to_string())
            .unwrap_or_else(|| "auto".into()),
        load.elapsed().as_secs_f64()
    );
    Some((backend, label))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

#[test]
fn report_latency_and_throughput() {
    let Some((backend, label)) = open() else {
        return;
    };
    eprintln!("\n=== {label} ===");
    eprintln!(
        "{:>7} {:>6} {:>10} {:>10} {:>12}",
        "tokens", "batch", "p50 ms", "p95 ms", "pairs/s"
    );

    // Batch sizes are swept even though the backend declares !Caps::BATCH and loops
    // internally. That is the point: it measures what coalescing would have to beat, and
    // a flat line across batch size is the evidence that batching is unimplemented rather
    // than unhelpful.
    let batches: Vec<usize> = std::env::var("OPENJEV_BENCH_BATCHES")
        .unwrap_or_else(|_| "1,4,16".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    for tokens in [32usize, 128, 512] {
        for &b in &batches {
            // Fewer iterations as the work grows; enough samples for a p95 to mean
            // something without the whole sweep taking an afternoon.
            let iters = match tokens {
                0..=64 => 30,
                65..=256 => 20,
                _ => 10,
            }
            .max(6);

            let inputs: Vec<EncodedInput> = (0..b)
                .map(|k| EncodedInput::unpadded(ids(k as u32 + 7, tokens)).expect("input"))
                .collect();

            for _ in 0..WARMUP {
                backend.forward(&inputs).expect("warmup forward");
            }

            let mut samples = Vec::with_capacity(iters);
            let total = Instant::now();
            for _ in 0..iters {
                let t = Instant::now();
                backend.forward(&inputs).expect("forward");
                samples.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let elapsed = total.elapsed().as_secs_f64();
            samples.sort_by(f64::total_cmp);

            eprintln!(
                "{:>7} {:>6} {:>10.1} {:>10.1} {:>12.1}",
                tokens,
                b,
                percentile(&samples, 0.50),
                percentile(&samples, 0.95),
                (iters * b) as f64 / elapsed
            );
        }
    }
    eprintln!();
}
