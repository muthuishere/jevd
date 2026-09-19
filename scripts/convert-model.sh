#!/usr/bin/env bash
# Convert AlexWortega/openjev @ qwen3.5-4b-nli-v2 into the two artefacts openjev serves:
# a GGUF trunk and a standalone classification head.
#
# Organised against design risk R4: upstream retrains, we re-convert, and nobody
# remembers which converter revision or which workaround produced the last GGUF.
#
# Python is allowed HERE and forbidden at runtime. This runs once, on a maintainer
# machine, at publish time. Users get a binary and a download.
#
#   ./scripts/convert-model.sh /path/to/workdir
#
# Leaves in <workdir>/out: the F16 GGUF, Q8_0/Q5_K_M/Q4_K_M, score.safetensors, and
# SHA256SUMS. Needs ~40 GB free and about twenty minutes.

set -euo pipefail

WORK="${1:?usage: convert-model.sh <workdir>}"
REPO="AlexWortega/openjev"
# Pinned, never a branch — the same sha `models.toml` carries. Reproducibility is the
# entire point of this file.
REV="4b5f9a67fa2ebe77466bce0656ce350effc3148c"
SUB="qwen3.5-4b-nli-v2"
NAME="openjev-4b-nli-v2"

mkdir -p "$WORK"/{hf,conv,out}
cd "$WORK"

say() { printf '\n== %s ==\n' "$*"; }

# ---------------------------------------------------------------- 1. weights
say "fetching $REPO@${REV:0:8}/$SUB"
for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do
  [ -s "hf/$f" ] || curl -fSL --retry 3 -C - -o "hf/$f" \
    "https://huggingface.co/$REPO/resolve/$REV/$SUB/$f"
done

# ---------------------------------------------------------------- 2. converter
#
# Two things are wrong with converting this checkpoint with a stock llama.cpp, and only
# one of them is the famous one:
#
#   a) Issue #27019 (ssm_conv1d kernel dim, in_proj_a/b expansion) with draft PR #27132
#      unmerged. Checked on master 2026-09-20: **already fixed**, independently of that
#      PR — `conversion/qwen.py` now carries `_LinearAttentionVReorderBase` which handles
#      both. Do not apply #27132; it is stale and would conflict.
#
#   b) The converter registers `Qwen3_5ForConditionalGeneration` / `ForCausalLM` and NOT
#      `Qwen3_5ForSequenceClassification`, and it hard-fails on the `score.weight`
#      classifier tensor with "Can not map tensor". Both are worked around below: the
#      architecture is rewritten for the converter's benefit (the trunk tensors are
#      identical), and the head is dropped from the GGUF because openjev applies it in
#      Rust anyway.
#
# The pairing that actually matters is converter-vs-runtime on the linear-attention V-head
# order. Master and the tree vendored by llama-cpp-sys-2 0.1.156 both broadcast with the
# same tiled `ggml_repeat_4d`, so master's output loads correctly. That is asserted, not
# assumed: `tests/golden.rs` is what proves it.
say "llama.cpp"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
cd llama.cpp
git rev-parse HEAD > ../out/llama.cpp.commit
if ! grep -q 'openjev: `Qwen3_5ForSequenceClassification`' conversion/qwen.py; then
  python3 - <<'PY'
p = "conversion/qwen.py"
s = open(p).read()
old = """    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        num_k_heads = self.hparams.get("linear_num_key_heads", 0)"""
new = """    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # openjev: `Qwen3_5ForSequenceClassification` checkpoints carry a `score.weight`
        # classifier head that GGUF has no tensor name for. openjev extracts it to a
        # standalone safetensors and applies it in Rust, so drop it here rather than
        # failing the whole conversion on one 60 KB tensor.
        if name.startswith("score."):
            return

        num_k_heads = self.hparams.get("linear_num_key_heads", 0)"""
assert old in s, "converter layout changed — re-check the qwen3_5 path before trusting this"
open(p, "w").write(s.replace(old, new, 1))
print("patched conversion/qwen.py")
PY
fi
cd ..

say "python (publish-time only)"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip -q install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# ---------------------------------------------------------------- 3. convert
say "staging"
for f in model.safetensors tokenizer.json tokenizer_config.json; do
  ln -sf "../hf/$f" "conv/$f"
done
./venv/bin/python - <<PY
import json
d = json.load(open("hf/config.json"))
# The trunk tensors are identical under either architecture name; only the converter's
# registry lookup cares. The head is extracted separately, so nothing is lost.
d["architectures"] = ["Qwen3_5ForConditionalGeneration"]
json.dump(d, open("conv/config.json", "w"), indent=1)
PY

say "convert -> F16"
# --no-mtp: the config declares mtp_num_hidden_layers = 1 and we serve no speculative
# draft. The vision tower drops out on its own — this is a text-only path by design
# (design 01 §1(b)), and the NLI template never emits an image token.
./venv/bin/python llama.cpp/convert_hf_to_gguf.py conv \
  --outfile "out/$NAME-F16.gguf" --outtype f16 --no-mtp

say "quantise"
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
cmake --build llama.cpp/build --config Release -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
  --target llama-quantize >/dev/null
for q in Q8_0 Q5_K_M Q4_K_M; do
  ./llama.cpp/build/bin/llama-quantize "out/$NAME-F16.gguf" "out/$NAME-$q.gguf" "$q" 10
done

# ---------------------------------------------------------------- 4. the head
say "head"
./venv/bin/python - <<PY
from safetensors.torch import load_file, save_file
st = load_file("hf/model.safetensors")
w = st["score.weight"]
assert list(w.shape) == [3, 2560], w.shape
# No bias in this checkpoint. \`models.toml\` says bias = false and the loader refuses a
# mismatch, so assert it here rather than discovering it at boot.
assert "score.bias" not in st, "checkpoint grew a head bias — update models.toml"
save_file({"score.weight": w.contiguous()}, "out/score.safetensors")
print("score.weight", w.dtype, list(w.shape))
PY

say "digests"
cd out && shasum -a 256 ./*.gguf score.safetensors | tee SHA256SUMS
cat <<EOF

Next: upload out/ to the HF repo named in models.toml, then paste the Q8_0 digest into
  [models."openjev-4b-nli-v2".backends.llamacpp.weights].sha256
An empty digest means unpinned, and unpinned means a silent swap is undetectable.
EOF
