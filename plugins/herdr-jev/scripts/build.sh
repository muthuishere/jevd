#!/usr/bin/env bash
# Builds the single Go binary the manifest runs. No node, no runtime deps, no model:
# inference belongs to `openjev serve`, which this plugin only ever speaks HTTP to.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p bin
go build -trimpath -o bin/herdr-jev ./cmd/herdr-jev
echo "built bin/herdr-jev"
