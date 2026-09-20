#!/usr/bin/env bash
#
# Build herdr-jev: one Go binary, no node, no runtime deps, no model. Inference
# belongs to `openjev serve`, which this plugin only ever speaks HTTP to.
#
#   ./scripts/build.sh                 build for this machine  -> bin/herdr-jev
#   ./scripts/build.sh --release       build the full matrix   -> dist/
#
# Environment:
#   VERSION     version string baked into the binary (default: git describe)
#   GO          go binary to use    (default: go)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GO="${GO:-go}"
PKG="./cmd/herdr-jev"
BIN_NAME="herdr-jev"

VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

LDFLAGS="-s -w -X main.version=${VERSION} -X main.commit=${COMMIT} -X main.buildDate=${BUILD_DATE}"

RELEASE=0
for arg in "$@"; do
  case "$arg" in
    --release)  RELEASE=1 ;;
    -h|--help)  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found on PATH. $2"; }

need "$GO" "Install Go 1.25+ from https://go.dev/dl/"

# CGO_ENABLED=0 gives a static binary and clean cross-compilation.
# -trimpath keeps absolute build paths out of the binary.
go_build() {
  local goos="$1" goarch="$2" out="$3"
  mkdir -p "$(dirname "$out")"
  CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" \
    "$GO" build -trimpath -ldflags "$LDFLAGS" -o "$out" "$PKG"
}

if [ "$RELEASE" -eq 1 ]; then
  log "release matrix, version ${VERSION}"
  rm -rf dist
  for target in darwin/amd64 darwin/arm64 linux/amd64 linux/arm64; do
    goos="${target%/*}"
    goarch="${target#*/}"
    out="dist/${BIN_NAME}_${VERSION}_${goos}_${goarch}/${BIN_NAME}"
    log "  ${goos}/${goarch}"
    go_build "$goos" "$goarch" "$out"
    ( cd "$(dirname "$out")" && tar -czf "../${BIN_NAME}_${VERSION}_${goos}_${goarch}.tar.gz" "$BIN_NAME" )
  done
  ( cd dist && shasum -a 256 ./*.tar.gz > "${BIN_NAME}_${VERSION}_SHA256SUMS" )
  log "release artifacts in dist/"
  ls -1 dist/*.tar.gz
else
  log "building bin/${BIN_NAME}, version ${VERSION}"
  go_build "$("$GO" env GOOS)" "$("$GO" env GOARCH)" "bin/${BIN_NAME}"
  log "built bin/${BIN_NAME}"
fi
