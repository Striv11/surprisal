#!/usr/bin/env bash
# Single quality gate used by CI and by build-lab before every push:
# format check, lint, tests.
set -euo pipefail
cd "$(dirname "$0")"
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all
# Throughput budget (>= 20k symbols/s, one- and two-level)
cargo run --release --example bench
# WASM build size + native/wasm parity (skipped if node or the wasm target is missing)
if command -v node >/dev/null && rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
  cargo build --release --lib --target wasm32-unknown-unknown
  node tests-js/parity.mjs
else
  echo "skipping wasm parity (node or wasm32 target missing)"
fi
# Web app: typecheck, unit tests (need the wasm built above) and a production build.
if [ -d web/node_modules ]; then
  (cd web && npm run typecheck && npm test && npm run build)
else
  echo "skipping web checks (run 'npm ci' in web/ first)"
fi
