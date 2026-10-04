#!/bin/sh
set -eu
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --no-default-features
cargo build --locked --release
cargo build --locked --release --example benchmark
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" \
    cargo build --locked --release --target wasm32-unknown-unknown --no-default-features --features wasm
mkdir -p dist/web dist/demo
cp target/release/parakeet-redux dist/
cp target/release/examples/benchmark dist/parakeet-benchmark
cp web-demo/*.js web-demo/*.css web-demo/*.html dist/demo/
cp tests/fixtures/jfk.pcm dist/demo/
cp LICENSE NOTICE README.md dist/
cp LICENSE NOTICE dist/web/
wasm-bindgen --target web --out-dir dist/web --out-name parakeet_redux \
    target/wasm32-unknown-unknown/release/parakeet_redux.wasm
(cd dist && sha256sum parakeet-redux parakeet-benchmark LICENSE NOTICE README.md web/* demo/* > SHA256SUMS)
mkdir -p dist/site
cp -R dist/web dist/demo dist/site/
printf '%s\n' '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=demo/"><title>Parakeet Speech Lab</title><a href="demo/">Open the speech lab</a>' > dist/site/index.html
