FROM docker.io/library/rust:1.97.1-bookworm AS builder
RUN rustup component add rustfmt clippy \
    && rustup target add wasm32-unknown-unknown \
    && cargo install wasm-bindgen-cli --version 0.2.104 --locked
WORKDIR /work
COPY Cargo.toml Cargo.lock ./
COPY LICENSE NOTICE README.md ./
COPY src ./src
COPY tests ./tests
COPY examples ./examples
COPY web-demo ./web-demo
COPY scripts/build-artifacts.sh ./scripts/build-artifacts.sh
RUN sh scripts/build-artifacts.sh

FROM docker.io/library/debian:bookworm-slim AS artifacts
COPY --from=builder /work/dist /artifacts
ENTRYPOINT ["/artifacts/parakeet-redux"]
