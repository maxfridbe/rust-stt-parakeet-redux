#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
image=${PARAKEET_BUILD_IMAGE:-localhost/parakeet-redux-build}
podman build --tag "$image" --file Containerfile .
container=$(podman create "$image")
trap 'podman rm "$container" >/dev/null' EXIT HUP INT TERM
mkdir -p dist
podman cp "$container:/artifacts/." dist/
printf 'Native and browser artifacts written to dist/\n'
