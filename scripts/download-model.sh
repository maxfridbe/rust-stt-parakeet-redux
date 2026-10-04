#!/bin/sh
set -eu
destination=${1:-models/parakeet-redux}
revision=2bf128600aac4b16946f7ed8372e56117fe5e23b
mkdir -p "$destination"
for file in config.json ternary.json tokenizer.json model.safetensors; do
    curl --fail --location --retry 3 \
        "https://huggingface.co/moondream/parakeet-redux/resolve/$revision/$file" \
        --output "$destination/$file.part"
    mv "$destination/$file.part" "$destination/$file"
done
cd "$destination"
sha256sum --check <<'CHECKSUMS'
503c653b2e3bb788adbcb04f5abdee532d958686564081baeed133ff10143f6e  config.json
78ec25733ee0d0c1586d1346fc86db9d0c2e436e3a8ab1d32a82d1bb8f848d21  model.safetensors
1221c6d3ce901ffe09c089da758a8db8b76189f80cff41c5afc244fc61e2051d  ternary.json
bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91  tokenizer.json
CHECKSUMS
