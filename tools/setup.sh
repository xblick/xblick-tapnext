#!/bin/bash
# Sets durchblick-tapnext up from scratch or updates it:
#
# * DeepMind's tapnet (the submodule, at the commit this repository fixes; only its torch TAPNext is used)
# * the venv exactly as uv.lock fixes it (.venv; the core blick from github.com/xblick/durchblick, torch with CUDA 13)
# * TAPNext++'s checkpoints in weights/ (model.py's CHECKPOINTS: 256 and 512, 2.5 GB each, from DeepMind's storage;
#   each checked against its SHA-256, downloaded only if missing; the node runs offline)
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH), git and curl. Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV (set UV to the uv binary)" >&2; exit 1; }
git submodule update --init third_party/tapnet
"$UV" sync --frozen
.venv/bin/python -c "
from durchblick_tapnext.model import CHECKPOINTS
for name, url, sha in ((n, u, s) for n, _, u, s in CHECKPOINTS.values()):
    print(name, url, sha)" | while read -r name url sha; do
    target="weights/$name"
    if [[ ! -f "$target" ]]; then
        mkdir -p weights
        curl -fsSL "$url" -o "$target.part"
        mv "$target.part" "$target"
    fi
    echo "$sha  $target" | sha256sum --check --quiet || {
        echo "$target: wrong SHA-256; delete it and run again" >&2; exit 1; }
done
echo "durchblick-tapnext: set up; checkpoints in weights/"
