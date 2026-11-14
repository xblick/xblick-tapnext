#!/bin/bash
# Sets xblick-tapnext up from scratch or updates it:
#
# * DeepMind's tapnet (the submodule, at the commit this repository fixes; only its torch TAPNext is used)
# * the venv exactly as uv.lock fixes it (.venv; the core xblick from github.com/xblick/xblick, torch with CUDA 13)
# * TAPNext++'s checkpoints in ~/.cache/xblick/tapnext (the node's cache directory) through xblick-tapnext --download-only (weights.py's CHECKPOINTS: 256 and
#   512, 2.5 GB each, from DeepMind's storage; each checked against its SHA-256, downloaded only if missing). The node
#   downloads the one it needs at its start too.
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH) and git. Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV (set UV to the uv binary)" >&2; exit 1; }
git submodule update --init third_party/tapnet
"$UV" sync --frozen
.venv/bin/xblick-tapnext --download-only --checkpoint 256
.venv/bin/xblick-tapnext --download-only --checkpoint 512
echo "xblick-tapnext: set up"
