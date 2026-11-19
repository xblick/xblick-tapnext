#!/bin/bash
# Sets xblick-tapnext up from scratch or updates it:
#
# * DeepMind's tapnet (the submodule, at the commit this repository fixes; only its torch TAPNext is used)
# * the venv exactly as uv.lock fixes it (.venv; Python 3.12, the core xblick from PyPI, torch with CUDA 13)
# * TAPNext++'s checkpoints in ~/.cache/xblick/tapnext (the node's cache directory) through xblick-tapnext --download-only (weights.py's CHECKPOINTS: 256 and
#   512, 2.5 GB each, from DeepMind's storage; each checked against its SHA-256, downloaded only if missing). The node
#   downloads the one it needs at its start too.
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH) and git: without them it stops with the command that installs them.
# Without an NVIDIA GPU with driver 580 or newer (CUDA 13) it says so and goes on: the node needs one to start,
# --download-only does not. It never asks and never uses sudo. Run again, it does only what changed.
set -euo pipefail

PROG=xblick-tapnext
URL=https://github.com/xblick/xblick-tapnext.git
MIN_DRIVER=580  # NVIDIA's minimum Linux driver for CUDA 13.0, which the cu130 torch is built for

usage() {
    cat <<'EOF'
Usage: tools/setup.sh

Sets xblick-tapnext up in this clone, or updates it. It fetches DeepMind's tapnet code (the submodule
third_party/tapnet), creates .venv exactly as uv.lock fixes it (Python 3.12, PyTorch for CUDA 13, the core xblick
from PyPI) and downloads the TAPNext++ checkpoints 256 and 512 into the node's cache directory
(~/.cache/xblick/tapnext). A second call does only what changed. It never asks and never uses sudo.

Needs git and uv (UV=/path/to/uv names another uv). The node needs an NVIDIA GPU with driver 580 or newer.

  -h, --help  this text
EOF
}

die() {
    echo "setup.sh: $*" >&2
    exit 1
}

for arg in "$@"; do
    case $arg in
        -h | --help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $arg" ;;
    esac
done

cd "$(dirname "$0")/.."
UV=${UV:-uv}

# ------------------------------------------------------------------------------------------------- prerequisites

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] ||
    die "the node needs Linux on x86-64 (the core's wheels), this is $(uname -s) $(uname -m)"
command -v git > /dev/null || die "no git: sudo apt install git"
command -v "$UV" > /dev/null ||
    die "no uv ($UV): install it with  wget -qO- https://astral.sh/uv/install.sh | sh  and open a new terminal"
git rev-parse --is-inside-work-tree > /dev/null 2>&1 ||
    die "$PWD is not a git clone (a ZIP download has no submodule): git clone --recurse-submodules $URL"

if ! command -v nvidia-smi > /dev/null; then
    echo "$PROG: no NVIDIA driver found (no nvidia-smi). The node needs an NVIDIA GPU with driver $MIN_DRIVER or" \
        "newer to start (Ubuntu: sudo apt install nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
elif ! driver=$(timeout 30 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null) ||
        [[ -z $driver ]]; then
    echo "$PROG: nvidia-smi failed or lists no GPU. The node needs an NVIDIA GPU with driver $MIN_DRIVER or newer" \
        "to start. The setup goes on."
else
    driver=$(cut -d. -f1 <<< "$driver" | sort -n | head -1 | tr -d ' ')
    ((driver >= MIN_DRIVER)) ||
        echo "$PROG: NVIDIA driver $driver. PyTorch for CUDA 13 needs $MIN_DRIVER or newer (Ubuntu: sudo apt install" \
            "nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
fi

# ------------------------------------------------------------------------------------------------- the setup

cat <<EOF
$PROG: setting up $PWD. The first call downloads:
  * DeepMind's tapnet code, the submodule third_party/tapnet (Apache License 2.0)
  * .venv as uv.lock fixes it: Python 3.12, the core xblick, and PyTorch for CUDA 13 with NVIDIA's CUDA libraries
    (NVIDIA's licence, License.txt in each nvidia package); about 3 GB, 6 GB on disk
  * the TAPNext++ checkpoints 256 and 512 (2.5 GB each, Apache License 2.0) into the node's cache directory
EOF
git submodule update --init third_party/tapnet
"$UV" sync --frozen
.venv/bin/xblick-tapnext --download-only --checkpoint 256
.venv/bin/xblick-tapnext --download-only --checkpoint 512
echo "$PROG: set up. The first start: README.md, Install, step 5."
