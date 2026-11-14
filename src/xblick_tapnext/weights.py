"""The node's weights: which files it reads, where they come from, and their download.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-tapnext
--download-only``) both use it. ``STORE.ensure`` touches neither the disk nor the network when every file is there. A
missing file is downloaded under a temporary name in its directory (``<name>.part``), checked and only then renamed,
all under a file lock, so that two nodes started at once download once and never read half a file.

* ``CHECKPOINTS``: name -> (file in ``weights/``, the input resolution it was fine-tuned at, URL, SHA-256, size in
  bytes); DeepMind's TAPNext++ README and VOTSp2026 tracker
* ``needed(checkpoint)``: the file a start with ``--checkpoint`` reads (none for a .pt file of one's own)

DeepMind's tapnet (the submodule third_party/tapnet) is code, not weights: ``tools/setup.sh`` fetches it.

The download itself (lock, ``.part`` files, checks, progress, ``--download-only``/``--no-download``) is the core's
``xblick.weights``; this module keeps the node's files, its terms notice and its ``STORE``.
"""

from __future__ import annotations

from pathlib import Path

from xblick.weights import Store, Url, Weight

PROG = "xblick-tapnext"
NODE = "tapnext"  # the short name: its cache directory is ~/.cache/xblick/tapnext
REQUIRED = ()  # (path, what): code from tools/setup.sh that must be there first; none here
REPO = Path(__file__).resolve().parents[2]
FORMER = REPO / "weights"  # the weights' place before 0.2.3 (the checkout): not used, the store names the move
CHECKPOINTS = {  # name -> (file in weights/, input resolution, URL, SHA-256, size); Apache 2.0
    "256": ("tapnextpp_ckpt.pt", 256, "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt",
            "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da", 2_532_282_370),
    "512": ("tapnextpp_512.ckpt", 512, "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt",
            "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f", 2_532_283_010),
}
TERMS = """\
TAPNext++ is the work of Google DeepMind, under its terms, and you are the one who uses it:
  * The TAPNext++ checkpoints (DeepMind's storage, storage.googleapis.com) are under the Apache License 2.0, as
    tapnet/tapnextpp/README.md in DeepMind's tapnet states."""


def needed(checkpoint: str) -> list[Weight]:
    """The file of ``--checkpoint``: that of a name in CHECKPOINTS, none for a file of one's own."""
    if checkpoint not in CHECKPOINTS:
        return []
    name, _, url, sha256, size = CHECKPOINTS[checkpoint]
    return [Weight(Path(name), Url(url), size, sha256)]  # below the store's root


def options(args) -> list[str]:
    """The options that select the weights (and --weights-dir), for the download command the lines name."""
    return ["--checkpoint", args.checkpoint] + (["--weights-dir", args.weights_dir] if args.weights_dir else [])


def store_for(parser, args) -> Store:
    """STORE at --weights-dir (not given: the node's cache directory); a wrong --weights-dir is an argument error."""
    try:
        return STORE.at(args.weights_dir)
    except ValueError as error:
        parser.error(f"--weights-dir: {error}")


# the download (xblick.weights): the root is the node's cache directory, its lock there
STORE = Store(PROG, node=NODE, terms=TERMS, required=REQUIRED, repo=REPO, former=[FORMER])
