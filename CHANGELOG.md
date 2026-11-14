# Changelog

## 0.2.3 (2026-11-14)

- The weights moved out of the checkout into the node's cache directory, `~/.cache/xblick/tapnext/`. `XBLICK_HOME` and
  `XDG_CACHE_HOME` move it, and `--weights-dir DIR` names another directory. Weights at the former place, the
  checkout's `weights/`, are not used. The node prints the command that moves them. In the checkout it is
  `mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/tapnext`.
- `--weights-dir DIR` is a new start option and a read-only parameter.
- The message about a missing file names the full download command (`xblick-tapnext --download-only --checkpoint 256`).
- The node needs `xblick>=0.5.5,<0.6`.
- `model.load` takes the weights' directory (`weights.STORE.root`, or `--weights-dir`'s).

## 0.2.2 (2026-11-14)

- The download of the weights is the core's `xblick.weights` (xblick 0.5.3). The node's own copy of that code is
  gone. `weights.py` keeps the node's files, its terms notice and its `STORE`. The options, the messages, and the
  files and their places stay as they were.
- The node needs `xblick>=0.5.3,<0.6`.
- `tests/test_weights.py` keeps the node's own checks. The core's tests cover the download itself.

## 0.2.1 (2026-11-14)

- The node downloads a missing checkpoint at its start, before the model loads. It prints the model's terms, the
  file and its size first. With the checkpoint there it uses no network and writes no file.
- `--download-only` downloads and checks the checkpoint of `--checkpoint`, then exits 0. It needs no camera, Zenoh
  router or GPU, and `--camera` may be left out. `tools/setup.sh` calls it for 256 and 512 after `uv sync` and no
  longer needs curl.
- `--no-download` ends a start at once when the checkpoint is missing and names it. It is a read-only parameter.
- `weights.py` is the one list of the checkpoints. `model.CHECKPOINTS` moved there and gained the file sizes.
- A download goes to `<name>.part` under the lock `weights/.download.lock` and gets its final name after the SHA-256
  check.
- A download is also checked against the exact size in `weights.py`, before the SHA-256.

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-tapnext | github.com/xblick/xblick-tapnext |
| Distribution | `durchblick-tapnext` | `xblick-tapnext` |
| Python package | `durchblick_tapnext` | `xblick_tapnext` |
| Commands | `durchblick-tapnext`, `durchblick-tapnext-queries` | `xblick-tapnext`, `xblick-tapnext-queries` |
| Program name in log lines, the logger and reports | `durchblick-tapnext` | `xblick-tapnext` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |
| Other nodes named in the docs | `durchblick-sam3` (link github.com/xblick/durchblick-sam3), `durchblick-nemo`, `durchblick-foundationpose`, `durchblick-fast-foundationstereo` | `xblick-sam3` (github.com/xblick/xblick-sam3), `xblick-nemo`, `xblick-foundationpose`, `xblick-fast-foundationstereo` |
| CI | secret `DURCHBLICK_READ_TOKEN` to read the private core | none |

## 0.1.0

The node as `durchblick-tapnext` (Python package `durchblick_tapnext`) on the core `blick` 0.4.
