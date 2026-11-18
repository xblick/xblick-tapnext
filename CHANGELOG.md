# Changelog

## 0.2.5 (2026-11-18)

The Durchblick core 0.5.14.

### Changed

- The node needs `xblick>=0.5.14,<0.6`; uv.lock fixes xblick 0.5.14 (was 0.5.5). No other package changed.
- The node declares itself on the network (core 0.5.9): its name, version, host, process and whole command line on
  `xblick/sessions/<zid>/program`. The terminal interface lists it in its Nodes and Graph tabs.
- The start line says where the node reads its frames, for example `reading the host ring: shm` (core 0.5.8).
- A `--camera` that names a stream (`cam/video2/color`) ends the node with one line that names the camera, instead of
  waiting (core 0.5.7). A router that cannot be reached ends it with one line instead of a traceback.
- A `--connect` to another PC's router in the default mode prints a notice with the three remedies (core 0.5.7).
- A parameter set whose deadline has passed when the node comes to act on it is refused, and the caller raises
  `Expired`. A set meant for another node on the same key is refused (core 0.5.13).
- The pairing default of the core (20 ms since 0.5.11) changes nothing here: `--max-offset-ms` was 20 ms already.
- `tests/test_weights.py`: its check for network use lets an address given as numbers resolve. The core looks up the
  hosts of `--connect` since 0.5.7, to tell whether they are this PC.

### Fixed

- A node that closed its session before its parameters no longer prints a traceback (core 0.5.7).
- Closing no longer raises "Already mutably borrowed" in the thread of the program's declaration (core 0.5.13).

Compatible with earlier clients: nothing on the wire changed.

## 0.2.4 (2026-11-15)

Queries answered at once, a fixed pairing offset and a fixed GPU.

### Changed

- Changed default: `--max-offset-ms` is a fixed 20 ms, no longer half the depth stream's measured frame period. It pairs
  the colour and depth of one capture of the Azure Kinect (up to 17 ms apart) and of the rc_visard (equal stamps). A
  colour frame without a depth frame that close is skipped and counted, as before.
- `--device cuda` is GPU 0, `cuda:N` GPU N. Another value, or a GPU that cannot be used, ends the node with one line.
- The start line says where the model computes (`model on cuda:0`).
- README: how to cite TAPNext++.

### Fixed

- The queryable `queries` answers at once, also while the tracker works on a frame. `source` and `clear` apply from the
  next frame on.
- A `points` query is answered within 2 s: its track ids, or an error, and then no tracks are made of the points.
- `xblick-tapnext-queries` no longer waits past its timeout when Zenoh does not end a query.

Compatible with earlier clients: nothing on the wire changed.

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
