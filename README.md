# xblick-tapnext

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)

xblick-tapnext follows points on objects through a camera's colour stream and lifts them to 3D with the depth
stream. It tells you where the surface points of an object go, frame by frame, at 5.6 ms per frame.

It is a node of [Durchblick](https://github.com/xblick/xblick) around Google DeepMind's
[TAPNext++](https://tap-next-plus-plus.github.io/) (CVPR 2026). TAPNext++ is an online point tracker. You give it
points in one frame, and it reports the position and the visibility of each point in every later frame. By default
the node places the points on the objects that an xblick-sam3 instance segments.

## What it publishes

| Topic | Message type | Content |
| --- | --- | --- |
| `<camera>/tapnext/tracks` | point cloud stream | One point per track: `x, y, z`, `u, v`, `track_id`, `object_id`, `visible`. |
| `<camera>/tapnext/queries` | queryable, JSON | The state of the node. A payload adds points by hand, switches the seeding or clears the tracks. |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | The status `tapnext`, once a second. |

The node reads a colour stream with its `camera_info`, a depth stream in the geometry of the colour camera, and
the instance image of xblick-sam3. Every cloud has the stamp and the camera frame of its colour frame.

Details: [docs/tracks.md](docs/tracks.md).

## Requirements

- An NVIDIA GPU with CUDA. The node needs about 1.7 GB of VRAM with the checkpoint 256 and 2.6 GB with 512.
- Durchblick's Python package `xblick` 0.5 or newer, below 0.6. The setup script installs it from source.
- [uv](https://docs.astral.sh/uv/) and git.
- A [xblick-sam3](https://github.com/xblick/xblick-sam3) instance on the same camera, unless
  you give all points by hand.
- DeepMind's tapnet code and the TAPNext++ checkpoints are under Apache 2.0.

## Install

Clone this repository and run the setup script.

```bash
tools/setup.sh
```

The script does three things. It fetches DeepMind's tapnet as a git submodule. It creates `.venv` exactly as
`uv.lock` fixes it, with PyTorch for CUDA 13. It runs `xblick-tapnext --download-only` for the two TAPNext++
checkpoints (256 and 512, 2.5 GB each), which downloads them into `~/.cache/xblick/tapnext/` and checks their SHA-256. A second call
does only what changed. The node then works offline.

The node itself also downloads a missing checkpoint at its start, before it loads the model. It prints the model's
terms, the file and its size first. A preparation script downloads without a start:

```bash
.venv/bin/xblick-tapnext --download-only --checkpoint 256
```

`--download-only` needs no camera, no Zenoh router and no GPU, and it exits with 0 when the weights are complete. A
machine that must never start a download on its own, such as a robot, starts the node with `--no-download`. The
node then ends at once when a file is missing, and it names the file.

### Where the weights are

The node keeps its weights in its cache directory, `~/.cache/xblick/tapnext/`. `XBLICK_HOME=/dir` moves it to `/dir/cache/tapnext/`,
and `XDG_CACHE_HOME` moves it as well. `--weights-dir DIR` names another directory. The place is the same for a
checkout and for a pip install. [Where files are](https://github.com/xblick/xblick/blob/main/docs/reference/files.md) lists the places, and it shows how to copy the cache to a
PC without internet.

Up to version 0.2.2, the weights were in the checkout's `weights/`. The node does not read them there. At its start it
prints the command that moves them. In the checkout it is:

```bash
mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/tapnext
```

## Run

Start xblick-sam3 on the camera first. Then start the node. The camera names are examples. The node connects
to the Zenoh router of the PC, which [Sessions](https://github.com/xblick/xblick/blob/main/docs/concepts/sessions.md) explains.

```bash
.venv/bin/xblick-tapnext --camera cam/azurekinect:depth=depth_to_color --compile default
```

`--compile default` compiles the network during the first frames, which takes about a minute. After that a frame
takes about half the time.

`xblick rerun` shows the tracks without options. It draws the lifted points in 3D and the visible points over
the colour image, with one colour per object.

Add points by hand and read the state of the node:

```bash
.venv/bin/xblick-tapnext-queries --camera cam/azurekinect --json '{"points": [[640, 360], [700, 400]]}'
.venv/bin/xblick-tapnext-queries --camera cam/azurekinect
```

The options that most users change:

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera` | required | `CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME]`. The defaults are `color`, `depth` and `sam3`. |
| `--no-depth` | off | Track in 2D only. `x, y, z` are then NaN. |
| `--no-segmenter` | off | Follow only the points that you give by hand. |
| `--points-per-object` | 32 | The number of points with which the node follows an object. Live: it changes while the node runs. |
| `--max-tracks` | 1024 | The highest number of tracks. |
| `--forget` | 6 | The seconds for which a track may be hidden before it ends. Live: it changes while the node runs. |
| `--checkpoint` | `256` | The TAPNext++ checkpoint: `256` or `512`. |
| `--compile MODE` | off | Compile the network with `torch.compile`, for example `default`. |
| `--no-download` | off | End at once when the checkpoint is missing. The node downloads nothing. |
| `--download-only` | off | Download the missing checkpoint, check it and end. `--camera` may be left out. |
| `--weights-dir DIR` | the cache directory | where the weights are and go, in place of `~/.cache/xblick/tapnext/` |

All options, the cloud layout and the rules for tracks: [docs/tracks.md](docs/tracks.md).

Every option is also a parameter of the node `<camera>/tapnext`, which Durchblick's terminal interface (`xblick`, tab Parameters) and `xblick param` show. `points_per_object`, `forget` and `margin` change while the node runs. The other options apply at the next start, where `--param NAME:=VALUE` or `--params-file FILE` set them. `--download-only` is a mode of the command line and has no parameter.

## Why it is fast

A frame takes 5.6 ms with `--compile default` and 10.7 ms without, for 512 tracks on an RTX PRO 6000.

| Change | Effect |
| --- | --- |
| One model state holds all tracks in slots. | 1024 slots take 12.7 ms, 64 slots take 10.8 ms, both without compiling. |
| `torch.compile` compiles the network. | 10.7 ms become 5.6 ms. |
| The default checkpoint is 256. | 512 takes 13.5 ms compiled and is no more accurate at 640 x 480. |
| New points catch up in a state of their own. | A seeding costs one tracker step per kept frame since the frame of the mask. |

The measurements and the method: [docs/performance.md](docs/performance.md).

## Limits

- Version 0.2.3. We tested the node on TUM RGB-D with ground truth and with xblick-sam3 on a replayed RGB-D
  sequence. We have not tested it on a robot.
- We have not measured the speed on the GPUs of a robot.
- The node publishes the state of each track per frame. A consumer builds the trajectory of a track from its id.
- A change of the image size, the frame id or the camera model resets all tracks.
- The node processes only frames with growing stamps.
- We tested the download against a local server. We have not downloaded a checkpoint from DeepMind's storage
  through the node.

## Documentation

- [docs/tracks.md](docs/tracks.md): inputs, the cloud layout, how tracks begin and end, the queryable, all options.
- [docs/performance.md](docs/performance.md): speed and accuracy, with setup and method.
- [docs/design.md](docs/design.md): how the node adds points to a working tracker, the decisions, the tests.
- Durchblick: [Point clouds](https://github.com/xblick/xblick/blob/main/docs/guides/point-clouds.md), [Pairing streams](https://github.com/xblick/xblick/blob/main/docs/guides/pairing-streams.md),
  [Timestamps](https://github.com/xblick/xblick/blob/main/docs/concepts/timestamps.md), [Transports](https://github.com/xblick/xblick/blob/main/docs/concepts/transports.md).

## Licence and credits

TAPNext and TAPNext++ are Google DeepMind's work. The model code is DeepMind's tapnet, which is the submodule
`third_party/tapnet` under Apache 2.0. The node imports only its PyTorch TAPNext and copies nothing. The TAPNext++
checkpoints are under Apache 2.0 as well, as `tapnet/tapnextpp/README.md` in the submodule states.

The code of this repository is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). [NOTICE](NOTICE) names
the copyright holder and the licences of tapnet and the checkpoints.
