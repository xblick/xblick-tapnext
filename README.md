# xblick-tapnext

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)
[![tests](https://github.com/xblick/xblick-tapnext/actions/workflows/ci.yml/badge.svg)](https://github.com/xblick/xblick-tapnext/actions/workflows/ci.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![device](https://img.shields.io/badge/needs-NVIDIA%20GPU-76b900)](README.md)

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

- Linux on x86-64 and an NVIDIA GPU with driver 580 or newer (PyTorch for CUDA 13). The node needs about 1.7 GB
  of VRAM with the checkpoint 256 and 2.6 GB with 512.
- Durchblick's Python package `xblick` 0.5 or newer, below 0.6. The setup script installs it from PyPI.
- git and [uv](https://docs.astral.sh/uv/). Step 1 of the installation installs them.
- About 6 GB of disk for `.venv` and 5 GB for the checkpoints.
- A [xblick-sam3](https://github.com/xblick/xblick-sam3) instance on the same camera, unless
  you give all points by hand.
- DeepMind's tapnet code and the TAPNext++ checkpoints are under Apache 2.0.

## Install

1. Install the prerequisites. `nvidia-smi` prints the GPU and the driver version, which must be 580 or newer. On
   Ubuntu, `sudo apt install nvidia-driver-580` and a restart install that driver. After the installation of uv, open
   a new terminal, so that `uv` is on `PATH`.

   ```bash
   nvidia-smi
   sudo apt install git
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository with its submodule, DeepMind's tapnet:

   ```bash
   git clone --recurse-submodules https://github.com/xblick/xblick-tapnext.git
   cd xblick-tapnext
   ```

3. Run the setup script. It fetches tapnet, creates `.venv` exactly as `uv.lock` fixes it, with Python 3.12 and
   PyTorch for CUDA 13 (about 3 GB to download, 6 GB on disk), and downloads the two TAPNext++ checkpoints 256 and
   512 (2.5 GB each) into `~/.cache/xblick/tapnext/`. It prints the licence of each part before its download and the
   progress of each download. When git or uv is missing, it stops and prints the command that installs it. A second
   call does only what changed, and after it the node needs no network. `tools/setup.sh --help` describes it.

   ```bash
   tools/setup.sh
   ```

4. Install the core's command `xblick` with the viewer and video files. uv puts it on `PATH`.

   ```bash
   uv tool install "xblick[rerun,video]>=0.5,<0.6"
   ```

5. Start the node on a video, without a camera. Any short video works, for example one from a phone. Run each
   command in its own terminal, in the clone. If the PC already runs a Zenoh router (`xblick router` or `zenohd`),
   leave out the first one.

   ```bash
   xblick router                                               # the Zenoh router of this PC
   xblick play ~/Videos/clip.mp4 --camera demo/video --loop    # the video as the camera demo/video
   .venv/bin/xblick-tapnext --camera demo/video --no-depth --no-segmenter
   xblick rerun                                                # the viewer
   ```

   When the node prints its frames per second, give it points to follow, in pixels of the video:

   ```bash
   .venv/bin/xblick-tapnext-queries --camera demo/video --json '{"points": [[320, 240], [400, 300]]}'
   ```

   The viewer shows the video with a coloured dot on each point. The dots move with the surface they lie on. A
   point that the tracker loses, for example on a plain wall, ends after 6 seconds. Send new points at any time.
   `--no-depth` tracks in 2D, and `--no-segmenter` follows only the points that you give. [Run](#run) uses a camera
   with depth and xblick-sam3.

### Downloads at the start

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

- The node publishes the state of each track per frame. A consumer builds the trajectory of a track from its id.
- A change of the image size, the frame id or the camera model resets all tracks.
- The node processes only frames with growing stamps.

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

## Citation

This node uses TAPNext++ from the repository google-deepmind/tapnet. If you use it in research, cite their work. The
repository asks for all of these entries. The entries below are copied from [its
README](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/README.md):

```bibtex
@article{doersch2022tap,
  title={{TAP}-Vid: A Benchmark for Tracking Any Point in a Video},
  author={Doersch, Carl and Gupta, Ankush and Markeeva, Larisa and Recasens, Adria and Smaira, Lucas and Aytar, Yusuf and Carreira, Joao and Zisserman, Andrew and Yang, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={13610--13626},
  year={2022}
}

@inproceedings{doersch2023tapir,
  title={{TAPIR}: Tracking any point with per-frame initialization and temporal refinement},
  author={Doersch, Carl and Yang, Yi and Vecerik, Mel and Gokay, Dilara and Gupta, Ankush and Aytar, Yusuf and Carreira, Joao and Zisserman, Andrew},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10061--10072},
  year={2023}
}

@article{vecerik2023robotap,
  title={{RoboTAP}: Tracking arbitrary points for few-shot visual imitation},
  author={Vecerik, Mel and Doersch, Carl and Yang, Yi and Davchev, Todor and Aytar, Yusuf and Zhou, Guangyao and Hadsell, Raia and Agapito, Lourdes and Scholz, Jon},
  journal={International Conference on Robotics and Automation},
  pages={5397--5403},
  year={2024}
}

@article{doersch2024bootstap,
  title={{BootsTAP}: Bootstrapped Training for Tracking-Any-Point},
  author={Doersch, Carl and Luc, Pauline and Yang, Yi and Gokay, Dilara and Koppula, Skanda and Gupta, Ankush and Heyward, Joseph and Rocco, Ignacio and Goroshin, Ross and Carreira, Jo{\~a}o and Zisserman, Andrew},
  journal={Asian Conference on Computer Vision},
  year={2024}
}

@article{koppula2024tapvid,
  title={{TAPVid}-{3D}: A Benchmark for Tracking Any Point in {3D}},
  author={Koppula, Skanda and Rocco, Ignacio and Yang, Yi and Heyward, Joe and Carreira, Jo{\~a}o and Zisserman, Andrew and Brostow, Gabriel and Doersch, Carl},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{zholus2025tapnext,
  title={TAPNext: Tracking Any Point (TAP) as Next Token Prediction},
  author={Zholus, Artem and Doersch, Carl and Yang, Yi and Koppula, Skanda and Patraucean, Viorica and He, Xu Owen and Rocco, Ignacio and Sajjadi, Mehdi S. M. and Chandar, Sarath and Goroshin, Ross},
  journal={arXiv preprint arXiv:2504.05579},
  year={2025}
}

@article{allen2025trajan,
  title={Direct Motion Models for Assessing Generated Videos},
  author={Allen, Kelsey and Doersch, Carl and Zhou, Guangyao and Suhail, Mohammed and Driess, Danny and Rocco, Ignacio and Rubanova, Yulia and Kipf, Thomas and Sajjadi, Mehdi S. M. and Murphy, Kevin and Carreira, Joao and van Steenkiste, Sjoerd},
  journal={arXiv preprint},
  year={2025}
}
```
