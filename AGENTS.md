# AGENTS.md

durchblick-tapnext: DeepMind's TAPNext++ online point tracker as a Durchblick node. By default it seeds 32 points on
each object of a durchblick-sam3 instance on the same camera, follows them frame by frame and lifts them to 3D with
the camera's depth. Version 0.1.0, tested on TUM RGB-D and end to end with durchblick-sam3 on a replay, never on the
robot. Human documentation: README.md (entry), docs/tracks.md (topics, track rules, queryable, options),
docs/performance.md (speed, accuracy), docs/design.md. Style of all documentation: the core's
docs/project/writing-docs.md. Transport-wide traps (Zenoh sessions and routers, CUDA rings, looped playback, hanging
tests): AGENTS.md of the core repository (`github.com/xblick/durchblick`).

## Commands

```bash
tools/setup.sh                              # submodule, uv sync --frozen, checkpoints 256 + 512 (2.5 GB each)
.venv/bin/python -m pytest -q               # 12 tests; test_model.py needs a CUDA device and weights/
.venv/bin/python -m pytest -q tests/test_tracks.py   # CPU: track rules, seeding, lifting, queries, cloud layout
.venv/bin/ruff check src tests              # lint; 120 columns and rules in pyproject.toml, third_party excluded
tools/check_references.sh                   # no internal references (patterns in the script; also in CI)
.venv/bin/durchblick-tapnext --camera robot/camera:depth=depth_to_color --compile default
.venv/bin/durchblick-tapnext-queries --camera robot/camera [--json '{...}']   # read or change the state
```

- Console scripts: `durchblick-tapnext` (the node), `durchblick-tapnext-queries` (client of the queryable).
- `tools/setup.sh` needs uv (`UV`, default `uv` on PATH), git and curl. It downloads only what is missing.
- Standalone order: `zenohd` on the PC, durchblick-sam3 on the camera, then the node. `blick rerun` shows the
  tracks in 3D and over the image.
- `--compile default`: about a minute of compiling at the first frames.
- Without a camera: `blick play <recording.mcap> --loop` of an RGB-D recording, or a simulation.
- Depends on durchblick-sam3: default segmenter is the `sam3` instance on the same camera (`CAMERA:segmenter=NAME`
  changes it, `--no-segmenter` for points by hand only). The rc_visard also needs durchblick-fast-foundationstereo's
  depth.

## Rules

- Import tapnet from the submodule (`third_party/tapnet`, fixed commit; `model._tapnext_module` puts it on
  `sys.path`), only its torch TAPNext. Do not install the package's JAX dependencies. Copy nothing of it here.
- The cloud layout is a contract (docs/tracks.md table, `tests/test_tracks.py` checks it): field order and types,
  track order by id, `track_id` never reused while the node works, hidden tracks published with `visible` 0 and NaN
  x y z. Change it only with its readers (the core's Rerun bridge draws u, v over the image).
- Fail fast. Missing weights or submodule, bad queryable requests (`queries.parse` says why), slot misuse
  (`SlotTracker.add` on a used slot) raise.
- The node works offline.
- README.md and docs/ are the documentation of record. 120 columns.

## Traps

- **Slot trick** (`model.SlotTracker`). DeepMind's code cannot add queries to an existing state. The node keeps one
  state of `--max-tracks` slots plus one never-used sentinel slot. A free slot is fed as a query of the future ("not
  yet queried"). `add` makes it a query of the next frame. `free` copies the sentinel's RG-LRU and conv1d states into
  it. Checked on TUM: within 0.6-1.3 px (median) of a state per point group. Never give free slots real coordinates.
- **Coordinates.** Outside: pixels with centres at integers (ROS/`camera_info` K). Model: 256 x 256 with centres at
  .5 and [y, x] order (`to_model`, `to_pixels`), whatever the checkpoint's input resolution.
- **Late masks** (`model.Catchup`). SAM 3's result arrives frames after its frame, and a point must be queried in the
  frame it is on. New points are followed from their frame to the newest in a state of their own (frames kept by
  `--history`, 60), batched per frame, then enter the tracker as points of the newest frame. Points hidden there
  are dropped. A result whose frame was never tracked (skipped, or older than the kept frames) cannot seed (counted
  `seeds_too_old`).
- **SAM 3 re-ids its objects constantly** (about 200 ids for one desk in 30 s of replay). Tracks of objects absent
  from a result that lie inside a new id's mask are adopted by it (`Tracks.orphans`, `Tracks.adopt`). Without a free
  slot, the tracks hidden longest end (`Tracks.hidden_longest`). More than twice `--points-per-object` per object:
  its longest hidden ones end (`Tracks.surplus`).
- **Stamps only grow.** Symptom: no tracking on a looped replay. Cause: only colour frames (with their depth) newer
  than the last tracked one are processed, newest first, older ones skipped and counted. Fix: looped playback must
  shift stamps (the core's `blick play --loop` does). A change of the colour camera's size, frame id or model
  resets all tracks.
- **Depth** in the colour camera's geometry, any size, 16UC1 mm or 32FC1 m (Azure Kinect: `depth_to_color`;
  rc_visard: `color=left:depth=ffs/depth`). `--no-depth`: 2D only. Lifting undistorts with OpenCV (`lifting.py`).
- **`--compile default`** halves the time per frame (256: 10.7 -> 5.6 ms) but takes about a minute at the first
  frames. Default checkpoint 256. 512 gained nothing at 640 x 480. float16 autocast.
- **Zenoh (core).** Start nodes without endpoints (they find the PC's router). A session given only `--connect` does
  not listen, and the router does not route between such sessions. No transport compression for the clouds.
- **Background checks.** Start GPU tests and end-to-end checks in the background with `timeout`. Stop them by PID.

## Layout

| Path | Content |
| --- | --- |
| `src/durchblick_tapnext/model.py` | `CHECKPOINTS`, `load`, `SlotTracker`, `Catchup`, coordinate conversion |
| `src/durchblick_tapnext/tracks.py` | `Tracks`: ids, ending, drift, adoption, surplus |
| `src/durchblick_tapnext/seeding.py` | points spread inside a mask, `--margin` from its edge |
| `src/durchblick_tapnext/lifting.py` | `Lifter`: pixels + depth -> metres in the optical frame |
| `src/durchblick_tapnext/queries.py` | queryable request parsing and the `durchblick-tapnext-queries` client |
| `src/durchblick_tapnext/node.py` | the node: inputs, frame loop, seeding, publishing, diagnostics |
| `third_party/tapnet` | DeepMind's tapnet (submodule) |
| `weights/` | `tapnextpp_ckpt.pt` (256), `tapnextpp_512.ckpt` (made by `tools/setup.sh`, not in git) |
| `tests/` | `test_tracks.py` (CPU), `test_model.py` (GPU) |
| `docs/` | tracks, performance, design |

## Defaults

| What | Value |
| --- | --- |
| Camera spec | `CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME]`, defaults `color`, `depth`, `sam3` |
| Output | `<camera>/tapnext/tracks`: one `sensor_msgs/PointCloud2` per frame (x y z u v track_id object_id visible, 32 B a point) |
| Queryable | `<camera>/tapnext/queries` (JSON: points by hand, seeding source `sam3`/`none`, clear) |
| Status | `tapnext` on `<camera>/diagnostics` |
| `--points-per-object` / `--forget` / `--margin` | 32 / 6 s / 4 px |
| `--max-tracks` / `--history` / `--checkpoint` | 1024 / 60 / `256` |
| `--max-offset-ms` | half the depth's frame period |
| `--max-hz` / `--local` / `--device` / `--compile` | every frame / `auto` / `cuda` / off |
| Core | `blick>=0.4,<0.5` (pyproject; the core's Python package, renamed from `durchblick` at 0.4.0) |
| Environment | `UV` (setup), `CUDA_MPS_PIPE_DIRECTORY` (MPS expected) |
