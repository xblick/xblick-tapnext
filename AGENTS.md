# AGENTS.md

xblick-tapnext: DeepMind's TAPNext++ online point tracker as a Durchblick node. By default it seeds 32 points on
each object of an xblick-sam3 instance on the same camera, follows them frame by frame and lifts them to 3D with
the camera's depth. Version 0.2.4, tested on TUM RGB-D and end to end with xblick-sam3 on a replay, never on the
robot. Human documentation: README.md (entry), docs/tracks.md (topics, track rules, queryable, options),
docs/performance.md (speed, accuracy), docs/design.md. Style of all documentation: the core's
docs/project/writing-docs.md. Transport-wide traps (Zenoh sessions and routers, CUDA rings, looped playback, hanging
tests): AGENTS.md of the core repository (`github.com/xblick/xblick`).

## Commands

```bash
tools/setup.sh                              # submodule, uv sync --frozen, --download-only for 256 and 512 (2.5 GB each)
.venv/bin/python -m pytest -q               # test_model.py needs a CUDA device and the checkpoints in ~/.cache/xblick/tapnext
.venv/bin/python -m pytest -q tests/test_weights.py  # CPU, 4 s: the node's files and its start up to the weights (the download: the core's tests)
.venv/bin/xblick-tapnext --download-only --checkpoint 256   # the checkpoint if missing, SHA-256 checked, exit 0
.venv/bin/python -m pytest -q tests/test_tracks.py   # CPU: track rules, seeding, lifting, queries, cloud layout
.venv/bin/ruff check src tests              # lint; 120 columns and rules in pyproject.toml, third_party excluded
tools/check_references.sh                   # no internal references (patterns in the script; also in CI)
.venv/bin/xblick-tapnext --camera cam/azurekinect:depth=depth_to_color --compile default
.venv/bin/xblick-tapnext-queries --camera cam/azurekinect [--json '{...}']   # read or change the state
```

- Console scripts: `xblick-tapnext` (the node), `xblick-tapnext-queries` (client of the queryable).
- `tools/setup.sh` needs uv (`UV`, default `uv` on PATH) and git. It downloads only what is missing (`--download-only`).
- `--no-download`: a start with a missing checkpoint ends with one line (exit 1). Excludes `--download-only`.
- Standalone order: `zenohd` on the PC, xblick-sam3 on the camera, then the node. `xblick rerun` shows the
  tracks in 3D and over the image.
- `--compile default`: about a minute of compiling at the first frames.
- Without a camera: `xblick play <recording.mcap> --loop` of an RGB-D recording, or a simulation.
- Depends on xblick-sam3: default segmenter is the `sam3` instance on the same camera (`CAMERA:segmenter=NAME`
  changes it, `--no-segmenter` for points by hand only). The rc_visard also needs xblick-fast-foundationstereo's
  depth.

## Rules

- Files: the weights are in the node's cache directory (`xblick.paths.cache_dir("tapnext")`, `~/.cache/xblick/tapnext`), never in
  the checkout. `weights.STORE = Store(PROG, node=NODE, ..., former=[FORMER])`, `Weight` paths relative to its root;
  `store = weights.store_for(parser, args)` (`--weights-dir`) for `download_only`, `ensure` and `fail`, with
  `options=weights.options(args)`; model code takes its paths from `store.root`. `FORMER` (the checkout's `weights/`) is never
  read: the store prints the command that moves it. The core's `docs/reference/files.md` has the rule.
- Import tapnet from the submodule (`third_party/tapnet`, fixed commit; `model._tapnext_module` puts it on
  `sys.path`), only its torch TAPNext. Do not install the package's JAX dependencies. Copy nothing of it here.
- The cloud layout is a contract (docs/tracks.md table, `tests/test_tracks.py` checks it): field order and types,
  track order by id, `track_id` never reused while the node works, hidden tracks published with `visible` 0 and NaN
  x y z. Change it only with its readers (the core's Rerun bridge draws u, v over the image).
- Fail fast. Missing submodule, bad queryable requests (`queries.parse` says why), slot misuse
  (`SlotTracker.add` on a used slot) raise.
- The queryable answers in Zenoh's thread at once, from `view`, which the loop keeps after each frame; `source` and
  `clear` go through `inbox` to the loop. A `points` query waits (`Waiting`, `points_waiting`) until the loop put its points in the
  tracker, at most `QUERY_WAIT_S` (2 s): then the watchdog thread answers with an error, and the loop ends the tracks
  it made of them. Never answer from the frame loop alone: a query still open at the caller's timeout gets a
  `Timeout` reply, which a sender takes for a second node.
- The node works offline when its checkpoint is there. Weights: `weights.py` is the one list (`CHECKPOINTS`: file,
  resolution, URL, SHA-256, size; `model.py` imports it). `weights.STORE.ensure` runs at the start before the model loads;
  `--download-only` runs it alone. File there: no network, no file written (`test_weights.py` checks it). Keep it so.
- Downloads: `<name>.part` in ~/.cache/xblick/tapnext, renamed after the SHA-256 check, under the lock `.download.lock` there.
  Never let a partial or unchecked file get its final name. No fallbacks. Never clone tapnet at run time.
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
  shift stamps (the core's `xblick play --loop` does). A change of the colour camera's size, frame id or model
  resets all tracks.
- **Depth** in the colour camera's geometry, any size, 16UC1 mm or 32FC1 m (Azure Kinect: `depth_to_color`;
  rc_visard: `color=left:depth=ffs/depth`). `--no-depth`: 2D only. Lifting undistorts with OpenCV (`lifting.py`).
- **`--compile default`** halves the time per frame (256: 10.7 -> 5.6 ms) but takes about a minute at the first
  frames. Default checkpoint 256. 512 gained nothing at 640 x 480. float16 autocast.
- **`--download-only` and `--camera`.** `--camera` is required for a start only. `xblick.weights.parse_args` lifts
  `required` when the token `--download-only` is spelled out (an abbreviation is not seen).
- **`download_only` as a parameter.** `declare_options` would declare it: it is in `leave_out`. `no_download` is a
  read-only parameter.
- **Wrong SHA-256 after a download.** The `.part` file stays, the error names it. Remove it, start again.
- **Interrupt during a download.** Ctrl-C: exit 130, SIGTERM: exit 143 (`xblick.weights` overrides
  `node.start`'s handler while downloading). The `.part` file is removed.
- **Zenoh (core).** Start nodes without endpoints (they find the PC's router). A session given only `--connect` does
  not listen, and the router does not route between such sessions. No transport compression for the clouds.
- **Background checks.** Start GPU tests and end-to-end checks in the background with `timeout`. Stop them by PID.

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_tapnext/weights.py` | `CHECKPOINTS`, `needed`, `STORE` (the core's `xblick.weights.Store`: the download, `--download-only`/`--no-download`) |
| `src/xblick_tapnext/model.py` | `load`, `SlotTracker`, `Catchup`, coordinate conversion |
| `src/xblick_tapnext/tracks.py` | `Tracks`: ids, ending, drift, adoption, surplus |
| `src/xblick_tapnext/seeding.py` | points spread inside a mask, `--margin` from its edge |
| `src/xblick_tapnext/lifting.py` | `Lifter`: pixels + depth -> metres in the optical frame |
| `src/xblick_tapnext/queries.py` | queryable request parsing and the `xblick-tapnext-queries` client |
| `src/xblick_tapnext/node.py` | the node: inputs, frame loop, seeding, publishing, diagnostics |
| `third_party/tapnet` | DeepMind's tapnet (submodule) |
| `~/.cache/xblick/tapnext/` | `tapnextpp_ckpt.pt` (256), `tapnextpp_512.ckpt` (downloaded by the node), `.download.lock`: the node's cache directory. The checkout's `weights/` is the former place. |
| `tests/` | `test_tracks.py`, `test_weights.py` (CPU), `test_model.py`, `test_node.py` (GPU; `slow_step.py` slows the tracker for it) |
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
| `--max-offset-ms` | 20 (fixed) |
| `--max-hz` / `--local` / `--device` / `--compile` | every frame / `auto` / `cuda` / off |
| Core | `xblick>=0.5.5,<0.6` (pyproject; the core's Python package, `durchblick` up to 0.3 and `blick` in 0.4) |
| Environment | `UV` (setup), `CUDA_MPS_PIPE_DIRECTORY` (MPS expected), `XBLICK_HOME`, `XDG_CACHE_HOME` (cache directory) |
