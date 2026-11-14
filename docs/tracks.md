# Tracks, topics and options

This page describes what xblick-tapnext reads and publishes, how a track begins and ends, how you control the
node through its queryable, and every option.

## Inputs

You name the camera as Durchblick's other RGB-D nodes do, for example xblick-foundationpose and
xblick-nemo:

```
CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME]
```

The defaults are `color`, `depth` and `sam3`. The separators are colons, because some launchers split arguments
at commas.

| Input | Requirement |
| --- | --- |
| The colour stream | Encoding `rgb8`, `rgba8`, `bgr8`, `bgra8` or `mono8`, with its `camera_info`. |
| The depth stream | In the geometry of the colour camera, any size, `16UC1` in mm or `32FC1` in m. |
| The segmenter | An xblick-sam3 instance on the colour stream. |

Examples of camera names:

| Camera | Argument |
| --- | --- |
| Azure Kinect | `test/camera:depth=depth_to_color` |
| rc_visard, with the depth of xblick-fast-foundationstereo | `robot/rc_visard:color=left:depth=ffs/depth` |

With `--no-depth` the node tracks in 2D only. With `--no-segmenter` it follows only points that you give by hand.

**Which frames the node takes.** The node pairs each colour frame with the depth frame whose stamp is nearest,
within `--max-offset-ms`. It sends each pair through the tracker once and takes the newest pair first. It leaves
out the frames that arrived while it was busy, and it counts them. The node processes only frames whose stamp is
newer than the last frame that it tracked.

## The cloud on `<camera>/tapnext/tracks`

The node publishes one point cloud per frame, as a Durchblick point cloud stream (`cloud`, `cloud/shm/<host>`).
[Point clouds](https://github.com/xblick/xblick/blob/main/docs/guides/point-clouds.md) describes how to read such a stream. The cloud has the stamp and the
camera frame of its colour frame.

The cloud has height 1. Its width is the number of tracks, and a cloud without tracks is empty. The points are in
the order of their track ids.

| Field | Type | Content |
| --- | --- | --- |
| `x, y, z` | float32 | The point in metres, in the optical frame of the camera. NaN where the depth image has no value, and while the point is hidden. |
| `u, v` | float32 | The point in pixels of the colour image. Pixel centres are at integers. |
| `track_id` | uint32 | The id of the track. The node never uses an id twice while it works. |
| `object_id` | uint16 | The instance id of the segmenter on which the node seeded the point. It is the value in `sam3/instances`. 0 marks a point given by hand. |
| `visible` | uint8 | 1 while the tracker sees the point in this frame. |

A point has 32 bytes. With 1024 tracks, a cloud has 32 KB.

**Hidden tracks.** The node also publishes hidden tracks. Their `visible` is 0, their `u, v` is the guess of the
tracker, and their `x, y, z` is NaN. TAPNext++ finds a point again after an occlusion, so a consumer keeps its
track ids.

**Trajectories.** Each cloud is the state at its frame. A consumer builds a trajectory by collecting the points of
one track id.

`--node` changes the `tapnext` part of the topic names.

### Read the tracks

```python
import xblick as db

tracks = db.PointCloudSubscriber(session, "test/camera/tapnext/tracks")
cloud = tracks.latest(timeout=1.0).numpy()[0]       # structured array: cloud["track_id"], cloud["x"], ...
seen = cloud[cloud["visible"] == 1]
```

## How a track begins

**Seeding.** The node looks at each object in a result of the segmenter. If fewer than half of
`--points-per-object` (default 32) of the object's tracks are visible inside its mask, the node adds points up to
that number. It places a new point inside the mask, at least `--margin` (default 4) pixels from the edge. It
spreads the points: each new point is the mask pixel that is farthest from the object's existing points and from
the new points before it. The code is in `seeding.py`.

**Late masks.** A result of the segmenter arrives some frames after its colour frame. A point must enter the
tracker in the frame in which it lies on the object. The node therefore keeps the last `--history` frames
(default 60). It follows the new points from the frame of the result to the newest frame, in a separate model
state. Then it hands them to the tracker as points of the newest frame. It drops the points that are not visible
there. [docs/design.md](design.md#late-masks-and-the-catch-up) explains this catch-up.

**By hand.** You add points through the queryable, see below.

## How a track ends

The rules are in `tracks.py`. A track ends in each of these cases.

| Case | Rule |
| --- | --- |
| Hidden | The track is hidden for `--forget` seconds (default 6). |
| Object gone | The object of the track is absent from the segmenter's results for `--forget` seconds. |
| Drifted | The track is visible but outside the mask of its object in 2 results in a row. The node grows the mask by `--margin` for this check. |
| Too many tracks on one object | An object has more than twice `--points-per-object` tracks. Its tracks that are hidden longest end. |
| No free track | New points arrive and all `--max-tracks` are in use. The tracks that are hidden longest end, whatever their object. |
| Cleared | You send `{"clear": true}`. |

**An object with a new id keeps its tracks.** The segmenter gives an object a new id when it sees the object again
after its own tracker lost it. The node looks at the tracks of objects that are absent from a result. If the
tracker saw such a track inside the mask of an object in the result, the track now belongs to that object. The
object gets new points only by the normal seeding rule.

## The queryable `<camera>/tapnext/queries`

A query without a payload returns the state of the node. A query with a JSON payload is one of three requests.

| Request | Effect |
| --- | --- |
| `{"points": [[u, v], ...], "stamp": 1730000000123456789}` | Follow these pixels of the colour image. |
| `{"source": "sam3"}` or `{"source": "none"}` | Seed the objects of the camera's segmenter, or stop seeding. |
| `{"clear": true}` | End all tracks. |

Rules for `points`:

- `stamp` is the stamp of the colour frame of the points, in nanoseconds. The frame must be one of the node's last
  `--history` frames. Without `stamp`, the points refer to the newest frame.
- The node answers once the points are in the tracker, which is after the next frame.
- The answer is `{"track_ids": [...], "stamp": ...}`. It has one track id per point, and the stamp of the frame
  into which the node put the points. An id is null for a point that the tracker did not see in that frame.
- The tracks have `object_id` 0. They end by the same rules as all tracks.

The node answers the other two requests at once, with its state. It answers a bad request with an error that
states the reason.

The client `xblick-tapnext-queries` sends these queries.

```bash
.venv/bin/xblick-tapnext-queries --camera test/camera                              # the state
.venv/bin/xblick-tapnext-queries --camera test/camera --json '{"points": [[640, 360], [700, 400]]}'
.venv/bin/xblick-tapnext-queries --camera test/camera --json '{"source": "none"}'  # no more seeding
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera` | required | The base name of the camera. |
| `--node` | `tapnext` | The node's part of the topic name. |
| `--json` | | The request. Without it, the client asks for the state. |
| `--timeout` | 10 | The seconds to wait for the answer. |

## The status

The node publishes the status `tapnext` on `<camera>/diagnostics` once a second. It contains:

- the frames per second and the milliseconds per frame,
- the tracks and the visible tracks per frame,
- the number of seeded points,
- the time, the points and the frames of the catch-up,
- the tracks that drifted and the tracks that an object with a new id adopted,
- the hidden tracks that ended for new points,
- the results of the segmenter,
- the frames left out,
- the points without a free track,
- the seeding source.

## Options of `xblick-tapnext`

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera` | required | The camera, as described under "Inputs". |
| `--no-depth` | off | 2D only. The node reads no depth stream, and `x, y, z` are NaN. |
| `--no-segmenter` | off | No segmenter. The node follows only points given by hand. |
| `--max-offset-ms` | half the depth's frame period | The largest difference between the stamps of a colour frame and its depth frame. `0` demands equal stamps. |
| `--node` | `tapnext` | The part of the topic names after the camera. |
| `--points-per-object` | 32 | The number of points with which the node follows an object. Live: it changes while the node runs. |
| `--forget` | 6 | The seconds for which a track may be hidden, or its object unseen, before it ends. Live: it changes while the node runs. |
| `--max-tracks` | 1024 | The highest number of tracks. These are the slots of the tracker. |
| `--margin` | 4 | The pixels that a new point keeps from the edge of its mask. A track may also be this far outside its mask. Live: it changes while the node runs. |
| `--checkpoint` | `256` | `256` or `512`, the input resolution at which DeepMind fine-tuned the checkpoint. You can also give a `.pt` file of a 256 checkpoint. |
| `--compile MODE` | off | Compile the network with `torch.compile` in this mode during the first frames. |
| `--history` | 60 | The number of frames that the node keeps. Results of the segmenter and queries refer to them by stamp. |
| `--device` | `cuda` | The GPU of the node. |
| `--max-hz` | every frame | The highest number of colour frames per second. `0` or `none` also takes every frame. |
| `--local` | `auto` | The transport for a camera on the same host: `auto`, `shm` or `cuda`. |
| `--connect`, `--listen`, `--config` | the router of the PC | The Zenoh session, as for every Durchblick program. |

**`--max-hz`.** The default is every frame, because a tracker follows a point best from frame to frame. With a
limit, the node takes at most that many colour frames per second, and always the newest one. It decodes a depth
frame only for a colour frame that it takes. The report in the log states the rate in effect.

**`--local`.** With `auto`, the node reads a camera on the same host from its CUDA ring if that ring is on the
node's GPU. Otherwise it reads from the host ring in shared memory. `shm` and `cuda` force one of the two. With
`cuda`, the node stops if the camera offers no CUDA ring. The start log and every report state which transport
each stream uses.

**MPS.** If you set `CUDA_MPS_PIPE_DIRECTORY`, the node expects an MPS daemon and stops at start when none answers.
[Sharing a GPU](https://github.com/xblick/xblick/blob/main/docs/guides/sharing-a-gpu.md) explains MPS.

## Parameters

Every option is also a parameter of the node `<camera>/tapnext`. The live ones change while the node runs:
`points_per_object`, `forget`, `margin`; the tables mark them. The others are read-only and apply at the next start,
where `--param NAME:=VALUE` or `--params-file FILE` set them. Durchblick's terminal interface (`xblick`, tab
Parameters) and `xblick param` show them.
