# Performance

This page states how fast and how accurate xblick-tapnext is, and how we measured both. The
[README](../README.md) has the short version.

## Speed

DeepMind state 5.05 to 5.33 ms per frame for TAPNext with 256 to 1024 points on an H100, per frame and online, with
`torch.compile` ("crucial for fast inference", TAPNext paper, Table 2), at 256 x 256; 5.18 to 5.23 ms for TAPNext++.
Their figures are for the compiled model. The node's compiled step is in the same range:

| Checkpoint | Eager | `--compile default` | VRAM |
| --- | --- | --- | --- |
| 256 | 10.7 ms | 5.6 ms | about 1.7 GB |
| 512 | 21 ms | 13.5 ms | about 2.6 GB |

Setup: an RTX PRO 6000, 512 slots, float16 autocast, time per frame, measured on 2026-11-02.

The number of slots (`--max-tracks`) changes the time of a step. The default of 1024 slots holds 32 objects at 32
points each. A scene with few objects is faster with fewer slots:

| `--max-tracks` | Eager | `--compile default` | Objects at 32 points each |
| --- | --- | --- | --- |
| 256 | 10.5 ms | 5.7 ms | 8 |
| 1024 (default) | 12.2 ms | 7.7 ms | 32 |

Setup: an RTX PRO 6000 that ran nothing else, 256 checkpoint, float16 autocast, the model step alone, median of 200
frames, measured on 2026-11-19. When all slots are in use, the tracks that are hidden longest end.

Three terms need an explanation.

- *Eager* is PyTorch's normal mode. Python executes the model operation by operation.
- *`torch.compile`* translates a PyTorch model into optimised code. The node compiles at its start, before it reads
  a stream, which takes one to two minutes (`--compile default`, the default; `--compile none` does not compile).
- *Autocast* is PyTorch's mechanism that converts values to a 16-bit float format on the fly. A GPU computes
  16-bit floats faster than 32-bit floats.

### The cost of the slots

A *slot* is the place of one track in the model state. `--max-tracks` sets the number of slots.

| Slots | Eager |
| --- | --- |
| 64 | 10.8 ms |
| 1024 | 12.7 ms |

The slots cost little.

### Several trackers in one pass

Trackers of 64 slots, each on a frame of its own, compiled, one after the other (`SlotTracker.step` each) and in one
pass (`SlotTrackers.step`). Median ms per step of all trackers; RTX PRO 6000 Blackwell, no other job on the GPU, load
average 17 of 64 CPUs, 2026-11-22.

| Trackers | One after the other | In one pass |
| --- | --- | --- |
| 1 | 5.1 | 5.4 |
| 2 | 10.1 | 7.7 |
| 3 | 15.1 | 9.0 |
| 4 | 20.1 | 12.9 |
| 5 | 25.0 | 13.6 |

A step costs about 5 ms per tracker on its own and about 2 ms per further tracker in one pass. Most of a step is the
model's work on the 1024 patches of each frame, which grows with the number of frames; the pass saves the launches
and the reading of the weights. [docs/design.md](design.md#several-trackers-in-one-pass) explains the batch.

### The cost of seeding

New points catch up from the frame of their mask to the newest frame. This adds one model step per kept frame
since the frame of the mask, once per seeding, and a step costs about as much as a frame of the tracker: the model
works on all patches of the frame in every step. [docs/design.md](design.md#late-masks-and-the-catch-up) explains
the catch-up.

Measured on the end-to-end replay below (TUM, xblick-sam3 on the same GPU at about 15 results per second, both
without compiling), the catch-up took about a quarter of the node's time. Per second there were about 5 catch-up
passes of 3 to 4 model steps each (at most 5 to 7). Most seeding was not for new objects: about 7 objects a second
got new points, of them about 1 without any track. The others had enough tracks, but the model saw too few of them in
the result's frame.

`--catchup-stride 2` (the default) leaves out about a quarter of these steps. It did not change the accuracy of the
points it catches up. On the same sequence we caught up 48 points with depth from 200 random frames over 3, 5 and 8
frames and compared them with their true positions (the true poses and depth):

| Catch-up | Camera | Stride 1 | Stride 2 | Stride 3 |
| --- | --- | --- | --- | --- |
| 5 frames | still (up to 15 deg/s) | 1.51 px | 1.47 px | 1.47 px |
| 5 frames | moving (more than 15 deg/s) | 2.06 px | 2.02 px | 2.02 px |
| 8 frames | moving | 2.23 px | 2.10 px | 2.12 px |

The numbers are median errors. The camera turns at up to 67 deg/s in this sequence (90 % of the frames: up to
38 deg/s).

In a simulation of ARMAR-6's head sweeping over a table at up to 2.5 deg per frame (75 deg/s at 30 Hz), with objects
standing and on a turntable and their true poses as truth, stride 2 put the caught-up points at most 0.17 px further
off in the median and 0.3 px at the 90th percentile than stride 1, over 3 to 8 frames. Stride 3 was further off.

## Accuracy against ground truth

**Setup.** The sequence is `fr3/long_office_household` of TUM RGB-D. It has 2585 frames at 30 Hz with 640 x 480
pixels. A hand-held camera moves around a desk, and the data set has ground-truth poses.

**Method.**

1. Every 150 frames, we put 128 grid points that have depth into the working model state.
2. We end these points after 300 frames, and the node reuses their slots.
3. The truth for a point is its 3D position, projected into each frame with the true poses. The 3D position comes
   from its depth in its first frame.
4. A point counts as truly visible in a frame where the depth of the sensor agrees with the truth within 5 cm.
5. We also track the same points in a model state of their own per group. This is the way DeepMind's code works.

**Result.** The table states the median 2D error of the visible points, by the age of the point. Each cell has two
values: the slot state of the node, then a state per group.

| Age in frames | Checkpoint 256 | Checkpoint 512 |
| --- | --- | --- |
| 0-29 | 1.96 / 1.81 px | 1.97 / 1.81 px |
| 30-99 | 3.31 / 3.19 px | 3.33 / 3.15 px |
| 100-299 | 4.38 / 4.11 px | 4.27 / 4.01 px |

- The visibility agrees with the truth for 97-100 % of the points.
- The positions of the slot state lie 0.6-1.3 px (median) from the positions of the states per group.
- The truth itself has an error of about one or two pixels, from depth noise and pose timing.
- The checkpoint 512 gains nothing at 640 x 480. The default is therefore 256.

## End to end with xblick-sam3

**Setup (2026-11-02).** We replayed the first 900 frames of the same sequence at 30 Hz as the camera
`robot/camera`, with `color` and `depth_to_color`. xblick-sam3 used the words keyboard, cup, book,
monitor and teddy bear. Both nodes worked on one RTX PRO 6000 without compilation. They read the camera on the same
host from the host ring.

**3D consistency.** We put each lifted track into the world with the true poses. A static point should then stay
at one place.

| Measure | Value |
| --- | --- |
| Distance of a track's points from their median, median | 1.3 cm |
| The same, 90th percentile | 4.3 cm |
| Tracks in this statistic (at least 10 visible frames) | 2852 |
| Visible frames per track, median | 42 |

The distance contains the drift of the tracker, the noise of the depth and the snapping to pixels.

**Rates.**

| Measure | Value |
| --- | --- |
| Frames tracked | 643 of 900 |
| Rate of the node | 18-27 frames per second |
| Rate of xblick-sam3 | 15 frames per second |
| Time per frame | 19-40 ms |
| Catch-ups of new points | up to 0.3 s per second |
| Tracks at a time | up to 1024, of about 17 objects |
| Visible tracks | a third |

In this scene a hand-held camera passes a cluttered desk. xblick-sam3 gave the objects of one desk about
200 ids over the 30 s. 788 tracks moved to the new id of their object. 182 hidden tracks ended to make room for new
points.

Points given by hand through the queryable got track ids, and the node followed them.

## Status and limits

- The end-to-end numbers are without `--compile`.
- We measured accuracy only at 640 x 480.

[Making a node fast](https://github.com/xblick/xblick/blob/main/docs/guides/making-a-node-fast.md) compares these findings with Durchblick's other nodes.
