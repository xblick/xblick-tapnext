# Performance

This page states how fast and how accurate xblick-tapnext is, and how we measured both. The
[README](../README.md) has the short version.

## Speed

| Checkpoint | Eager | `--compile default` | VRAM |
| --- | --- | --- | --- |
| 256 | 10.7 ms | 5.6 ms | about 1.7 GB |
| 512 | 21 ms | 13.5 ms | about 2.6 GB |

Setup: an RTX PRO 6000, 512 slots, float16 autocast, time per frame, measured on 2026-11-02.

Three terms need an explanation.

- *Eager* is PyTorch's normal mode. Python executes the model operation by operation.
- *`torch.compile`* translates a PyTorch model into optimised code. The node compiles during the first frames,
  which takes about a minute.
- *Autocast* is PyTorch's mechanism that converts values to a 16-bit float format on the fly. A GPU computes
  16-bit floats faster than 32-bit floats.

### The cost of the slots

A *slot* is the place of one track in the model state. `--max-tracks` sets the number of slots.

| Slots | Eager |
| --- | --- |
| 64 | 10.8 ms |
| 1024 | 12.7 ms |

The slots cost little.

### The cost of seeding

New points catch up from the frame of their mask to the newest frame. This adds one tracker step per kept frame
since the frame of the mask, once per seeding. [docs/design.md](design.md#late-masks-and-the-catch-up) explains the
catch-up. In the end-to-end test below, the catch-ups took up to 0.3 s of each second.

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
