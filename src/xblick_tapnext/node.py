"""xblick-tapnext: TAPNext++ (model.py) follows points of a camera's colour stream, frame by frame; by default the
points of a segmenter's objects (xblick-sam3's instances), lifted to 3D with the camera's depth.

    xblick-tapnext --camera cam/azurekinect:depth=depth_to_color [--no-depth] [--no-segmenter] \\
                       [--node tapnext] [--points-per-object 32] [--forget 6] \\
                       [--max-tracks 1024] [--margin 4] [--checkpoint 256 | 512 | FILE] [--compile MODE] \\
                       [--history 60] [--max-offset-ms 20] [--local shm|cuda] [--no-download]
                       [--connect ...] [--listen ...] [--config FILE]   (default: the PC's Zenoh router)

The camera is Durchblick's CameraSpec, ``CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME]`` (defaults color,
depth, sam3; the Azure Kinect's depth in its colour camera's geometry is depth_to_color). Each colour frame (with
depth, each that has its depth: colour and depth of one stamp) goes through the tracker
once, the newest first: frames that came while it was busy are left out (counted). For each it publishes, stamped as
the frame and in its camera frame:

* ``<camera>/<node>/tracks``  a point cloud stream (Durchblick's points.py: ``cloud``, ``cloud/shm/<host>``,
  ``cloud/cuda/<host>/<gpu>``),
  one point per track: x, y, z (float32, metres in the camera's optical frame; NaN without depth there or while the
  point is hidden), u, v (float32, pixels of the colour image, centres at integers), track_id (uint32, never used twice
  while the node runs), object_id (uint16: the segmenter's instance id it was seeded on, its value in the instance
  image; 0 given by hand), visible (uint8: 1 the tracker sees it in this frame). Height 1, width the number of tracks
  (0: an empty cloud), in the order of their track ids.

Tracks begin on the segmenter's objects (seeding.py: --points-per-object spread over its mask, --margin inside it) or by
hand over the queryable ``<camera>/<node>/queries`` (queries.py), which also switches the seeding and clears the tracks
(answered at once, also while the model works; points within QUERY_WAIT_S); they end as tracks.py tells (hidden or their
object unseen for --forget seconds, drifted out of their mask). A segmenter's mask comes some frames after its frame:
its new points are followed from that frame to the newest in a state of their own first (model.Catchup), then go into
the tracker's. Frames per second, the time a frame takes, tracks and visible tracks go to ``<camera>/diagnostics``
(status ``<node>``) once a second.

Weights (weights.py; in the node's cache directory ~/.cache/xblick/tapnext, or --weights-dir): a missing checkpoint is
downloaded at the start, before the model loads; --no-download ends the node instead. ``xblick-tapnext --download-only
[--checkpoint 256|512]`` only downloads and checks it (no camera, no Zenoh, no GPU), for preparation scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import xblick as db
from xblick import mps, node, tensors, topics
from xblick import weights as xblick_weights
from xblick.camera import check_local
from xblick.history import FrameHistory, PartnerHistory, StampedHistory
from xblick.msg import PointField
from xblick.rgbd import CameraSpec, check_same_camera
from xblick.session import add_arguments, open_from_args

from . import weights
from .queries import QUERIES, parse

POINT_STEP = 32
FIELDS = [PointField("x", 0, PointField.FLOAT32), PointField("y", 4, PointField.FLOAT32),
          PointField("z", 8, PointField.FLOAT32), PointField("u", 12, PointField.FLOAT32),
          PointField("v", 16, PointField.FLOAT32), PointField("track_id", 20, PointField.UINT32),
          PointField("object_id", 24, PointField.UINT16), PointField("visible", 26, PointField.UINT8)]
QUERY_WAIT_S = 2.0  # a points query is answered within this: its track ids, or an error and no tracks
FREE_SLOTS = ("decoys", "future")  # --free-slots
CATCHUP_STRIDE = 2  # --catchup-stride: on TUM no worse than 1 (docs/performance.md), a quarter fewer steps
MAX_CATCHUP = 30  # --max-catchup: one second at 30 Hz
MAX_OFFSET_MS = 20.0  # --max-offset-ms: one capture's colour and depth (an Azure Kinect: up to 17 ms apart), not two


def parse_device(text: str) -> str:
    """``--device``: cuda (GPU 0) or cuda:N, as ``cuda:N``."""
    if text == "cuda":
        return "cuda:0"
    if text.startswith("cuda:") and text[5:].isdigit():
        return text
    raise argparse.ArgumentTypeError(f"{text!r}: cuda (GPU 0) or cuda:N")


def cloud(xyz: np.ndarray, uv: np.ndarray, track_id: np.ndarray, object_id: np.ndarray, visible: np.ndarray):
    """The tracks as one row of points (FIELDS)."""
    from xblick.points import points_dtype  # noqa: PLC0415

    points = np.zeros((1, len(uv)), points_dtype(FIELDS, POINT_STEP))
    for name, values in (("x", xyz[:, 0]), ("y", xyz[:, 1]), ("z", xyz[:, 2]), ("u", uv[:, 0]), ("v", uv[:, 1]),
                         ("track_id", track_id), ("object_id", object_id), ("visible", visible)):
        points[name][0] = values
    return points


# the parameters that change while the node runs (Parameters.declare_options); every other option applies at the start
LIVE = {"points_per_object": {"range": (1, 1024, 1), "unit": "points"},
        "refill": {"range": (0.0, 1.0, 0.05)},
        "catchup_stride": {"range": (1, 8, 1), "unit": "frames"},
        "max_catchup": {"range": (0, 600, 1), "unit": "steps"},
        "forget": {"range": (0.5, 600.0, 0.5), "unit": "s"},
        "margin": {"range": (0, 64, 1), "unit": "pixels"}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xblick-tapnext", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True,
                        help="CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME], e.g. "
                             "cam/azurekinect:depth=depth_to_color or cam/rc_visard:color=left:depth=ffs/depth; "
                             "the depth in the colour camera's geometry (16UC1 mm or 32FC1 m), the segmenter a "
                             "xblick-sam3 instance on the colour stream")
    parser.add_argument("--no-depth", action="store_true", help="2D only: no depth stream, x, y, z NaN")
    parser.add_argument("--no-segmenter", action="store_true", help="no segmenter: points by hand only")
    parser.add_argument("--max-offset-ms", type=float, default=MAX_OFFSET_MS,
                        help="pair a colour frame with the nearest depth frame at most this far apart, else skip it "
                             "(counted); 20 covers the Azure Kinect's colour and depth of one capture (up to 17 ms "
                             "apart) and the rc_visard's (equal stamps), and pairs no neighbouring frame of a 30 or "
                             "15 Hz depth stream (default: %(default)s); 0: only equal stamps")
    parser.add_argument("--node", default="tapnext", help="the results' part of the topic: <camera>/<node>/...")
    parser.add_argument("--points-per-object", type=int, default=32, help="points an object is followed by")
    parser.add_argument("--forget", type=float, default=6.0,
                        help="seconds a track may be hidden, and its object unseen, before it ends")
    parser.add_argument("--max-tracks", type=int, default=1024,
                        help="tracks at most (the tracker's slots; 1024 cost a fifth more time than 64)")
    parser.add_argument("--margin", type=int, default=4,
                        help="pixels new points keep from their mask's edge, and a track may be outside its mask")
    parser.add_argument("--checkpoint", default="256",
                        help="TAPNext++'s: 256 or 512 (the input resolution it was fine-tuned at; in the cache "
                             "directory, downloaded at the first start; README, \"Speed\") or a .pt file of a 256 one")
    parser.add_argument("--compile", default=None, metavar="MODE",
                        help="torch.compile the network in this mode (e.g. default: about twice as fast; compiles "
                             "at the first frames, about a minute)")
    parser.add_argument("--free-slots", choices=FREE_SLOTS, default="decoys",
                        help="what a free slot of the tracker holds: decoys (a real query at a point of the image; "
                             "many free slots then do not degrade the tracks, docs/design.md) or future (a query that "
                             "has not come yet, as before 0.2.6)")
    parser.add_argument("--refill", type=float, default=0.5,
                        help="an object gets new points when fewer than this share of --points-per-object of its "
                             "tracks are visible in its mask (tracks it adopted count)")
    parser.add_argument("--catchup-stride", type=int, default=CATCHUP_STRIDE,
                        help="a late mask's points catch up over every n-th kept frame (and their own): fewer model "
                             "steps, larger steps between frames (docs/performance.md)")
    parser.add_argument("--max-catchup", type=int, default=MAX_CATCHUP,
                        help="the most model steps a catch-up may take: points of an older mask are dropped (counted); "
                             "0: no bound")
    parser.add_argument("--history", type=int, default=60,
                        help="frames kept for the segmenter's results and queries to refer to (by stamp)")
    parser.add_argument("--device", type=parse_device, default="cuda:0",
                        help="the GPU: cuda (GPU 0) or cuda:N; a GPU that cannot be used is an error (default: cuda:0)")
    xblick_weights.add_arguments(parser)  # --download-only (a mode of the command line), --no-download
    add_arguments(parser)  # --connect, --listen, --config; default: the PC's Zenoh router
    # --max-hz: every frame by default (a tracker follows best frame by frame)
    # --local: auto (the CUDA ring on this GPU, else shared memory), shm or cuda
    node.add_arguments(parser, local=("auto", "shm", "cuda"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    args = xblick_weights.parse_args(parser, argv)  # --download-only: --camera not required
    if args.download_only:
        return weights.store_for(parser, args).download_only(
            weights.needed(args.checkpoint), options=weights.options(args))
    mps.check()  # with CUDA_MPS_PIPE_DIRECTORY (MPS meant) its daemon has to run: else stop here
    # the options as parameters (the terminal interface shows them): read-only, but LIVE
    session = open_from_args(args)
    params = node.Parameters(session, args.node, args.camera.split(":", 1)[0])
    params.configure(args)
    params.declare_options(parser, args, live=LIVE, leave_out=("camera", "node", "download_only"))
    params.check_overrides()
    if args.points_per_object < 1 or args.max_tracks < 1 or args.forget <= 0 or args.history < 2 or args.margin < 0 \
            or args.max_offset_ms < 0 or args.catchup_stride < 1 or args.max_catchup < 0 \
            or not 0 <= args.refill <= 1:
        parser.error("--points-per-object, --max-tracks, --catchup-stride: at least 1; --forget: positive; --history: "
                     "at least 2; --margin, --max-offset-ms, --max-catchup: not negative; --refill: 0 to 1")
    try:
        spec = CameraSpec.parse(args.camera, segmenter=None if args.no_segmenter else "sam3")
    except ValueError as error:
        parser.error(f"--camera: {error}")
    if args.no_segmenter and spec.segmenter is not None:
        parser.error("--no-segmenter with a segmenter in --camera")
    camera = topics.check_name(spec.camera)
    keys = dict(spec.keys)
    if args.no_depth:
        del keys["depth"]
    for k in keys.values():
        topics.check_name(k)

    node.start("xblick-tapnext", max_hz=args.max_hz, local=args.local)
    store = weights.store_for(parser, args)  # --weights-dir, also given as a parameter
    try:
        store.ensure(weights.needed(args.checkpoint), download=not args.no_download, options=weights.options(args))
    except xblick_weights.WeightsError as error:
        params.close()
        session.close()
        return store.fail(error)

    import torch  # noqa: PLC0415 - with the model, after the arguments are checked

    from .lifting import Lifter  # noqa: PLC0415
    from .model import Catchup, SlotTracker, load, to_pixels  # noqa: PLC0415
    from .seeding import grow_labels, sample  # noqa: PLC0415
    from .tracks import Tracks, catchup_plan  # noqa: PLC0415

    device = torch.device(args.device)
    if device.index >= torch.cuda.device_count():  # no CUDA, or not that many GPUs: one line, never another device
        print(f"xblick-tapnext: --device {args.device}: no such GPU ({torch.cuda.device_count()} visible)",
              file=sys.stderr, flush=True)
        params.close()
        session.close()
        return 1
    torch.cuda.set_device(device)
    net, resolution = load(args.checkpoint, device, store.root)
    if args.compile is not None:
        net = torch.compile(net, mode=args.compile)
    forget_ns = round(args.forget * 1e9)

    wake = threading.Event()

    def subscriber(key, camera_info, max_hz=None):
        return db.ImageSubscriber(session, keys[key], transport="auto", camera_info=camera_info, local=args.local,
                                  max_hz=max_hz)

    def reader(key, camera_info, max_hz=None):
        return FrameHistory(subscriber(key, camera_info, max_hz), args.history, wake.set)

    color = reader("color", True, args.max_hz)  # --max-hz limits the colour frames, the depth follows them
    # the depth nearest each colour stamp within --max-offset-ms, decoded only for the colour frames tracked
    max_offset_ns = round(args.max_offset_ms * 1e6)
    depth = (PartnerHistory(subscriber("depth", True), args.history, max_offset_ns, wake.set)
             if "depth" in keys else None)
    instances = reader("instances", False) if "instances" in keys else None
    detections = None
    if "detections" in keys:
        detections_history = StampedHistory(args.history, wake.set)
        detections = db.DetectionSubscriber(session, keys["detections"],
                                            lambda m: detections_history.put(m.header.stamp.to_ns(), m))
    tracks_pub = None  # with the first frame: its camera frame
    report = node.Reporter(session, camera, args.node, "xblick-tapnext", max_hz=args.max_hz)
    report.watch(color)
    counts = report.counts
    source = "sam3" if instances is not None else "none"
    print(f"xblick-tapnext: {' + '.join(keys.values())} -> {camera}/{args.node}/tracks (TAPNext++ "
          f"{args.checkpoint}, input {resolution} x {resolution}, {args.max_tracks} tracks at most, "
          f"{args.points_per_object} per object); model on {args.device}, lifting on the CPU", flush=True)

    state = {}  # the camera's model and everything made for it (reset when it changes)

    def reset(info, frame_id):
        state.clear()
        tracker = SlotTracker(net, resolution, args.max_tracks, device, decoys=args.free_slots == "decoys")
        state.update(key=(info.width, info.height, frame_id), tracker=tracker,
                     tracks=Tracks(args.max_tracks, args.points_per_object, forget_ns, args.refill),
                     lifter=Lifter(info),
                     prepared=OrderedDict(), positions=StampedHistory(args.history), seen=-1)
        state["catchup"] = Catchup(state["tracker"])

    depth_max = [0]  # the deepest catch-up (model steps) since the last report
    seeds = []  # (stamp, object id, (k, 2) pixels of that frame, Waiting or None)
    done = -1  # the stamp of the newest frame looked at
    # the queryable answers in Zenoh's thread from ``view``, which the loop keeps; what it accepts goes to ``inbox``
    # for the loop; a points query waits in ``points_waiting`` (answered by the loop, or by the watchdog at its
    # deadline)
    query_lock = threading.Lock()
    view = {"state": None, "kept": set(), "done": -1}
    inbox = []  # ("source", value), ("clear", None), ("points", Waiting)
    points_waiting = []
    stop_watchdog = threading.Event()

    def describe() -> dict:
        tracks = state.get("tracks")
        used = tracks.used if tracks is not None else np.zeros(0, bool)
        objects = sorted({int(o) for o in tracks.object_id[used]}) if tracks is not None else []
        return {"source": source, "segmenter": spec.segmenter, "points_per_object": args.points_per_object,
                "forget": args.forget, "tracks": int(used.sum()), "max_tracks": args.max_tracks, "objects": objects}

    def end(slots):
        if len(slots):
            state["tracker"].free(torch.as_tensor(slots, device=device))
            state["tracks"].end(slots)
            counts["freed"] += len(slots)

    key = f"{camera}/{args.node}/{QUERIES}"

    class Waiting:
        """A points query until the loop put its points in the tracker (answered with their track ids) or its
        deadline passed (the watchdog answers with an error, and the loop then makes no tracks of it)."""

        def __init__(self, query, stamp, points):
            self.query, self.stamp, self.points = query, stamp, points
            self.deadline, self.answered = time.monotonic() + QUERY_WAIT_S, False

        def claim(self) -> bool:
            """Takes the answer on (under query_lock) unless it is taken already; False then."""
            if self.answered:
                return False
            self.answered = True
            return True

        def send(self, text: str | None, error: str | None = None):
            """The answer (after claim), and the query dropped."""
            if error is None:
                self.query.reply(key, text)
            else:
                self.query.reply_err(error)
            self.query = None

    def keep_view():
        """The state the queryable answers from: after each frame and each change (in this thread)."""
        with query_lock:
            view.update(state=describe(), kept=set(state.get("prepared", ())), done=done)

    def answer(query):
        """Answers at once, in Zenoh's thread: the state, or a request checked and accepted (the loop applies it
        before the next frame; the answer is the state from then on); points are answered by the loop or the
        watchdog within QUERY_WAIT_S."""
        try:
            payload = query.payload.to_string() if query.payload is not None else ""
            request = parse(json.loads(payload)) if payload.strip() else {}
            with query_lock:
                reply = dict(view["state"])
                if "source" in request:
                    if request["source"] == "sam3" and instances is None:
                        raise ValueError("the node has no segmenter (--no-segmenter)")
                    inbox.append(("source", request["source"]))
                    reply["source"] = request["source"]
                elif "clear" in request:
                    inbox.append(("clear", None))
                    reply.update(tracks=0, objects=[])
                elif "points" in request:
                    if view["done"] < 0:
                        raise ValueError("no frame tracked yet")
                    stamp = request["stamp"] if request["stamp"] is not None else view["done"]
                    if stamp not in view["kept"]:
                        raise ValueError(f"no tracked frame of stamp {stamp} (the node keeps its last {args.history})")
                    entry = Waiting(query, stamp, request["points"])
                    inbox.append(("points", entry))
                    points_waiting.append(entry)
                    return  # answered with the track ids once they are in the tracker, or with an error in time
            query.reply(key, json.dumps(reply))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            query.reply_err(f"{type(error).__name__}: {error}")

    def watchdog():
        """Answers points queries at their deadline with an error (the model was busy, or no frame came)."""
        while not stop_watchdog.wait(0.02):
            now = time.monotonic()
            with query_lock:
                for entry in [w for w in points_waiting if w.deadline <= now and w.claim()]:
                    entry.send(None, f"the points were not in the tracker within {QUERY_WAIT_S} s (the model was "
                                       "busy, or no colour frame came): no tracks were made of them")
                points_waiting[:] = [w for w in points_waiting if not w.answered]

    def apply_requests():
        """The requests accepted on the queryable, applied in this thread (between frames)."""
        nonlocal source
        with query_lock:
            taken = list(inbox)
            inbox.clear()
        for what, value in taken:
            if what == "source":
                source = value
            elif what == "clear":
                if "tracks" in state:
                    end(np.nonzero(state["tracks"].used)[0])
                seeds.clear()
            elif not value.answered:  # points: not answered by the watchdog meanwhile
                seeds.append((value.stamp, 0, torch.tensor(value.points, device=device), value))
        if taken:
            keep_view()

    def frame_to_track():
        """The newest stamp with a colour frame (and its depth) not yet tracked, and how many newer than the last
        tracked one are left out for it."""
        stamps = color.stamps() if depth is None else {s for s in color.stamps() if depth.has(s)}
        new = sorted(s for s in stamps if s > done)
        return (new[-1], len(new) - 1) if new else (None, 0)

    def put_seeds(stamp, f, height, width):
        """The pending seeds into the tracker: followed from their frames to this one (``stamp``, prepared ``f``),
        all in one state (each point queried in its own frame)."""
        tracker, tracks, prepared = state["tracker"], state["tracks"], state["prepared"]
        replies = []
        usable = []
        for seed in seeds:
            if seed[3] is not None and seed[3].answered:
                continue  # its query was answered with an error at its deadline: no tracks of it
            if seed[0] in prepared:
                usable.append(seed)
            else:
                counts["seeds_too_old"] += 1
                if seed[3] is not None:
                    replies.append((seed[3], [], False, []))
        seeds.clear()
        # one pass for all waiting seeds, from the oldest; --max-catchup drops deeper ones, --catchup-stride thins
        stamps, dropped, skipped = catchup_plan(list(prepared), [seed[0] for seed in usable], args.catchup_stride,
                                                args.max_catchup)
        for seed in usable:
            if seed[0] in dropped:
                counts["catchup_dropped"] += 1
                if seed[3] is not None:
                    replies.append((seed[3], [], False, []))
        usable = [seed for seed in usable if seed[0] not in dropped]
        counts["catchup_skipped"] += skipped
        if not usable:
            return replies
        index = {s: i for i, s in enumerate(stamps)}
        xy = torch.cat([seed[2] for seed in usable])
        starts = torch.cat([torch.full((len(seed[2]),), index[seed[0]], device=device) for seed in usable])
        began = time.monotonic()
        moved, visible = state["catchup"].run([prepared[s] for s in stamps] + [f], xy, height, width, starts)
        counts["catchup_passes"] += 1
        depth_max[0] = max(depth_max[0], len(stamps) + 1)  # counts are totals (Reporter.take): the maximum apart
        counts["objects_seeded"] += len(usable)
        inside = (moved[:, 0] >= 0) & (moved[:, 0] <= width - 1) & (moved[:, 1] >= 0) & (moved[:, 1] <= height - 1)
        kept = (visible & inside).cpu().numpy()
        counts["catchup_busy"] += time.monotonic() - began
        counts["catchup_points"] += len(xy)
        offset = 0
        for _, object_id, points, query in usable:
            keep = kept[offset:offset + len(points)].copy()
            here = moved[offset:offset + len(points)]
            offset += len(points)
            if object_id:
                end(tracks.surplus(object_id, int(keep.sum())))
            free = tracks.free_slots()
            if len(free) < keep.sum():  # new points before the tracks hidden longest (tracks.py)
                room = tracks.hidden_longest(int(keep.sum()) - len(free))
                end(room)
                counts["made_room"] += len(room)
                free = tracks.free_slots()
            if len(free) < keep.sum():
                counts["no_room"] += int(keep.sum()) - len(free)
                keep[np.nonzero(keep)[0][len(free):]] = False
            slots = free[:int(keep.sum())]
            started = tracks.start(slots, object_id, stamp)
            tracker.add(torch.as_tensor(slots, device=device), here[torch.as_tensor(keep, device=device)], height,
                        width)
            counts["seeded"] += len(slots)
            if query is not None:
                it = iter(started.tolist())
                replies.append((query, [next(it) if k else None for k in keep], True, slots))
        counts["catchup_frames"] += len(stamps) + 1  # the model's steps: the kept frames since the oldest, and this one
        return replies

    def track(stamp, skipped):
        nonlocal tracks_pub, done
        done = stamp
        frame = color.at(stamp)
        depth_frame = depth.at(stamp) if depth is not None else None  # both now: the histories move on
        if depth is not None and depth_frame is None:
            counts["stale"] += 1  # the depth's ring slot was overwritten before it was read
            return
        info = color.subscriber.camera_info
        if info is None or (info.height, info.width) != tuple(frame.image.shape[:2]):
            counts["no_info"] += 1
            return
        frame_id = info.header.frame_id or frame.header.frame_id
        if state.get("key") != (info.width, info.height, frame_id) or Lifter.key_of(info) != state["lifter"].key:
            if state:
                print(f"xblick-tapnext: {keys['color']} is now {info.width} x {info.height} in {frame_id}: the "
                      "tracks start anew", flush=True)
            reset(info, frame_id)
            seeds.clear()
        if tracks_pub is None:
            tracks_pub = db.PointCloudPublisher(session, f"{camera}/{args.node}/tracks", fields=FIELDS,
                                                point_step=POINT_STEP, frame_id=frame_id, source="tapnext")
        tracks_pub.frame_id = frame_id
        check_local(color.subscriber, args.local)
        started = time.monotonic()
        tracker, tracks, prepared = state["tracker"], state["tracks"], state["prepared"]
        height, width = info.height, info.width
        f = tracker.prepare(tensors.rgb(frame, device))
        metres = tensors.depth_metres(depth_frame, device) if depth_frame is not None else None
        replies = put_seeds(stamp, f, height, width)
        yx, visible = tracker.step(f)
        answers = []  # the points queries: claimed now (sent after the frame), unless the watchdog answered them
        with query_lock:
            for entry, ids, ok, slots in replies:
                if not entry.claim():
                    end(slots)  # answered with an error at its deadline: no tracks of these points
                elif ok:
                    answers.append((entry, json.dumps({"track_ids": ids, "stamp": stamp}), None))
                else:
                    answers.append((entry, None, f"the frame {entry.stamp} is no longer kept"))
            points_waiting[:] = [w for w in points_waiting if not w.answered]
        prepared[stamp] = f
        while len(prepared) > args.history:
            prepared.popitem(last=False)
        uv = to_pixels(yx[:args.max_tracks], height, width).cpu().numpy()
        visible = visible[:args.max_tracks].cpu().numpy()
        tracks.update(visible, stamp)
        end(tracks.expired(stamp))
        used = tracks.used
        state["positions"].put(stamp, (uv, visible & used, tracks.track_id.copy()))
        order = np.nonzero(used)[0]
        order = order[np.argsort(tracks.track_id[order])]
        xyz = np.full((len(order), 3), np.nan)
        if metres is not None and len(order):
            depth_info = depth.subscriber.camera_info
            if depth_info is not None:
                if state.get("depth_key") != (tuple(depth_info.k), depth_info.width, depth_info.height):
                    check_same_camera(info, depth_info)
                    state["depth_key"] = (tuple(depth_info.k), depth_info.width, depth_info.height)
                rows, cols, inside = state["lifter"].depth_pixels(uv[order], tuple(metres.shape))
                z = metres[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)].cpu().numpy()
                z = np.where(inside & visible[order], z, np.nan)
                xyz = state["lifter"].lift(uv[order], z)
            else:
                counts["no_depth_info"] += 1
        tracks_pub.publish(cloud(xyz, uv[order], tracks.track_id[order], tracks.object_id[order],
                                 visible[order].astype(np.uint8)), stamp_ns=stamp)
        counts["frames"] += 1
        counts["skipped"] += skipped
        counts["tracks"] += len(order)
        counts["visible"] += int(visible[order].sum())
        counts["busy"] += time.monotonic() - started
        return answers

    def check_segmenter():
        """The newest segmenter result of a tracked frame not looked at yet: drift, presence and new points."""
        if not state or instances is None:
            return
        positions = state["positions"]
        stamps = instances.stamps() & detections_history.stamps() & positions.stamps()
        new = [s for s in stamps if s > state["seen"]]
        if not new:
            return
        stamp = max(new)
        counts["results_passed_over"] += len(new) - 1  # a newer result came before these were looked at
        state["seen"] = stamp
        message = detections_history.at(stamp)
        ids_frame = instances.at(stamp)
        if ids_frame.encoding != "mono16":
            raise ValueError(f"{keys['instances']}: {ids_frame.encoding}, mono16 expected")
        ids = tensors.tensor(ids_frame, device).to(torch.int32)  # CUDA indexes no uint16 tensors
        if tuple(ids.shape) != (state["key"][1], state["key"][0]):
            raise ValueError(f"{keys['instances']}: {tuple(ids.shape[::-1])}, the colour image's size "
                             f"{state['key'][:2]} expected")
        uv, visible, track_id = positions.at(stamp)
        tracks = state["tracks"]
        present = [int(d.id) for d in message.detections]
        counts["new_objects"] += sum(o not in tracks.objects for o in present)
        cols = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, ids.shape[1] - 1)
        rows = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, ids.shape[0] - 1)
        # all at the result's own frame: the tracks' positions then (positions), not the newest ones
        under = ids[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)].cpu().numpy()
        grown_ids = grow_labels(ids, args.margin)  # each pixel's object id, masks grown by the margin (0: none)
        under_grown = grown_ids[torch.as_tensor(rows, device=device),
                                torch.as_tensor(cols, device=device)].cpu().numpy()
        inside = (under_grown == tracks.object_id) | (under == tracks.object_id)
        masks = {o: ids == o for o in present}
        # a track its object does not claim (gone from the result, or outside its mask) and that lies in another
        # object's mask becomes that object's (tracks.py): no new points beside it
        unclaimed = tracks.unclaimed(present, visible, inside, track_id)
        if len(unclaimed):
            taken = unclaimed[np.isin(under[unclaimed], present) & (under[unclaimed] != tracks.object_id[unclaimed])]
            for o in np.unique(under[taken]):
                tracks.adopt(taken[under[taken] == o], int(o))
            inside[taken] = True
            counts["adopted"] += len(taken)
        drifted, needs = tracks.check(stamp, present, visible, inside, track_id)
        end(drifted)
        counts["drifted"] += len(drifted)
        counts["results"] += 1
        if source != "sam3":
            return
        pending = {seed[1] for seed in seeds}
        for o, n in needs.items():
            if o in pending:  # seeded from an earlier result, not yet in the tracker
                continue
            have = torch.as_tensor(uv[(tracks.object_id == o) & visible], device=device)
            xy = sample(masks[o], n, have, args.margin)
            if len(xy):
                seeds.append((stamp, o, xy, None))
                # why it needs points: its tracks at the result's frame (tracked then), hidden or outside its mask
                mine = tracks.used & (tracks.object_id == o) & (tracks.track_id == track_id)
                counts["refill_new"] += not mine.any()
                counts["refill_hidden"] += int((mine & ~visible).sum())
                counts["refill_outside"] += int((mine & visible & ~inside).sum())

    def apply_parameters():
        """The live parameters' values now, in this thread: the seeding's margin, and the tracks' points per object
        and forgetting (from the next object seeded and the next frame on)."""
        args.margin = params["margin"]
        args.refill, args.catchup_stride, args.max_catchup = (params["refill"], params["catchup_stride"],
                                                              params["max_catchup"])
        if state.get("tracks") is not None:
            state["tracks"].refill = args.refill
            state["tracks"].per_object = params["points_per_object"]
            state["tracks"].forget_ns = round(params["forget"] * 1e9)
        args.points_per_object, args.forget = params["points_per_object"], params["forget"]

    keep_view()
    queryable = session.declare_queryable(key, answer)  # answered at once, in Zenoh's thread
    guard = threading.Thread(target=watchdog, name="queries", daemon=True)
    guard.start()
    try:
        while True:
            wake.wait(0.05)
            wake.clear()
            apply_parameters()
            apply_requests()
            for history in (color, depth, instances):
                if history is not None:
                    history.check()
            if detections is not None:
                detections.check()
            stamp, skipped = frame_to_track()
            if stamp is not None:
                answers = track(stamp, skipped)
                keep_view()  # before the points' answers: a state asked after them has their tracks
                for entry, text, error in answers or ():
                    entry.send(text, error)
            check_segmenter()
            if report.due():
                d, seconds = report.take()
                n, ms = max(d["frames"], 1), d["busy"] / max(d["frames"], 1) * 1e3
                problems = ([f"{d['no_info']} frames without camera_info"] if d["no_info"] else []) + \
                    ([f"{d['no_depth_info']} frames without the depth's camera_info"] if d["no_depth_info"] else []) + \
                    ([f"{d['skipped']} frames left out (busy)"] if d["skipped"] else []) + \
                    ([f"{d['no_room']} points without a free track"] if d["no_room"] else []) + \
                    ([f"{d['seeds_too_old']} seeds of frames no longer kept"] if d["seeds_too_old"] else []) + \
                    ([f"{d['catchup_dropped']} seeds too deep to catch up"] if d["catchup_dropped"] else [])
                waiting = None if d["frames"] else "; ".join(
                    f"{keys[k]}: {r.subscriber.describe()}" for k, r in
                    (("color", color), ("depth", depth)) if r is not None)
                level, message = node.state(f"{d['frames'] / seconds:.2f} frames/s", problems, waiting)
                report.publish(level, message,
                               {"frames/s": d["frames"] / seconds, "ms/frame": ms, "tracks": d["tracks"] / n,
                                "visible": d["visible"] / n, "seeded": d["seeded"], "drifted": d["drifted"],
                                "segmenter results": d["results"], "frames left out": d["skipped"],
                                "points without a free track": d["no_room"], "tracks adopted": d["adopted"],
                                "hidden tracks ended for new points": d["made_room"],
                                "catch-up frames": d["catchup_frames"], "catch-up passes": d["catchup_passes"],
                                "catch-up depth max": depth_max[0], "new objects": d["new_objects"],
                                "objects seeded": d["objects_seeded"], "tracks ended": d["freed"],
                                "results passed over": d["results_passed_over"],
                                "catch-up dropped": d["catchup_dropped"],
                                "objects seeded without tracks": d["refill_new"],
                                "hidden tracks of objects seeded": d["refill_hidden"],
                                "tracks outside the mask of objects seeded": d["refill_outside"],
                                "catch-up frames skipped": d["catchup_skipped"],
                                "catch-up points": d["catchup_points"], "catch-up ms": d["catchup_busy"] * 1e3,
                                "source": source},
                               log_extra=f", {ms:.0f} ms a frame (catch-up {d['catchup_busy'] * 1e3:.0f} ms of it, "
                                         f"{d['catchup_points']} points over {d['catchup_frames']} frames), "
                                         f"{d['tracks'] / n:.0f} tracks, {d['visible'] / n:.0f} visible, "
                                         f"{d['adopted']} adopted, {d['made_room']} hidden ended for new points")
                depth_max[0] = 0
    except KeyboardInterrupt:
        pass
    finally:
        for history in (color, depth, instances):
            if history is not None:
                history.close()  # and its subscriber
        for closable in (detections, tracks_pub, report):
            if closable is not None:
                closable.close()
        queryable.undeclare()
        stop_watchdog.set()
        guard.join(timeout=1.0)
        params.close()
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
