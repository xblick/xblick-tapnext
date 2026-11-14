"""xblick-tapnext: TAPNext++ (model.py) follows points of a camera's colour stream, frame by frame; by default the
points of a segmenter's objects (xblick-sam3's instances), lifted to 3D with the camera's depth.

    xblick-tapnext --camera cam/azurekinect:depth=depth_to_color [--no-depth] [--no-segmenter] \\
                       [--node tapnext] [--points-per-object 32] [--forget 6] \\
                       [--max-tracks 1024] [--margin 4] [--checkpoint 256 | 512 | FILE] [--compile MODE] \\
                       [--history 60] [--max-offset-ms 0] [--local shm|cuda]
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

Tracks begin on the segmenter's objects (seeding.py: --points-per-object spread over its mask, --margin inside it) or
by hand over the queryable ``<camera>/<node>/queries`` (queries.py), which also switches the seeding and clears the
tracks; they end as tracks.py tells (hidden or their object unseen for --forget seconds, drifted out of their mask). A
segmenter's mask comes some frames after its frame: its new points are followed from that frame to the newest in a
state of their own first (model.Catchup), then go into the tracker's. Frames per second, the time a frame takes,
tracks and visible tracks go to ``<camera>/diagnostics`` (status ``<node>``) once a second.
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
from xblick.camera import check_local
from xblick.history import FrameHistory, PartnerHistory, StampedHistory
from xblick.msg import PointField
from xblick.rgbd import CameraSpec, check_same_camera
from xblick.session import add_arguments, open_from_args

from .queries import QUERIES, parse

POINT_STEP = 32
FIELDS = [PointField("x", 0, PointField.FLOAT32), PointField("y", 4, PointField.FLOAT32),
          PointField("z", 8, PointField.FLOAT32), PointField("u", 12, PointField.FLOAT32),
          PointField("v", 16, PointField.FLOAT32), PointField("track_id", 20, PointField.UINT32),
          PointField("object_id", 24, PointField.UINT16), PointField("visible", 26, PointField.UINT8)]
QUERY_WAIT_S = 2.0  # how long a points query waits for a frame before it is answered with an error


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
    parser.add_argument("--max-offset-ms", type=float, default=None,
                        help="pair a colour frame with the nearest depth frame this close (default: half the "
                             "depth's frame period, measured: each colour frame with its nearest depth frame, as a "
                             "camera whose depth is stamped apart from its colour needs, one measured Azure Kinect: "
                             "11-17 ms); 0: only equal stamps")
    parser.add_argument("--node", default="tapnext", help="the results' part of the topic: <camera>/<node>/...")
    parser.add_argument("--points-per-object", type=int, default=32, help="points an object is followed by")
    parser.add_argument("--forget", type=float, default=6.0,
                        help="seconds a track may be hidden, and its object unseen, before it ends")
    parser.add_argument("--max-tracks", type=int, default=1024,
                        help="tracks at most (the tracker's slots; 1024 cost a fifth more time than 64)")
    parser.add_argument("--margin", type=int, default=4,
                        help="pixels new points keep from their mask's edge, and a track may be outside its mask")
    parser.add_argument("--checkpoint", default="256",
                        help="TAPNext++'s: 256 or 512 (the input resolution it was fine-tuned at; in weights/ from "
                             "tools/setup.sh; README, \"Speed\") or a .pt file of a 256 one")
    parser.add_argument("--compile", default=None, metavar="MODE",
                        help="torch.compile the network in this mode (e.g. default: about twice as fast; compiles "
                             "at the first frames, about a minute)")
    parser.add_argument("--history", type=int, default=60,
                        help="frames kept for the segmenter's results and queries to refer to (by stamp)")
    parser.add_argument("--device", default="cuda")
    add_arguments(parser)  # --connect, --listen, --config; default: the PC's Zenoh router
    # --max-hz: every frame by default (a tracker follows best frame by frame)
    # --local: auto (the CUDA ring on this GPU, else shared memory), shm or cuda
    node.add_arguments(parser, local=("auto", "shm", "cuda"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    args = parser.parse_args(argv)
    mps.check()  # with CUDA_MPS_PIPE_DIRECTORY (MPS meant) its daemon has to run: else stop here
    # the options as parameters (the terminal interface shows them): read-only, but LIVE
    session = open_from_args(args)
    params = node.Parameters(session, args.node, args.camera.split(":", 1)[0])
    params.configure(args)
    params.declare_options(parser, args, live=LIVE, leave_out=("camera", "node"))
    params.check_overrides()
    if args.points_per_object < 1 or args.max_tracks < 1 or args.forget <= 0 or args.history < 2 or args.margin < 0 \
            or (args.max_offset_ms is not None and args.max_offset_ms < 0):
        parser.error("--points-per-object, --max-tracks: at least 1; --forget: positive; --history: at least 2; "
                     "--margin, --max-offset-ms: not negative")
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

    import torch  # noqa: PLC0415 - with the model, after the arguments are checked

    from .lifting import Lifter  # noqa: PLC0415
    from .model import Catchup, SlotTracker, load, to_pixels  # noqa: PLC0415
    from .seeding import grow, sample  # noqa: PLC0415
    from .tracks import Tracks  # noqa: PLC0415

    device = torch.device(args.device)
    net, resolution = load(args.checkpoint, device)
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
    max_offset_ns = None if args.max_offset_ms is None else round(args.max_offset_ms * 1e6)
    depth = (PartnerHistory(subscriber("depth", True), args.history, max_offset_ns, wake.set)
             if "depth" in keys else None)
    instances = reader("instances", False) if "instances" in keys else None
    detections = None
    if "detections" in keys:
        detections_history = StampedHistory(args.history, wake.set)
        detections = db.DetectionSubscriber(session, keys["detections"],
                                            lambda m: detections_history.put(m.header.stamp.to_ns(), m))
    queryable = session.declare_queryable(f"{camera}/{args.node}/{QUERIES}")  # answered between frames
    tracks_pub = None  # with the first frame: its camera frame
    report = node.Reporter(session, camera, args.node, "xblick-tapnext", max_hz=args.max_hz)
    report.watch(color)
    counts = report.counts
    source = "sam3" if instances is not None else "none"
    print(f"xblick-tapnext: {' + '.join(keys.values())} -> {camera}/{args.node}/tracks (TAPNext++ "
          f"{args.checkpoint}, input {resolution} x {resolution}, {args.max_tracks} tracks at most, "
          f"{args.points_per_object} per object)", flush=True)

    state = {}  # the camera's model and everything made for it (reset when it changes)

    def reset(info, frame_id):
        state.clear()
        state.update(key=(info.width, info.height, frame_id), tracker=SlotTracker(net, resolution, args.max_tracks,
                                                                                  device),
                     tracks=Tracks(args.max_tracks, args.points_per_object, forget_ns), lifter=Lifter(info),
                     prepared=OrderedDict(), positions=StampedHistory(args.history), seen=-1)
        state["catchup"] = Catchup(state["tracker"])

    seeds = []  # (stamp, object id, (k, 2) pixels of that frame, query or None)
    waiting_queries = []  # (query, its seed) answered once the seed is in the tracker
    newest_frame_at = time.monotonic()
    done = -1  # the stamp of the newest frame looked at

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

    def answer(query):
        nonlocal source
        key = f"{camera}/{args.node}/{QUERIES}"
        try:
            payload = query.payload.to_string() if query.payload is not None else ""
            request = parse(json.loads(payload)) if payload.strip() else {}
            if "source" in request:
                if request["source"] == "sam3" and instances is None:
                    raise ValueError("the node has no segmenter (--no-segmenter)")
                source = request["source"]
            elif "clear" in request:
                if "tracks" in state:
                    end(np.nonzero(state["tracks"].used)[0])
                seeds.clear()
            elif "points" in request:
                if "tracks" not in state:
                    raise ValueError("no frame tracked yet")
                stamp = request["stamp"] if request["stamp"] is not None else done
                if stamp not in state["prepared"]:
                    raise ValueError(f"no tracked frame of stamp {stamp} (the node keeps its last {args.history})")
                seed = (stamp, 0, torch.tensor(request["points"], device=device), query)
                seeds.append(seed)
                waiting_queries.append((query, seed, time.monotonic()))
                return  # answered with the track ids, after the next frame
            query.reply(key, json.dumps(describe()))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            query.reply_err(f"{type(error).__name__}: {error}")

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
            if seed[0] in prepared:
                usable.append(seed)
            else:
                counts["seeds_too_old"] += 1
                if seed[3] is not None:
                    replies.append((seed[3], [], False))
        seeds.clear()
        if not usable:
            return replies
        first = min(seed[0] for seed in usable)
        stamps = [s for s in prepared if s >= first]  # in the order they were tracked: by stamp
        index = {s: i for i, s in enumerate(stamps)}
        xy = torch.cat([seed[2] for seed in usable])
        starts = torch.cat([torch.full((len(seed[2]),), index[seed[0]], device=device) for seed in usable])
        began = time.monotonic()
        moved, visible = state["catchup"].run([prepared[s] for s in stamps] + [f], xy, height, width, starts)
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
                replies.append((query, [next(it) if k else None for k in keep], True))
        counts["catchup_frames"] += len(stamps)
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
        for query, ids, ok in replies:
            waiting_queries[:] = [w for w in waiting_queries if w[0] is not query]
            if ok:
                query.reply(f"{camera}/{args.node}/{QUERIES}", json.dumps({"track_ids": ids, "stamp": stamp}))
            else:
                query.reply_err(f"the frame {stamp} is no longer kept")

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
        cols = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, ids.shape[1] - 1)
        rows = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, ids.shape[0] - 1)
        orphans = tracks.orphans(present, visible, track_id)
        if len(orphans):  # an object under a new id keeps its tracks (tracks.py)
            under = ids[torch.as_tensor(rows[orphans], device=device),
                        torch.as_tensor(cols[orphans], device=device)].cpu().numpy()
            for o in present:
                tracks.adopt(orphans[under == o], o)
            counts["adopted"] += int(np.isin(under, present).sum())
        inside = np.zeros(len(uv), bool)
        masks = {}
        for o in present:
            masks[o] = ids == o
            mine = np.nonzero(tracks.object_id == o)[0]
            if len(mine):
                grown = grow(masks[o], args.margin)
                inside[mine] = grown[torch.as_tensor(rows[mine], device=device),
                                     torch.as_tensor(cols[mine], device=device)].cpu().numpy()
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

    def apply_parameters():
        """The live parameters' values now, in this thread: the seeding's margin, and the tracks' points per object
        and forgetting (from the next object seeded and the next frame on)."""
        args.margin = params["margin"]
        if state.get("tracks") is not None:
            state["tracks"].per_object = params["points_per_object"]
            state["tracks"].forget_ns = round(params["forget"] * 1e9)
        args.points_per_object, args.forget = params["points_per_object"], params["forget"]

    try:
        while True:
            wake.wait(0.05)
            wake.clear()
            apply_parameters()
            for history in (color, depth, instances):
                if history is not None:
                    history.check()
            if detections is not None:
                detections.check()
            while (query := queryable.try_recv()) is not None:
                answer(query)
            stamp, skipped = frame_to_track()
            if stamp is not None:
                newest_frame_at = time.monotonic()
                track(stamp, skipped)
            check_segmenter()
            if waiting_queries and time.monotonic() - newest_frame_at > QUERY_WAIT_S:
                for query, seed, _ in waiting_queries:
                    query.reply_err(f"no colour frame for {QUERY_WAIT_S} s")
                    if seed in seeds:
                        seeds.remove(seed)
                waiting_queries.clear()
            if report.due():
                d, seconds = report.take()
                n, ms = max(d["frames"], 1), d["busy"] / max(d["frames"], 1) * 1e3
                problems = ([f"{d['no_info']} frames without camera_info"] if d["no_info"] else []) + \
                    ([f"{d['no_depth_info']} frames without the depth's camera_info"] if d["no_depth_info"] else []) + \
                    ([f"{d['skipped']} frames left out (busy)"] if d["skipped"] else []) + \
                    ([f"{d['no_room']} points without a free track"] if d["no_room"] else []) + \
                    ([f"{d['seeds_too_old']} seeds of frames no longer kept"] if d["seeds_too_old"] else [])
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
                                "catch-up frames": d["catchup_frames"],
                                "catch-up points": d["catchup_points"], "catch-up ms": d["catchup_busy"] * 1e3,
                                "source": source},
                               log_extra=f", {ms:.0f} ms a frame (catch-up {d['catchup_busy'] * 1e3:.0f} ms of it, "
                                         f"{d['catchup_points']} points over {d['catchup_frames']} frames), "
                                         f"{d['tracks'] / n:.0f} tracks, {d['visible'] / n:.0f} visible, "
                                         f"{d['adopted']} adopted, {d['made_room']} hidden ended for new points")
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
        params.close()
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
