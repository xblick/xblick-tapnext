"""xblick-tapnext-queries: asks or tells a running xblick-tapnext over its queryable
``<camera>/<node>/queries``.

    xblick-tapnext-queries --camera cam/azurekinect                        # the node's state
    xblick-tapnext-queries --camera cam/azurekinect --json '{"points": [[320, 240]]}'
                               [--node tapnext] [--timeout 10] [--connect ...] [--listen ...] [--config FILE]

A query without payload is answered with the node's state; one with a JSON payload is one of::

    {"points": [[u, v], ...], "stamp": 1730000000123456789}   # follow these pixels (stamp: of their colour frame,
                                                              # one of the node's last --history; default its newest)
    {"source": "sam3"}            # seed the objects of the camera's segmenter, or "none": no new seeds
    {"clear": true}               # end all tracks

``points`` is answered once the points are in the tracker (after the next frame): ``{"track_ids": [...], "stamp":
<the frame they were put in>}``, a track id per point, null for a point the tracker did not see in that frame. Its
tracks have object_id 0 and end as all do (README, "Tracks"). The others are answered at once, with the state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

QUERIES = "queries"  # <camera>/<node>/queries
ASK_AGAIN_S = 0.2  # a query that reached no queryable: asked again after this
SOURCES = ("sam3", "none")


def parse(request) -> dict:
    """A request as documented above, checked; ValueError otherwise."""
    if not isinstance(request, dict) or len(request) == 0:
        raise ValueError('a request is {"points": [[u, v], ...], "stamp": ns}, {"source": ...} or {"clear": true}')
    if "points" in request:
        if set(request) - {"points", "stamp"}:
            raise ValueError("points: only with a stamp")
        points = request["points"]
        if not isinstance(points, list) or not points or not all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(v, int | float) and not isinstance(v, bool)
                                                            for v in p) for p in points):
            raise ValueError("points: a non-empty list of [u, v] (pixels of the colour image)")
        stamp = request.get("stamp")
        if stamp is not None and (not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < 0):
            raise ValueError(f"stamp {stamp!r}: nanoseconds, an integer")
        return {"points": [[float(u), float(v)] for u, v in points], "stamp": stamp}
    if set(request) == {"source"}:
        if request["source"] not in SOURCES:
            raise ValueError(f"source {request['source']!r}: one of {', '.join(SOURCES)}")
        return dict(request)
    if request == {"clear": True}:
        return dict(request)
    raise ValueError(f"{request!r}: points (with a stamp), source or clear, one of them")


def main(argv=None) -> int:
    from xblick import topics  # noqa: PLC0415
    from xblick.session import add_arguments, open_from_args  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="xblick-tapnext-queries", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="the camera's base, e.g. cam/azurekinect")
    parser.add_argument("--node", default="tapnext", help="the node's part of the topic: <camera>/<node>/queries")
    parser.add_argument("--json", help="the request (above); without: the node's state")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="seconds to wait for the answer (the node answers between frames)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    key = f"{topics.check_name(args.camera)}/{args.node}/{QUERIES}"
    payload = None
    if args.json:
        parse(json.loads(args.json))  # fails here, not in the node
        payload = args.json
    session = open_from_args(args)
    try:
        # right after the session opens a query reaches no queryable yet: asked again until an answer or the timeout
        replies, end = [], time.monotonic() + args.timeout
        while not replies and (remaining := end - time.monotonic()) > 0:
            replies = list(session.get(key, payload=payload, timeout=remaining))
            if not replies:
                time.sleep(min(ASK_AGAIN_S, max(0.0, end - time.monotonic())))
    finally:
        session.close()
    if not replies:
        print(f"xblick-tapnext-queries: no answer on {key} within {args.timeout} s", file=sys.stderr)
        return 1
    failed = False
    for reply in replies:
        if reply.ok is not None:
            print(reply.ok.payload.to_string())
        else:
            print(f"xblick-tapnext-queries: {reply.err.payload.to_string()}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
