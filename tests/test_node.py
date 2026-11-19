"""The node as its own process on a published colour stream (needs a CUDA device and the 256 checkpoint in the node's
cache directory): its queryable answers at once while the tracker works on a frame (tests/slow_step.py), points
within QUERY_WAIT_S."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch
import xblick as db
from xblick.session import open_session, replies, timed_out

from xblick_tapnext.model import CHECKPOINTS
from xblick_tapnext.node import QUERY_WAIT_S
from xblick_tapnext.weights import STORE

CAM = "test/tapnext_cam"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not (STORE.root / CHECKPOINTS["256"][0]).is_file(),
                                reason="needs a CUDA device and the checkpoint (tools/setup.sh)")


def test_queries_are_answered_at_once_while_the_model_works():
    """During a 4 s step of the tracker the state is answered within 0.5 s, and a points query within QUERY_WAIT_S
    with an error (no tracks made of it); never a "Timeout" reply. After the step a points query gets its track ids."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    info = db.CameraInfo.pinhole(320, 240, 300.0, 300.0, 159.5, 119.5, frame_id="cam")
    color = db.CameraPublisher(session, f"{CAM}/color", encoding="rgb8", frame_id="cam", camera_info=info,
                               source="test")
    stop = threading.Event()

    def publish():
        rng, stamp = np.random.default_rng(0), 1_000_000_000
        texture = rng.integers(0, 255, (240, 400, 3), dtype=np.uint8)
        while not stop.is_set():
            shift = (stamp // 100_000_000) % 80
            color.publish(np.ascontiguousarray(texture[:, shift:shift + 320]), stamp_ns=stamp)
            stamp += 100_000_000
            time.sleep(0.1)

    command = [sys.executable, os.path.join(os.path.dirname(__file__), "slow_step.py"), "4.0", "--camera", CAM,
               "--no-depth", "--no-segmenter", "--compile", "none", "--connect", f"127.0.0.1:{port}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    threading.Thread(target=publish, daemon=True).start()

    def ask(payload=None):
        began = time.monotonic()
        got = replies(session, f"{CAM}/tapnext/queries", 5.0, **({"payload": payload} if payload else {}))
        took = time.monotonic() - began
        assert not any(r.err is not None and timed_out(r) for r in got), "".join(output)
        assert len(got) == 1, "".join(output)
        reply = got[0]
        return (json.loads(reply.ok.payload.to_string()) if reply.ok is not None else reply.err.payload.to_string(),
                reply.ok is not None, took)

    try:
        deadline = time.monotonic() + 180.0
        while not any("slow step" in line for line in output):
            assert process.poll() is None and time.monotonic() < deadline, "".join(output)
            time.sleep(0.05)
        began = time.monotonic()
        state, ok, took = ask()
        assert ok and took < 0.5 and state["tracks"] == 0, (state, took)
        error, ok, took = ask(json.dumps({"points": [[100, 100], [200, 150]]}))
        assert not ok and "no tracks were made" in error, error
        assert QUERY_WAIT_S - 0.1 < took < QUERY_WAIT_S + 0.5, took
        assert time.monotonic() - began < 4.0, "the queries outlasted the step: no check"
        time.sleep(3.0)  # the step is over
        got, ok, took = ask(json.dumps({"points": [[100, 100], [200, 150]]}))
        assert ok and len(got["track_ids"]) == 2 and took < 1.0, (got, took, "".join(output))
        state, ok, _ = ask()
        assert ok and state["tracks"] == sum(i is not None for i in got["track_ids"]), state  # none of the first
    finally:
        stop.set()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        color.close()
        session.close()
    assert process.returncode == 0, "".join(output)


def test_the_node_compiles_by_default_and_a_failed_compilation_ends_it():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    try:
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "failing_compile.py"), "--camera", CAM,
                   "--no-depth", "--no-segmenter", "--connect", f"127.0.0.1:{port}"]
        done = subprocess.run(command, capture_output=True, text=True, timeout=180)
    finally:
        session.close()
    assert done.returncode == 1, done.stdout + done.stderr
    assert "compiling the network (--compile default; one to two minutes" in done.stdout, done.stdout
    assert done.stderr.strip().splitlines()[-1] == \
        "xblick-tapnext: --compile default failed: RuntimeError: no compiler here", done.stderr
