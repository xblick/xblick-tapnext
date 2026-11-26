"""TensorRT engines for slot trackers stepped together (tensorrt.py; needs a CUDA device, the 256 checkpoint and the
optional dependencies trt): the export copy computes what the network computes, trackers stepped in the engines
follow a moving texture as in PyTorch, and engines loaded from their directory give what the built ones gave."""

import pytest
import torch
from test_model import STEP, WEIGHTS, H, W, frames

from xblick_tapnext.model import CHECKPOINTS, SlotTracker, SlotTrackers, load, to_pixels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not (WEIGHTS / CHECKPOINTS["256"][0]).is_file(),
                                reason="needs a CUDA device and the checkpoint (tools/setup.sh)")
torch_tensorrt = pytest.importorskip("torch_tensorrt", reason="needs the optional dependencies trt (setup.sh --trt)")

SLOTS = 16


@pytest.fixture(scope="module")
def net():
    return load("256", torch.device("cuda"), WEIGHTS)


def test_the_export_copy_computes_the_networks_numbers(net):
    """One tracker's later frame through for_export's copy (in PyTorch) gives the network's tracks and visibility
    exactly."""
    from xblick_tapnext.tensorrt import _Flat, for_export

    network, resolution = net
    video = frames(3)
    tracker = SlotTracker(network, resolution, SLOTS, torch.device("cuda"), decoys=True)
    tracker.add(torch.arange(3, device="cuda"), torch.tensor([[100.0, 60.0], [150.0, 120.0], [200.0, 180.0]],
                                                             device="cuda"), H, W)
    tracker.step(tracker.prepare(video[0]))
    queries = tracker._queries().clone()
    queries[..., 0] -= tracker.state.step  # as SlotTrackers passes them: the state at step 0
    flat = [s.clone() for c in tracker.state.hidden_state for s in (c.rg_lru_state, c.conv1d_state)]
    frame = tracker.prepare(video[1])
    with torch.no_grad():
        ours = _Flat(network)(frame, queries, *flat)
        theirs = _Flat(for_export(network))(frame, queries, *flat)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1] > 0, theirs[1] > 0)


def test_trackers_in_the_engines_go_as_in_pytorch_and_load_as_built(net, tmp_path):
    from xblick_tapnext.tensorrt import TensorRTNetwork

    network, resolution = net
    device = torch.device("cuda")
    engines = TensorRTNetwork(network, "256", resolution, device, tmp_path)
    engines.build(SLOTS, 3)
    assert sorted(p.name.split("_b")[1].split("_")[0] for p in tmp_path.glob("*.pt2")) == ["1", "2-3"]
    video = frames(12)
    start = torch.tensor([[100.0, 60.0], [150.0, 120.0], [200.0, 180.0]], device="cuda")

    def run(backend, n):
        trackers = [SlotTracker(backend, resolution, SLOTS, device, decoys=True, check_slots=False) for _ in range(n)]
        for t in trackers:
            t.add(torch.arange(3, device="cuda"), start, H, W)
        batch = SlotTrackers()
        for rgb in video:
            f = trackers[0].prepare(rgb)
            out = batch.step(trackers, [f] * n)
        return [to_pixels(yx[:3], H, W) for yx, _ in out]

    expected = start + torch.tensor([STEP * 11, 0.0], device="cuda")
    for n in (1, 3):
        reference, engine = run(network, n), run(engines, n)
        for a, b in zip(reference, engine, strict=True):
            assert (b - expected).norm(dim=1).max() < 1.5
            assert (a - b).norm(dim=1).max() < 0.5
    loaded = TensorRTNetwork(network, "256", resolution, device, tmp_path)
    loaded.build(SLOTS, 3)
    for a, b in zip(run(engines, 3), run(loaded, 3), strict=True):
        assert torch.equal(a, b)
    with pytest.raises(RuntimeError, match="no TensorRT engine"):
        run(engines, 4)
