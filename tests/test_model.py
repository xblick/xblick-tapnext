"""The slot tracker (model.py) on a texture moving 2 pixels a frame (needs a CUDA device and the 256 checkpoint in
the node's cache directory): points given at the start follow it; points given later into the running state go as
they do in a state of their own; a freed slot takes a new point; the catch-up follows points from older frames."""

import pytest
import torch

from xblick_tapnext.model import CHECKPOINTS, Catchup, SlotTracker, load, to_model, to_pixels
from xblick_tapnext.weights import STORE

WEIGHTS = STORE.root  # the node's cache directory (XBLICK_HOME moves it)

H, W, STEP = 240, 320, 2.0  # frame size, pixels the texture moves right per frame

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not (WEIGHTS / CHECKPOINTS["256"][0]).is_file(),
                                reason="needs a CUDA device and the checkpoint (tools/setup.sh)")


@pytest.fixture(scope="module")
def net():
    return load("256", torch.device("cuda"), WEIGHTS)


def frames(n):
    """A smooth random texture, shifted right by STEP a frame (a crop of a larger one)."""
    g = torch.Generator().manual_seed(0)
    big = torch.rand(1, 3, H // 4, (W + int(STEP * n) + 8) // 4, generator=g)
    big = torch.nn.functional.interpolate(big, scale_factor=4, mode="bicubic").clamp(0, 1)[0].permute(1, 2, 0)
    big = (big * 255).to(torch.uint8)
    width = big.shape[1]
    return [big[:, width - W - int(STEP * i): width - int(STEP * i)].contiguous().cuda() for i in range(n)]


def test_coordinates_go_to_the_model_and_back():
    xy = torch.tensor([[0.0, 0.0], [319.0, 239.0], [10.25, 3.5]])
    assert torch.allclose(to_pixels(to_model(xy, H, W), H, W), xy, atol=1e-4)
    assert torch.allclose(to_model(xy, H, W)[0], torch.tensor([0.5 * 256 / H, 0.5 * 256 / W]))


def test_points_follow_late_ones_as_in_their_own_state_and_slots_are_reused(net):
    network, resolution = net
    video = frames(40)
    tracker = SlotTracker(network, resolution, 16, torch.device("cuda"))
    start = torch.tensor([[100.0, 60.0], [150.0, 120.0], [200.0, 180.0], [120.0, 200.0]], device="cuda")
    tracker.add(torch.arange(4, device="cuda"), start, H, W)
    late = torch.tensor([[90.0, 100.0], [160.0, 50.0]], device="cuda")
    prepared, seen = [], {}
    for i, rgb in enumerate(video):
        f = tracker.prepare(rgb)
        if i == 10:
            tracker.add(torch.tensor([4, 5], device="cuda"), late, H, W)
        if i == 20:
            tracker.free(torch.tensor([0], device="cuda"))
            tracker.add(torch.tensor([0], device="cuda"), start[:1], H, W)
        if i >= 10:
            prepared.append(f)
        yx, visible = tracker.step(f)
        seen[i] = (to_pixels(yx, H, W), visible)
    xy, visible = seen[39]
    expected = start + torch.tensor([STEP * 39, 0.0], device="cuda")
    assert visible[1:4].all() and (xy[1:4] - expected[1:]).norm(dim=1).max() < 1.5
    assert (xy[4:6] - (late + torch.tensor([STEP * 29, 0.0], device="cuda"))).norm(dim=1).max() < 1.5
    reused = start[0] + torch.tensor([STEP * 19, 0.0], device="cuda")
    assert visible[0] and (xy[0] - reused).norm() < 1.5
    own, own_visible = Catchup(tracker).run(prepared, late, H, W)
    assert own_visible.all() and (own - xy[4:6]).norm(dim=1).max() < 1.0
    # points of two frames in one catch-up state: the second group queried 5 frames later, where it is then
    later = late + torch.tensor([STEP * 5, 0.0], device="cuda")
    both, both_visible = Catchup(tracker).run(prepared, torch.cat([late, later]), H, W,
                                              torch.tensor([0, 0, 5, 5], device="cuda"))
    assert both_visible.all() and (both - torch.cat([xy[4:6], xy[4:6]])).norm(dim=1).max() < 1.5


def test_a_catchup_over_every_second_frame_ends_where_the_full_one_does(net):
    network, resolution = net
    video = frames(21)
    tracker = SlotTracker(network, resolution, 16, torch.device("cuda"))
    prepared = [tracker.prepare(rgb) for rgb in video]
    late = torch.tensor([[90.0, 100.0], [160.0, 50.0], [200.0, 180.0]], device="cuda")
    full, full_visible = Catchup(tracker).run(prepared, late, H, W)
    half, half_visible = Catchup(tracker).run(prepared[::2], late, H, W)  # 0, 2, ..., 20: steps of 2 * STEP
    expected = late + torch.tensor([STEP * 20, 0.0], device="cuda")
    assert full_visible.all() and half_visible.all()
    assert (full - expected).norm(dim=1).max() < 1.5 and (half - expected).norm(dim=1).max() < 1.5
