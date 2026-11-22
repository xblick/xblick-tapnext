"""The slot tracker (model.py) on a texture moving 2 pixels a frame (needs a CUDA device and the 256 checkpoint in
the node's cache directory): points given at the start follow it; points given later into the running state go as
they do in a state of their own; a freed slot takes a new point; the catch-up follows points from older frames."""

import pytest
import torch

from xblick_tapnext.model import CHECKPOINTS, Catchup, SlotTracker, SlotTrackers, load, to_model, to_pixels, warm_up
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


def test_decoys_fill_free_slots_and_leave_the_tracks_as_they_are(net):
    network, resolution = net
    video = frames(25)
    tracker = SlotTracker(network, resolution, 256, torch.device("cuda"), decoys=True)
    assert bool(tracker.decoy[:256].all()) and not bool(tracker.decoy[256]) and not bool(tracker.used.any())
    start = torch.tensor([[100.0, 60.0], [150.0, 120.0], [200.0, 180.0]], device="cuda")
    for i, rgb in enumerate(video):
        f = tracker.prepare(rgb)
        if i == 2:  # into decoy slots: they become tracks
            tracker.add(torch.tensor([5, 6, 7], device="cuda"), start, H, W)
            assert not bool(tracker.decoy[5:8].any()) and bool(tracker.used[5:8].all())
        if i == 12:
            tracker.free(torch.tensor([7], device="cuda"))
            assert bool(tracker.decoy[7]) and not bool(tracker.used[7])
        yx, visible = tracker.step(f)
    xy = to_pixels(yx, H, W)
    expected = start[:2] + torch.tensor([STEP * 22, 0.0], device="cuda")
    assert visible[5:7].all() and (xy[5:7] - expected).norm(dim=1).max() < 1.5


def test_trackers_stepped_together_go_as_each_alone(net):
    """Three trackers, each on a texture of its own moving at its own speed, one starting later, with late points, a
    freed and refilled slot and a step alone in between: SlotTrackers' one pass per frame gives each tracker's slots
    the places and visibility of the same tracker stepped alone (float16 rounding apart)."""
    network, resolution = net
    device = torch.device("cuda")
    speeds = (2.0, -1.0, 3.0)
    videos = []
    for k, speed in enumerate(speeds):
        g = torch.Generator().manual_seed(k + 1)
        n = 30
        big = torch.rand(1, 3, H // 4, (W + int(abs(speed) * n) + 8) // 4, generator=g)
        big = torch.nn.functional.interpolate(big, scale_factor=4, mode="bicubic").clamp(0, 1)[0].permute(1, 2, 0)
        big = (big * 255).to(torch.uint8)
        width = big.shape[1]
        cut = [width - W - int(speed * i) if speed > 0 else int(-speed * i) for i in range(n)]
        videos.append([big[:, c: c + W].contiguous().cuda() for c in cut])
    start = torch.tensor([[100.0, 60.0], [150.0, 120.0], [200.0, 180.0]], device="cuda")
    late = torch.tensor([[90.0, 100.0], [160.0, 50.0]], device="cuda")

    def run(together: bool):
        trackers = [SlotTracker(network, resolution, 16, device, decoys=True) for _ in speeds]
        batch = SlotTrackers()
        seen = []
        for i in range(30):
            if i == 0:
                trackers[0].add(torch.arange(3, device="cuda"), start, H, W)
                trackers[1].add(torch.arange(3, device="cuda"), start, H, W)
            if i == 5:
                trackers[2].add(torch.arange(3, device="cuda"), start, H, W)
            if i == 8:
                trackers[1].add(torch.tensor([5, 6], device="cuda"), late, H, W)
            if i == 12:
                trackers[0].free(torch.tensor([0], device="cuda"))
                trackers[0].add(torch.tensor([0], device="cuda"), late[:1], H, W)
            active = [k for k in range(3) if i >= (5 if k == 2 else 0)]
            frames = [trackers[k].prepare(videos[k][i]) for k in active]
            if together and i != 20:  # frame 20: one tracker alone, the others together
                out = batch.step([trackers[k] for k in active], frames)
            elif together:
                out = [trackers[0].step(frames[0])] + batch.step([trackers[k] for k in active[1:]], frames[1:])
            else:
                out = [trackers[k].step(f) for k, f in zip(active, frames, strict=True)]
            seen.append({k: (to_pixels(yx, H, W), vis) for k, (yx, vis) in zip(active, out, strict=True)})
        return seen

    alone, together = run(False), run(True)
    worst = 0.0
    for a, b in zip(alone, together, strict=True):
        for k in a:
            used = torch.tensor([0, 1, 2, 5, 6] if k == 1 else [0, 1, 2], device="cuda")
            assert torch.equal(a[k][1][used], b[k][1][used])
            worst = max(worst, float((a[k][0][used] - b[k][0][used]).norm(dim=1).max()))
    print(f"largest difference {worst:.4f} px")
    assert worst < 0.05
    # and they track: the first tracker's late point (slot 0 since frame 12) moved with its texture
    xy, visible = together[-1][0]
    assert visible[0] and (xy[0] - (late[0] + torch.tensor([2.0 * 17, 0.0], device="cuda"))).norm() < 1.5


def test_compiled_trackers_together_after_the_warm_up_compile_nothing_more(net):
    """The compiled network, warmed up for batches of up to three trackers: steps of two and three trackers together
    (first frames and later ones) compile nothing new, and go as each tracker alone."""
    import torch._dynamo  # noqa: PLC0415

    network, resolution = net
    device = torch.device("cuda")
    torch._dynamo.reset()
    compiled = torch.compile(network)
    warm_up(compiled, resolution, 16, device, True, batch=3)
    compiles = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    video = frames(12)
    start = torch.tensor([[100.0, 60.0], [150.0, 120.0], [200.0, 180.0]], device="cuda")
    trackers = [SlotTracker(compiled, resolution, 16, device, decoys=True) for _ in range(3)]
    alone = [SlotTracker(compiled, resolution, 16, device, decoys=True) for _ in range(3)]
    for t in trackers + alone:
        t.add(torch.arange(3, device="cuda"), start, H, W)
    batch = SlotTrackers()
    worst = 0.0
    for i, rgb in enumerate(video):
        n = 2 if i < 6 else 3  # the third starts at frame 6
        f = trackers[0].prepare(rgb)
        together = batch.step(trackers[:n], [f] * n)
        for k in range(n):
            yx, _ = alone[k].step(f)
            worst = max(worst, float((to_pixels(yx[:3], H, W) - to_pixels(together[k][0][:3], H, W)).norm(dim=1).max()))
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == compiles
    assert worst < 0.05
