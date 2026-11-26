"""The slot tracker's bookkeeping without the GPU (CPU, a stand-in network): since 0.2.9 its add, free and step wait
for nothing; they must give 0.2.8's queries and states exactly. Also the coordinates and the read-back in one copy."""

import numpy as np
import pytest
import torch

from xblick_tapnext import model
from xblick_tapnext.model import MODEL, SlotTracker, decoy_points, read_tracks, to_host, to_model, to_pixels


class _Cache:
    def __init__(self, rg_lru_state, conv1d_state):
        self.rg_lru_state, self.conv1d_state = rg_lru_state, conv1d_state


class _State:
    def __init__(self, hidden_state, step):
        self.hidden_state, self.step = hidden_state, step


class RecordingNet:
    """A stand-in for TAPNext++: records each step's queries; its state's rows (4 patches, then the queries) carry the
    sum of the queries seen, so that a reset slot shows in it; tracks are the queries' places."""

    patches = 4

    def __init__(self):
        self.seen = []

    def __call__(self, video, query_points, state):
        self.seen.append(query_points.clone())
        q = query_points[0]
        rows = torch.cat([torch.zeros(self.patches, 3), q])
        old = None if state is None else state.hidden_state
        hidden = [_Cache(rows.sum(1, keepdim=True).repeat(1, 2) + (0 if old is None else old[0].rg_lru_state),
                         rows[:, None, :2] + (0 if old is None else old[0].conv1d_state))]
        visible = (q[:, 1:2] > 100).float()[None, None]
        return q[None, None, :, 1:], None, visible, _State(hidden, 0 if state is None else state.step + 1)


def _old_to_model(xy, height, width):
    """0.2.8's to_model: the scale a float32 tensor."""
    scale = torch.tensor([MODEL / width, MODEL / height], device=xy.device)
    return ((xy + 0.5) * scale).flip(-1)


def _old_to_pixels(yx, height, width):
    scale = torch.tensor([width / MODEL, height / MODEL], device=yx.device)
    return yx.flip(-1) * scale - 0.5


class Old(SlotTracker):
    """0.2.8's add, free, decoy placing and queries (boolean indexing, numbers assigned by index, the free check)."""

    def _place(self, slots):
        self._reset(slots)
        self.queries[0, slots, 0] = self.frames
        self.queries[0, slots, 1:] = decoy_points(len(slots), self._placed).to(self.device)
        self._placed += len(slots)
        self.decoy[slots] = True

    def add(self, slots, xy, height, width):
        if bool(((slots >= self.slots) | self.used[slots.clamp(max=self.slots)]).any()):
            raise ValueError(f"slots {slots.tolist()}: not all free")
        if self.decoys:
            which = torch.zeros_like(self.decoy)
            which[slots] = True
            self._reset_where(which & self.decoy)
        self.decoy[slots] = False
        self.queries[0, slots, 0] = self.frames
        self.queries[0, slots, 1:] = _old_to_model(xy.float(), height, width)
        self.used[slots] = True

    def free(self, slots):
        self.used[slots] = False
        if self.decoys:
            self._place(slots)
        else:
            self._reset(slots)

    def _queries(self):
        future = ~self.used & ~self.decoy
        self.queries[0, future, 0] = self.frames + 1
        self.queries[0, future, 1:] = MODEL / 2
        return self.queries


@pytest.mark.parametrize("decoys", [True, False])
@pytest.mark.parametrize("check_slots", [True, False])
def test_steps_give_0_2_8s_queries_and_states(decoys, check_slots):
    nets = [RecordingNet(), RecordingNet()]
    trackers = [Old(nets[0], 256, 8, torch.device("cpu"), decoys=decoys),
                SlotTracker(nets[1], 256, 8, torch.device("cpu"), decoys=decoys, check_slots=check_slots)]
    frame = torch.zeros(1, 1, 256, 256, 3)
    script = [("step",), ("add", [0, 1], [[10.0, 20.0], [300.5, 40.25]]), ("step",), ("step",), ("free", [0]),
              ("add", [2, 3], [[5.0, 5.0], [600.0, 300.0]]), ("step",), ("free", [1, 2]), ("add", [1], [[7.0, 8.0]]),
              ("step",)] + [("step",)] * 30  # past a refresh of the decoys
    for op in script:
        for t in trackers:
            if op[0] == "step":
                t.step(frame)
            elif op[0] == "add":
                t.add(torch.tensor(op[1]), torch.tensor(op[2]), 480, 640)
            else:
                t.free(torch.tensor(op[1]))
    assert len(nets[0].seen) == len(nets[1].seen) == 35
    for a, b in zip(nets[0].seen, nets[1].seen, strict=True):
        assert torch.equal(a, b)
    for name in ("rg_lru_state", "conv1d_state"):
        assert torch.equal(getattr(trackers[0].state.hidden_state[0], name),
                           getattr(trackers[1].state.hidden_state[0], name))
    assert torch.equal(trackers[0].used, trackers[1].used) and torch.equal(trackers[0].decoy, trackers[1].decoy)


def test_a_used_slot_raises_unless_the_caller_checks():
    tracker = SlotTracker(RecordingNet(), 256, 8, torch.device("cpu"))
    tracker.add(torch.tensor([0]), torch.tensor([[1.0, 2.0]]), 480, 640)
    with pytest.raises(ValueError, match="not all free"):
        tracker.add(torch.tensor([0]), torch.tensor([[1.0, 2.0]]), 480, 640)
    with pytest.raises(ValueError, match="not all free"):
        tracker.add(torch.tensor([8]), torch.tensor([[1.0, 2.0]]), 480, 640)  # the never-used slot


def test_coordinates_as_numbers_give_0_2_8s_values():
    """to_model and to_pixels (their scales as numbers, no copy to the GPU) give the values of 0.2.8's float32 scale
    tensor, also for float64 and integer input and other shapes."""
    xy = torch.rand(500, 2) * torch.tensor([1279.0, 719.0])
    for h, w in ((720, 1280), (333.0, 333.0), (97.5, 97.5), (240, 320)):
        assert torch.equal(to_model(xy, h, w), _old_to_model(xy, h, w))
        assert torch.equal(to_pixels(xy, h, w), _old_to_pixels(xy, h, w))
        assert torch.equal(to_model(xy.double(), h, w), _old_to_model(xy.double(), h, w))
        assert torch.equal(to_model(xy.long(), h, w), _old_to_model(xy.long(), h, w))
        assert torch.equal(to_pixels(xy.reshape(50, 10, 2), h, w), _old_to_pixels(xy.reshape(50, 10, 2), h, w))
    assert to_model(xy.long(), 720, 1280).dtype == torch.float32


def test_to_host_and_read_tracks_in_one_copy():
    parts = [torch.arange(6.0).reshape(3, 2), torch.zeros(0, 2), torch.ones(2, 2, 2)]
    out = to_host(parts)
    assert [o.shape for o in out] == [(3, 2), (0, 2), (2, 2, 2)]
    np.testing.assert_array_equal(out[0], np.arange(6.0).reshape(3, 2))
    with pytest.raises(ValueError, match="one dtype"):
        to_host([torch.zeros(2), torch.zeros(2, dtype=torch.long)])
    assert to_host([]) == []
    yx = torch.tensor([[10.0, 20.0], [128.0, 64.0]])
    visible = torch.tensor([True, False])
    (xy_a, vis_a), (xy_b, vis_b) = read_tracks([(yx, visible), (yx, visible)], [(480, 640), (90.0, 90.0)])
    np.testing.assert_array_equal(xy_a, to_pixels(yx, 480, 640).numpy().astype(np.float64))
    np.testing.assert_array_equal(xy_b, to_pixels(yx, 90.0, 90.0).numpy().astype(np.float64))
    assert xy_a.dtype == np.float64 and vis_a.tolist() == [True, False] and vis_b.tolist() == [True, False]


def test_to_device_copies_the_values():
    assert torch.equal(model.to_device([1, 2, 3], torch.long, "cpu"), torch.tensor([1, 2, 3]))
    assert model.to_device(np.zeros((2, 3)), torch.float32, "cpu").dtype == torch.float32
