"""TAPNext++ (Google DeepMind, CVPR 2026; TAPNext, ICCV 2025) as an online tracker of a fixed number of point slots.

TAPNext is a ViT whose tokens (the frame's patches and one per query point) each carry a recurrent state through time
(an RG-LRU, then attention between the tokens of the frame): fed one frame at a time, it follows each query from the
frame it was given in. DeepMind's code (third_party/tapnet, tapnet/tapnext/tapnext_torch.py) fixes the queries when
a state starts: "new queries cannot be added to an existing state". This module keeps one state of ``slots`` query
tokens and makes room in it instead, with what the model was trained on: a query whose frame has not come yet is an
"unknown" token until then, its point token in its frame, a mask token after it. So

* a free slot is a query of the future: its token is the unknown token every frame (no position, no time, so all free
  slots get the same input and have the same state);
* ``add`` gives a free slot its point: on the next frame it is that frame's query, as if it had been known from the
  start and queried late;
* ``free`` makes a slot free again: its state becomes the state of the slot that is never used (the last), the state
  of a query that has not come yet.

Coordinates are pixels of the frame, x right and y down, pixel centres at integers (ROS's, CameraInfo's K's); the
model works in 256 x 256 (``MODEL``), pixel centres at .5, whatever the input resolution.

``Catchup`` follows new points from an older frame to the newest (the stored frames since), in a state of its own: a
segmenter's mask comes some frames after its frame, and a query has to be given in the frame it is on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .weights import CHECKPOINTS  # the checkpoints' one list (weights.py)

REPO = Path(__file__).resolve().parents[2]
TAPNET = REPO / "third_party" / "tapnet"
MODEL = 256  # TAPNext's coordinates: 256 x 256 whatever the input resolution


def _tapnext_module():
    if not (TAPNET / "tapnet" / "tapnext" / "tapnext_torch.py").exists():
        raise FileNotFoundError(f"{TAPNET}: DeepMind's tapnet missing (tools/setup.sh: git submodule update --init)")
    if str(TAPNET) not in sys.path:
        sys.path.insert(0, str(TAPNET))  # tapnet.tapnext.* only: the package's dependencies (JAX ...) not needed
    from tapnet.tapnext import tapnext_torch  # noqa: PLC0415

    return tapnext_torch


def load(checkpoint: str, device: torch.device, root: Path):
    """TAPNext++'s network (``CHECKPOINTS`` name, its file in ``root``: the store's, ``weights.STORE.root``; or a path
    to a .pt of a 256-resolution TAPNext) and its input resolution."""
    tapnext_torch = _tapnext_module()
    if checkpoint in CHECKPOINTS:
        name, resolution, *_ = CHECKPOINTS[checkpoint]
        path = Path(root) / name
        if not path.is_file():
            raise FileNotFoundError(f"{path}: missing (xblick-tapnext downloads it at its start, or with "
                                    "--download-only)")
    else:
        path, resolution = Path(checkpoint), MODEL
    net = tapnext_torch.TAPNext(image_size=(MODEL, MODEL))
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("tapnext."): v for k, v in state.get("state_dict", state).items()}
    net.load_state_dict(state)
    return net.to(device).eval(), resolution


class _Net:
    """The network with its input: frames resized to its resolution, queries in its coordinates."""

    def __init__(self, net, resolution: int, device: torch.device):
        self.net, self.resolution, self.device = net, resolution, device

    def prepare(self, rgb: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) uint8 on the GPU -> the network's frame, (1, 1, R, R, 3) in [-1, 1] (as DeepMind's
        preprocess_frame: bilinear, not antialiased)."""
        x = rgb.permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return (x / 127.5 - 1.0).permute(0, 2, 3, 1)[None]

    @torch.no_grad()
    def forward(self, frame, queries, state):
        """queries (1, Q, 3) [t, y, x] in model coordinates, t in frames since the state's start."""
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            tracks, _, visible_logits, state = self.net(video=frame, query_points=queries, state=state)
        return tracks[0, 0].float(), visible_logits[0, 0, :, 0].float() > 0, state


def to_model(xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(N, 2) pixels [x, y] (centres at integers) -> (N, 2) model [y, x] (centres at .5)."""
    scale = torch.tensor([MODEL / width, MODEL / height], device=xy.device)
    return ((xy + 0.5) * scale).flip(-1)


def to_pixels(yx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(N, 2) model [y, x] -> (N, 2) pixels [x, y]."""
    scale = torch.tensor([width / MODEL, height / MODEL], device=yx.device)
    return yx.flip(-1) * scale - 0.5


DECOY_REFRESH = 30  # frames between re-placings of the decoys the model does not see


def decoy_points(n: int, offset: int = 0) -> torch.Tensor:
    """``n`` points spread over the model's image (an R2 low-discrepancy sequence from ``offset``), model [y, x]."""
    i = torch.arange(offset, offset + n, dtype=torch.float64) + 1
    g = 1.32471795724474602596  # the plastic number
    return (torch.stack([(0.5 + i / (g * g)) % 1, (0.5 + i / g) % 1], 1) * (MODEL - 2) + 1).float()


class SlotTracker:
    """One state of ``slots`` query tokens over a camera's frames (module docstring); the last slot is never used.

    ``decoys``: a free slot follows a decoy point (a real query at a point of the image) instead of being a query of
    the future. Hundreds of identical future queries degrade the real tracks: on TUM RGB-D the model then marked 7-35 %
    of truly visible points hidden and lost more of them, against 2-5 % with the same points alone or among real
    queries (docs/design.md). A decoy that the model does not see is placed anew every ``DECOY_REFRESH`` frames. The
    decoys' positions are never reported as tracks."""

    def __init__(self, net, resolution: int, slots: int, device: torch.device, decoys: bool = False):
        self.net = _Net(net, resolution, device)
        self.slots = slots
        self.device = device
        self.state = None
        self.frames = 0  # frames given so far: the next frame's time
        self.queries = torch.zeros(1, slots + 1, 3, device=device)  # [t, y, x]; free slots: always the next frame + 1
        self.used = torch.zeros(slots + 1, dtype=torch.bool, device=device)
        self.decoys = decoys
        self.decoy = torch.zeros(slots + 1, dtype=torch.bool, device=device)  # free slots that follow a decoy point
        self._placed = 0  # decoy points placed so far (the sequence goes on)
        self._patches = None
        if decoys:
            self._place(torch.arange(slots, device=device))

    def prepare(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net.prepare(rgb)

    def _reset(self, slots: torch.Tensor):
        """The slots' states become the never-used slot's (a query that has not come yet)."""
        if self.state is not None and len(slots):
            rows = self._patches + slots
            for cache in self.state.hidden_state:
                for s in (cache.rg_lru_state, cache.conv1d_state):
                    s[rows] = s[self._patches + self.slots].clone()

    def _reset_where(self, which: torch.Tensor):
        """``_reset`` of the slots where ``which`` (slots + 1, bool) is set, without reading the mask on the CPU."""
        if self.state is not None:
            for cache in self.state.hidden_state:
                for s in (cache.rg_lru_state, cache.conv1d_state):
                    rows = s[self._patches:]
                    mask = which.reshape(-1, *([1] * (s.dim() - 1)))
                    rows.copy_(torch.where(mask, s[self._patches + self.slots], rows))

    def _place(self, slots: torch.Tensor):
        """Free ``slots`` follow new decoy points, queried at the next frame."""
        self._reset(slots)
        self.queries[0, slots, 0] = self.frames
        self.queries[0, slots, 1:] = decoy_points(len(slots), self._placed).to(self.device)
        self._placed += len(slots)
        self.decoy[slots] = True

    def add(self, slots: torch.Tensor, xy: torch.Tensor, height: int, width: int):
        """Free ``slots`` (long) become queries of the next frame at ``xy`` (pixels of the next frame)."""
        if bool(((slots >= self.slots) | self.used[slots.clamp(max=self.slots)]).any()):  # one synchronisation
            raise ValueError(f"slots {slots.tolist()}: not all free")
        if self.decoys:  # a decoy's slot has followed its point: reset it (a mask, no synchronisation)
            which = torch.zeros_like(self.decoy)
            which[slots] = True
            self._reset_where(which & self.decoy)
        self.decoy[slots] = False
        self.queries[0, slots, 0] = self.frames
        self.queries[0, slots, 1:] = to_model(xy.float(), height, width)
        self.used[slots] = True

    def free(self, slots: torch.Tensor):
        """``slots`` become free: queries of the future again (their states the never-used slot's), or decoys."""
        self.used[slots] = False
        if self.decoys:
            self._place(slots)  # resets them
        else:
            self._reset(slots)

    def step(self, frame: torch.Tensor):
        """The next frame (``prepare``'s) -> each slot's position (slots + 1, 2) model [y, x] and visibility; free
        slots' are meaningless."""
        future = ~self.used & ~self.decoy
        self.queries[0, future, 0] = self.frames + 1  # not yet
        self.queries[0, future, 1:] = MODEL / 2
        tracks, visible, self.state = self.net.forward(frame, self.queries, self.state)
        if self._patches is None:
            self._patches = self.state.hidden_state[0].rg_lru_state.shape[0] - self.queries.shape[1]
        self.frames += 1
        if self.decoys and self.frames % DECOY_REFRESH == 0:
            self._place(torch.nonzero(self.decoy & ~visible).flatten())
        return tracks, visible


class Catchup:
    """Points of an older frame followed to a newer one, through the frames between, in a state of their own."""

    def __init__(self, tracker: SlotTracker):
        self.net = tracker.net

    def run(self, frames: list, xy: torch.Tensor, height: int, width: int, starts: torch.Tensor | None = None):
        """``frames`` (prepared, oldest first) -> the points in the last frame (pixels) and whether they are visible
        there. Each point is of the frame its ``starts`` (index into ``frames``; default all the first) tells: one
        state for points of several frames, queried late as in SlotTracker."""
        queries = torch.zeros(1, len(xy), 3, device=xy.device)
        if starts is not None:
            queries[0, :, 0] = starts.float()
        queries[0, :, 1:] = to_model(xy.float(), height, width)
        state = None
        for frame in frames:
            tracks, visible, state = self.net.forward(frame, queries, state)
        return to_pixels(tracks, height, width), visible


def warm_up(net, resolution: int, slots: int, device: torch.device, decoys: bool) -> None:
    """Runs the network through the shapes a node gives it, so that a compiled network (``torch.compile``) compiles
    now and not on the first frames: the tracker's step (its first frame and later ones) and catch-ups of a few
    different point counts (the first two compile the static and the dynamic graph)."""
    tracker = SlotTracker(net, resolution, slots, device, decoys=decoys)
    frame = torch.zeros(1, 1, resolution, resolution, 3, device=device)
    xy = torch.tensor([[10.0, 10.0], [50.0, 40.0]], device=device)
    tracker.step(frame)
    tracker.add(torch.tensor([0, 1], device=device), xy, MODEL, MODEL)
    for _ in range(2):
        tracker.step(frame)
    catchup = Catchup(tracker)
    for n in (5, 17, 40):
        catchup.run([frame, frame], torch.rand(n, 2, device=device) * (MODEL - 1), MODEL, MODEL)
    torch.cuda.synchronize(device)
