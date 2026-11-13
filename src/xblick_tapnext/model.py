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

REPO = Path(__file__).resolve().parents[2]
TAPNET = REPO / "third_party" / "tapnet"
WEIGHTS = REPO / "weights"
MODEL = 256  # TAPNext's coordinates: 256 x 256 whatever the input resolution
# checkpoints in weights/: the input resolution they were fine-tuned at (DeepMind's TAPNext++ README and VOTSp2026
# tracker; Apache 2.0)
CHECKPOINTS = {
    "256": ("tapnextpp_ckpt.pt", 256, "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt",
            "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"),
    "512": ("tapnextpp_512.ckpt", 512, "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt",
            "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f"),
}


def _tapnext_module():
    if not (TAPNET / "tapnet" / "tapnext" / "tapnext_torch.py").exists():
        raise FileNotFoundError(f"{TAPNET}: DeepMind's tapnet missing (tools/setup.sh: git submodule update --init)")
    if str(TAPNET) not in sys.path:
        sys.path.insert(0, str(TAPNET))  # tapnet.tapnext.* only: the package's dependencies (JAX ...) not needed
    from tapnet.tapnext import tapnext_torch  # noqa: PLC0415

    return tapnext_torch


def load(checkpoint: str, device: torch.device):
    """TAPNext++'s network (``CHECKPOINTS`` name, or a path to a .pt of a 256-resolution TAPNext) and its input
    resolution."""
    tapnext_torch = _tapnext_module()
    if checkpoint in CHECKPOINTS:
        name, resolution, _, _ = CHECKPOINTS[checkpoint]
        path = WEIGHTS / name
        if not path.is_file():
            raise FileNotFoundError(f"{path}: missing (tools/setup.sh)")
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


class SlotTracker:
    """One state of ``slots`` query tokens over a camera's frames (module docstring); the last slot is never used."""

    def __init__(self, net, resolution: int, slots: int, device: torch.device):
        self.net = _Net(net, resolution, device)
        self.slots = slots
        self.device = device
        self.state = None
        self.frames = 0  # frames given so far: the next frame's time
        self.queries = torch.zeros(1, slots + 1, 3, device=device)  # [t, y, x]; free slots: always the next frame + 1
        self.used = torch.zeros(slots + 1, dtype=torch.bool, device=device)
        self._patches = None

    def prepare(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net.prepare(rgb)

    def add(self, slots: torch.Tensor, xy: torch.Tensor, height: int, width: int):
        """Free ``slots`` (long) become queries of the next frame at ``xy`` (pixels of the next frame)."""
        if bool(self.used[slots].any()) or bool((slots >= self.slots).any()):
            raise ValueError(f"slots {slots.tolist()}: not all free")
        self.queries[0, slots, 0] = self.frames
        self.queries[0, slots, 1:] = to_model(xy.float(), height, width)
        self.used[slots] = True

    def free(self, slots: torch.Tensor):
        """``slots`` become free: queries of the future again, their states the never-used slot's."""
        self.used[slots] = False
        if self.state is not None:
            for cache in self.state.hidden_state:
                for s in (cache.rg_lru_state, cache.conv1d_state):
                    s[self._patches + slots] = s[self._patches + self.slots].clone()

    def step(self, frame: torch.Tensor):
        """The next frame (``prepare``'s) -> each slot's position (slots + 1, 2) model [y, x] and visibility; free
        slots' are meaningless."""
        free = ~self.used
        self.queries[0, free, 0] = self.frames + 1  # not yet
        self.queries[0, free, 1:] = MODEL / 2
        tracks, visible, self.state = self.net.forward(frame, self.queries, self.state)
        if self._patches is None:
            self._patches = self.state.hidden_state[0].rg_lru_state.shape[0] - self.queries.shape[1]
        self.frames += 1
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
