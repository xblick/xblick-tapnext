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

import numpy as np
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
    for block in net.blocks:  # after the weights: the wrapper adds no parameter names
        block.vit_block.self_attention = _TokensFirst(block.vit_block.self_attention)
    return net.to(device).eval(), resolution


class _TokensFirst(torch.nn.Module):
    """TAPNext's attention (torchvision's ``nn.MultiheadAttention``, ``batch_first``) given its input laid out tokens
    first in memory when the batch has several frames (``SlotTrackers``). The attention turns its input to (tokens,
    batch, channels) and projects it; compiled, a projection of that view of a batch-first tensor became one small
    matrix product per token with the weights read for each (0.65 ms a layer at a batch of two, three times the
    whole pass's time). Laid out tokens first, the view is contiguous and the projection is one product. The same
    numbers either way; one frame passes as it is."""

    def __init__(self, attention: torch.nn.MultiheadAttention):
        super().__init__()
        self.attention = attention

    def forward(self, query, key, value, need_weights: bool = False):
        if query.shape[0] == 1 or not (query is key is value):
            return self.attention(query, key, value, need_weights=need_weights)
        x = query.transpose(0, 1).contiguous().transpose(0, 1)  # (batch, tokens, channels), tokens first in memory
        return self.attention(x, x, x, need_weights=need_weights)


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

    @torch.no_grad()
    def forward_batch(self, frames, queries, state):
        """``forward`` of B frames (B, 1, R, R, 3) and their queries (B, Q, 3) -> tracks (B, Q, 2) model [y, x],
        visibility (B, Q), the state."""
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            tracks, _, visible_logits, state = self.net(video=frames, query_points=queries, state=state)
        return tracks[:, 0].float(), visible_logits[:, 0, :, 0].float() > 0, state


def _scale(size: float) -> float:
    """MODEL / size or size / MODEL as float32 rounds it: the value a float32 scale tensor had (0.2.8 and older)."""
    return float(np.float32(size))


def to_model(xy: torch.Tensor, height: float, width: float) -> torch.Tensor:
    """(..., 2) pixels [x, y] (centres at integers) -> (..., 2) model [y, x] (centres at .5). The scales are numbers,
    not a tensor copied to the GPU (such a copy waits for the GPU's queued work); the values and the dtype are those
    of a float32 scale tensor."""
    xy = xy.to(torch.promote_types(xy.dtype, torch.float32))
    return torch.stack([(xy[..., 1] + 0.5) * _scale(MODEL / height), (xy[..., 0] + 0.5) * _scale(MODEL / width)], -1)


def to_pixels(yx: torch.Tensor, height: float, width: float) -> torch.Tensor:
    """(..., 2) model [y, x] -> (..., 2) pixels [x, y] (as ``to_model``, without a copy to the GPU)."""
    yx = yx.to(torch.promote_types(yx.dtype, torch.float32))
    return torch.stack([yx[..., 1] * _scale(width / MODEL), yx[..., 0] * _scale(height / MODEL)], -1) - 0.5


def to_device(values, dtype: torch.dtype, device) -> torch.Tensor:
    """A small host array (list, numpy, CPU tensor) on ``device`` without waiting for the GPU. A plain copy
    (``torch.tensor(..., device=)``, ``torch.as_tensor``) waits until the GPU has done all work queued before it, so
    the CPU could not prepare the next work while a pass runs. Copied from pinned memory, asynchronously; the same
    values."""
    device = torch.device(device)
    t = torch.as_tensor(values, dtype=dtype)
    if device.type != "cuda":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


def to_host(parts: list[torch.Tensor]) -> list[np.ndarray]:
    """Several tensors of one dtype read to the host in one copy (each copy to the host waits for the GPU): the
    arrays in their shapes and order."""
    if not parts:
        return []
    if len({p.dtype for p in parts}) != 1:
        raise ValueError(f"to_host: one dtype for all parts, got {sorted({str(p.dtype) for p in parts})}")
    flat = torch.cat([p.reshape(-1) for p in parts]).cpu().numpy()
    out, start = [], 0
    for p in parts:
        out.append(flat[start:start + p.numel()].reshape(tuple(p.shape)))
        start += p.numel()
    return out


def read_tracks(results: list[tuple[torch.Tensor, torch.Tensor]],
                sizes: list[tuple[float, float]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Steps' results (``SlotTracker.step``'s or ``SlotTrackers.step``'s (places model [y, x], visibility)) of
    trackers whose frames have the (height, width) of ``sizes`` -> each one's places in its frame's pixels (float64)
    and visibility (bool), read back in one copy for all of them."""
    if len(results) != len(sizes):
        raise ValueError(f"read_tracks: {len(results)} results and {len(sizes)} sizes")
    parts = [torch.cat([to_pixels(yx.float(), h, w), visible[:, None].float()], 1)
             for (yx, visible), (h, w) in zip(results, sizes, strict=True)]
    return [(rows[:, :2].astype(np.float64), rows[:, 2] > 0.5) for rows in to_host(parts)]


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
    decoys' positions are never reported as tracks.

    Nothing in ``add``, ``free`` and ``step`` waits for the GPU (no copy of a number to it, no read of a mask's count),
    except ``add``'s check that its slots are free (``check_slots``, one read-back) and the decoys' refresh every
    ``DECOY_REFRESH`` frames. ``check_slots=False`` leaves the check to the caller (one that keeps its own list of
    free slots): then a step's work can be queued while the GPU still runs the last one.

    ``net`` may be a ``tensorrt.TensorRTNetwork``: the later frames of trackers stepped together (``SlotTrackers``)
    then run in its engines."""

    def __init__(self, net, resolution: int, slots: int, device: torch.device, decoys: bool = False,
                 check_slots: bool = True):
        self.net = net.engine_net(resolution, device) if hasattr(net, "engine_net") else _Net(net, resolution, device)
        self.slots = slots
        self.device = device
        self.check_slots = check_slots
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
        """Free ``slots`` follow new decoy points, queried at the next frame (the points from pinned memory, the
        values as numbers: no wait)."""
        self._reset(slots)
        self.queries[0, :, 0].index_fill_(0, slots, float(self.frames))
        self.queries[0, slots, 1:] = to_device(decoy_points(len(slots), self._placed), torch.float32, self.device)
        self._placed += len(slots)
        self.decoy.index_fill_(0, slots, True)

    def add(self, slots: torch.Tensor, xy: torch.Tensor, height: float, width: float):
        """Free ``slots`` (long) become queries of the next frame at ``xy`` (pixels of the next frame). With
        ``check_slots`` a used or out-of-range slot raises (one read-back)."""
        if self.check_slots and bool(((slots >= self.slots) | self.used[slots.clamp(max=self.slots)]).any()):
            raise ValueError(f"slots {slots.tolist()}: not all free")
        if self.decoys:  # a decoy's slot has followed its point: reset it (a mask, no synchronisation)
            which = torch.zeros_like(self.decoy).index_fill_(0, slots, True)
            self._reset_where(which & self.decoy)
        # index_fill_ takes the value as a number; x[slots] = value would copy it to the GPU (and wait)
        self.decoy.index_fill_(0, slots, False)
        self.queries[0, :, 0].index_fill_(0, slots, float(self.frames))
        self.queries[0, slots, 1:] = to_model(xy.float(), height, width)
        self.used.index_fill_(0, slots, True)

    def free(self, slots: torch.Tensor):
        """``slots`` become free: queries of the future again (their states the never-used slot's), or decoys."""
        self.used.index_fill_(0, slots, False)
        if self.decoys:
            self._place(slots)  # resets them
        else:
            self._reset(slots)

    def step(self, frame: torch.Tensor):
        """The next frame (``prepare``'s) -> each slot's position (slots + 1, 2) model [y, x] and visibility; free
        slots' are meaningless."""
        tracks, visible, state = self.net.forward(frame, self._queries(), self.state)
        self._stepped(state, visible)
        return tracks, visible

    def _queries(self) -> torch.Tensor:
        """The queries of the next step, (1, slots + 1, 3) [t, y, x], t in this tracker's frames. Set through a mask
        (torch.where): boolean indexing would read the mask's count back."""
        future = ~self.used & ~self.decoy
        q = self.queries[0]
        q[:, 0] = torch.where(future, float(self.frames + 1), q[:, 0])  # not yet
        q[:, 1:] = torch.where(future[:, None], MODEL / 2, q[:, 1:])
        return self.queries

    def _stepped(self, state, visible: torch.Tensor):
        """After a step: its state, the frame counted, the decoys the model did not see placed anew."""
        self.state = state
        if self._patches is None:
            self._patches = state.hidden_state[0].rg_lru_state.shape[0] - self.queries.shape[1]
        self.frames += 1
        if self.decoys and self.frames % DECOY_REFRESH == 0:
            self._place(torch.nonzero(self.decoy & ~visible).flatten())


class SlotTrackers:
    """Several SlotTrackers of one network and one number of slots, each with a frame of its own (a window of the
    camera's image, say), stepped in one network pass: the frames and the queries are a batch, and each tracker's state
    is its part of the batch's state. Each tracker steps as it would alone, up to float16 rounding: the network keeps
    the batch's elements apart (the recurrent units are per token, the attention is within a frame).

    The RG-LRU treats a state's first frame apart from the later ones, for the whole batch, so trackers whose first
    frame it is go through a pass of their own (once each). Each tracker's ``state`` holds views of the batch's state.
    When the next ``step`` has the same trackers in the same order, that state is passed on as it is; otherwise the
    trackers' states are concatenated (a copy of about 0.1 GB per tracker of 64 slots). ``add`` and ``free`` work on
    each tracker as before, and a tracker can still be stepped alone (``SlotTracker.step``)."""

    def __init__(self):
        self._trackers: tuple = ()  # the trackers of the last pass with a state, in its order
        self._states: tuple = ()  # their states after it (views of _hidden)
        self._hidden = None  # that pass's state's hidden_state

    def step(self, trackers: list[SlotTracker], frames: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Each tracker's next frame (``prepare``'s or its like, (1, 1, R, R, 3)) -> what ``SlotTracker.step`` returns,
        per tracker in the order given."""
        if len(trackers) != len(frames) or len({id(t) for t in trackers}) != len(trackers):
            raise ValueError(f"{len(trackers)} trackers (each once) and {len(frames)} frames: one frame per tracker")
        for t in trackers[1:]:
            if t.net.net is not trackers[0].net.net or t.slots != trackers[0].slots:
                raise ValueError("SlotTrackers: all trackers of one network and one number of slots")
        out: list = [None] * len(trackers)
        groups = {fresh: [i for i, t in enumerate(trackers) if (t.state is None) == fresh] for fresh in (True, False)}
        for fresh, order in groups.items():
            if order:
                results = self._pass([trackers[i] for i in order], torch.cat([frames[i] for i in order]), fresh)
                for i, result in zip(order, results, strict=True):
                    out[i] = result
        return out

    def _pass(self, trackers: list[SlotTracker], frames: torch.Tensor, fresh: bool):
        tapnext_torch = _tapnext_module()
        cache = tapnext_torch.tapnext_lru_modules.RecurrentBlockCache
        queries = torch.cat([t._queries() for t in trackers])
        state = None
        if not fresh:
            # each tracker's queries are in its own frames: the batch's state is at step 0 (forward subtracts it);
            # the steps from pinned memory (a plain copy to the GPU would wait for it)
            steps = to_device([t.state.step for t in trackers], queries.dtype, queries.device)
            queries = torch.cat([queries[..., :1] - steps[:, None, None], queries[..., 1:]], -1)
            if tuple(trackers) == self._trackers and all(
                    t.state is s for t, s in zip(trackers, self._states, strict=True)):
                hidden = self._hidden
            else:
                hidden = [cache(torch.cat([t.state.hidden_state[k].rg_lru_state for t in trackers]),
                                torch.cat([t.state.hidden_state[k].conv1d_state for t in trackers]))
                          for k in range(len(trackers[0].state.hidden_state))]
            state = tapnext_torch.TAPNextTrackingState(step=0, query_points=queries, hidden_state=hidden)
        tracks, visible, state = trackers[0].net.forward_batch(frames, queries, state)
        rows = state.hidden_state[0].rg_lru_state.shape[0] // len(trackers)
        parts = [(c.rg_lru_state.split(rows), c.conv1d_state.split(rows)) for c in state.hidden_state]
        results = []
        for b, t in enumerate(trackers):
            own = tapnext_torch.TAPNextTrackingState(
                step=(0 if fresh else t.state.step) + 1, query_points=t.queries,
                hidden_state=[cache(lru[b], conv[b]) for lru, conv in parts])
            t._stepped(own, visible[b])
            results.append((tracks[b], visible[b]))
        if not fresh:
            self._trackers, self._states = tuple(trackers), tuple(t.state for t in trackers)
            self._hidden = state.hidden_state
        return results


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


RECOMPILE_LIMIT = 32  # dynamo's graphs per function with batched trackers (warm_up)


def warm_up(net, resolution: int, slots: int, device: torch.device, decoys: bool, batch: int = 1) -> None:
    """Runs the network through the shapes a node gives it, so that a compiled network (``torch.compile``) compiles
    now and not on the first frames: the tracker's step (its first frame and later ones) and catch-ups of a few
    different point counts (the first two compile the static and the dynamic graph). With ``batch`` > 1 also
    ``SlotTrackers`` of 1 to ``batch`` trackers (first frames, later ones, the batch's state passed on and
    concatenated; from the second size on the batch's size is dynamic). These need more graphs than dynamo's default
    limit of 8 per function, beyond which it would run the network uncompiled: the limit is raised to
    ``RECOMPILE_LIMIT``."""
    tracker = SlotTracker(net, resolution, slots, device, decoys=decoys)
    # a frame as prepare gives it (its memory layout too: another layout would compile again on the first frame)
    frame = tracker.prepare(torch.zeros(resolution, resolution, 3, dtype=torch.uint8, device=device))
    xy = torch.tensor([[10.0, 10.0], [50.0, 40.0]], device=device)
    tracker.step(frame)
    tracker.add(torch.tensor([0, 1], device=device), xy, MODEL, MODEL)
    for _ in range(2):
        tracker.step(frame)
    catchup = Catchup(tracker)
    for n in (5, 17, 40):
        catchup.run([frame, frame], torch.rand(n, 2, device=device) * (MODEL - 1), MODEL, MODEL)
    if batch > 1:
        from torch import _dynamo  # noqa: PLC0415

        _dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, RECOMPILE_LIMIT)
    for size in range(1, batch + 1) if batch > 1 else ():
        trackers = [SlotTracker(net, resolution, slots, device, decoys=decoys) for _ in range(size)]
        together = SlotTrackers()
        for _ in range(3):
            together.step(trackers, [frame] * size)
    torch.cuda.synchronize(device)
