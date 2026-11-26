"""TAPNext++'s later frames of slot trackers stepped together (``model.SlotTrackers``) in TensorRT engines.

    from xblick_tapnext.model import load, SlotTracker, SlotTrackers
    from xblick_tapnext.tensorrt import TensorRTNetwork

    net, resolution = load("256", device, STORE.root)
    engines = TensorRTNetwork(net, "256", resolution, device)   # engine_dir: default_engine_dir()
    engines.build(64, 4)                                         # trackers of 64 slots, 1 to 4 together
    trackers = [SlotTracker(engines, resolution, 64, device, decoys=True) for _ in range(4)]
    SlotTrackers().step(trackers, frames)

Needs the optional dependencies ``trt`` (Torch-TensorRT of the installed PyTorch line). The network is exported with
PyTorch's float16 autocast in the graph, as ``model._Net`` runs it (TensorRT 11 is strongly typed and takes each
layer's precision from the graph), from a copy that TensorRT takes for a batch of any size (``for_export``). For one
number of slots there are two engines: one for a tracker alone and one for 2 to ``most`` trackers together (a dynamic
batch; torch.export takes a batch of one apart). A tracker's first frame, and a single tracker's ``step``, run in
PyTorch, not compiled: the RG-LRU treats a state's first frame apart. Each engine is built once (about 35 to 90 s)
and kept in ``engine_dir``, named by the checkpoint, the input, the slots, the batch sizes, the GPU and the versions;
a later start loads it. docs/performance.md has the times.
"""

from __future__ import annotations

import copy
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import _Net, _tapnext_module


def default_engine_dir() -> Path:
    """The engines' default directory: ``tensorrt`` in the node's cache directory (~/.cache/xblick/tapnext)."""
    from xblick.paths import cache_dir  # noqa: PLC0415

    return cache_dir("tapnext") / "tensorrt"


def for_export(module):
    """A copy of TAPNext++ for the engines, the same computation under float16 autocast, put so that TensorRT takes it
    for a batch of any size and with less in the engine:

    * the weights of the layers autocast runs in float16 (Linear, Conv2d, the attention's input projection) stored in
      float16, the values autocast casts them to in each pass;
    * the queries' position embedding sampled from its one table for all trackers of the batch (DeepMind's
      embed_queries repeats the 256 x 256 x 768 table per tracker: 0.2 GB in the engine per tracker), and the tokens
      and the prediction heads' index broadcast instead of repeated by the batch (Torch-TensorRT 2.14 converts no
      repeat by a dynamic batch);
    * the attention without ``model._TokensFirst``, which branches on a batch of one; the wrapper only lays the input
      out for torch.compile.

    For the static case (one tracker) the copy gives the network's numbers bit for bit."""
    net = copy.deepcopy(module)
    for m in net.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            m.half()
        elif isinstance(m, torch.nn.MultiheadAttention) and m.in_proj_weight is not None:
            m.in_proj_weight.data = m.in_proj_weight.data.half()
            if m.in_proj_bias is not None:
                m.in_proj_bias.data = m.in_proj_bias.data.half()
    for block in net.blocks:
        attention = block.vit_block.self_attention
        if not isinstance(attention, torch.nn.MultiheadAttention):  # model._TokensFirst
            block.vit_block.self_attention = attention.attention

    def embed_queries(self, timesteps, query_points):
        """DeepMind's TAPNext.embed_queries for one frame (online tracking, t == 1)."""
        if timesteps != 1:
            raise ValueError("the engines' embed_queries: one frame at a time")
        b, q, _ = query_points.shape
        c = self.width
        query_timesteps, query_positions = query_points[..., :1], query_points[..., 1:]
        table = self.query_pos_embed.view(1, self.image_size[0], self.image_size[1], c).permute(0, 3, 2, 1)
        grid = (query_positions.reshape(1, 1, b * q, 2) / torch.tensor(self.image_size, device=query_points.device)) \
            * 2 - 1
        spatial = F.grid_sample(table, grid, align_corners=False).permute(0, 2, 3, 1).reshape(b, 1, q, c)
        tokens = torch.where(query_timesteps.unsqueeze(1) == 0, self.point_query_token + spatial,
                             self.mask_token.expand(b, 1, q, c))
        is_unknown = torch.zeros(1, 1, 1, 1, device=query_points.device) < query_timesteps.unsqueeze(1)
        return torch.where(is_unknown, self.unknown_token.reshape(1, 1, 1, c).expand(b, 1, q, c), tokens)

    def prediction_heads(self, x):
        """DeepMind's TAPNext.prediction_heads (soft argmax within 20 of the argmax, temperature 0.5), its index
        broadcast."""
        track_logits = self.coordinate_head(x.float())
        position_x, position_y = track_logits.chunk(2, dim=-1)
        index = torch.arange(position_x.shape[-1], device=x.device)
        out = []
        for position in (position_x, position_y):
            near = (torch.abs(position.argmax(dim=-1, keepdim=True) - index) <= 20).float()
            probs = F.softmax(position * 0.5, dim=-1) * near
            probs = probs / probs.sum(dim=-1, keepdim=True)
            out.append(torch.sum(probs * index, dim=-1)[..., None])
        return torch.cat(out, -1) + 0.5, track_logits, self.visible_head(x)

    net.embed_queries = types.MethodType(embed_queries, net)
    net.prediction_heads = types.MethodType(prediction_heads, net)
    return net


class _Flat(torch.nn.Module):
    """TAPNext++ on a later frame of trackers stepped together, with plain tensors (what TensorRT takes): frames (B, 1,
    R, R, 3), queries (B, Q, 3) in the batch's own frames (step 0, as SlotTrackers passes them), then each block's
    rg_lru and conv1d state -> tracks (B, Q, 2) model [y, x], visibility logits (B, Q), the blocks' new states in the
    dtypes they came in. Under PyTorch's float16 autocast, as model._Net runs the network: exported, the casts are in
    the graph."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, frames, queries, *states):
        tn = _tapnext_module()
        cache = tn.tapnext_lru_modules.RecurrentBlockCache
        hidden = [cache(states[2 * k], states[2 * k + 1]) for k in range(len(states) // 2)]
        state = tn.TAPNextTrackingState(step=0, query_points=queries, hidden_state=hidden)
        with torch.autocast("cuda", dtype=torch.float16):
            tracks, _, visible, new = self.net(video=frames, query_points=queries, state=state)
        out = [tracks[:, 0].float(), visible[:, 0, :, 0].float()]
        for k, c in enumerate(new.hidden_state):
            out += [c.rg_lru_state.to(states[2 * k].dtype), c.conv1d_state.to(states[2 * k + 1].dtype)]
        return tuple(out)


class TensorRTNetwork:
    """TAPNext++ (``model.load``'s network, not compiled) whose later frames of trackers stepped together run in
    TensorRT engines (module docstring). ``build(slots, most)`` loads or builds the engines for 1 to ``most`` trackers
    of ``slots`` slots; ``SlotTracker(this, ...)`` steps with it. ``checkpoint`` names the engines (a CHECKPOINTS name
    or a file's stem); ``engine_dir`` keeps them (default: ``default_engine_dir()``)."""

    def __init__(self, module, checkpoint: str, resolution: int, device: torch.device, engine_dir: Path | None = None):
        try:
            import torch_tensorrt  # noqa: F401, PLC0415
        except ImportError as error:
            raise RuntimeError("TensorRT engines need the optional dependencies trt (xblick-tapnext[trt]: "
                               "Torch-TensorRT)") from error
        self.module, self.checkpoint, self.resolution = module, checkpoint, resolution
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:  # Torch-TensorRT needs the index
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.engine_dir = Path(engine_dir) if engine_dir is not None else default_engine_dir()
        self.engines: dict[tuple[int, int], torch.nn.Module] = {}  # (slots, trackers) -> engine
        self._exportable = None

    def exportable(self):
        """The copy of the network the engines are exported from (``for_export``), made once."""
        if self._exportable is None:
            self._exportable = for_export(self.module)
        return self._exportable

    def engine_net(self, resolution: int, device: torch.device):
        """What a SlotTracker steps with (``model._Net``'s methods): a batch's later frames in an engine."""
        if resolution != self.resolution:
            raise ValueError(f"TensorRT engines of input {self.resolution}, a tracker of {resolution}")
        return _EngineNet(self, resolution, device)

    def path(self, slots: int, least: int, most: int) -> Path:
        """The file of the engine for ``least`` to ``most`` trackers of ``slots`` slots on this GPU."""
        import tensorrt  # noqa: PLC0415
        import torch_tensorrt  # noqa: PLC0415

        major, minor = torch.cuda.get_device_capability(self.device)
        name = Path(self.checkpoint).stem if Path(self.checkpoint).suffix else self.checkpoint
        trackers = f"{least}" if least == most else f"{least}-{most}"
        return self.engine_dir / (f"tapnext_{name}_r{self.resolution}_s{slots}_b{trackers}_sm{major}{minor}"
                                  f"_trt{tensorrt.__version__}_tt{torch_tensorrt.__version__}.pt2")

    def build(self, slots: int, most: int):
        """The engines for 1 to ``most`` trackers of ``slots`` slots (loaded, or built and kept): one for a tracker
        alone, one for 2 to ``most`` together."""
        if most < 1:
            raise ValueError(f"TensorRT engines for 1 to {most} trackers: at least one")
        self.engines[slots, 1] = self._engine(slots, 1, 1)
        if most > 1:
            together = self._engine(slots, 2, most)
            for batch in range(2, most + 1):
                self.engines[slots, batch] = together
        self._exportable = None  # its memory back

    def _engine(self, slots: int, least: int, most: int) -> torch.nn.Module:
        import torch_tensorrt  # noqa: PLC0415

        path = self.path(slots, least, most)
        if path.is_file():
            return torch_tensorrt.load(str(path)).module()
        r, dev = self.resolution, self.device
        frames = torch.zeros(most, 1, r, r, 3, device=dev)
        queries = torch.zeros(most, slots + 1, 3, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            *_, state = self.module(video=frames, query_points=queries, state=None)
        example = [frames, queries] + [s.contiguous() for c in state.hidden_state
                                       for s in (c.rg_lru_state, c.conv1d_state)]
        rows = example[2].shape[0] // most  # each tracker's rows of the state: patches and queries
        dynamic, inputs = None, example
        if most > least:
            batch = torch.export.Dim("trackers", min=least, max=most)
            dynamic = ({0: batch}, {0: batch}, tuple({0: rows * batch} for _ in example[2:]))
            inputs = [torch_tensorrt.Input(min_shape=(least * per, *x.shape[1:]), opt_shape=(most * per, *x.shape[1:]),
                                           max_shape=(most * per, *x.shape[1:]), dtype=x.dtype)
                      for per, x in zip([1, 1] + [rows] * len(example[2:]), example, strict=True)]
        with torch.no_grad():
            exported = torch.export.export(_Flat(self.exportable()).eval(), tuple(example), dynamic_shapes=dynamic,
                                           strict=False)
            engine = torch_tensorrt.dynamo.compile(exported, arg_inputs=inputs, device=dev, min_block_size=1,
                                                   use_python_runtime=False)
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.stem + ".part.pt2")  # never a partial file under the final name
        # the compiled module takes the tensors one by one: its dynamic shapes flat
        flat = None if dynamic is None else (dynamic[0], dynamic[1], *dynamic[2])
        torch_tensorrt.save(engine, str(part), output_format="exported_program", arg_inputs=example,
                            dynamic_shapes=flat)
        part.rename(path)
        return engine


class _EngineNet:
    """``model._Net`` for a TensorRTNetwork: ``forward`` and a batch's first frame in PyTorch, a batch's later frames
    in the engine of its size."""

    def __init__(self, backend: TensorRTNetwork, resolution: int, device: torch.device):
        self.net, self.resolution, self.device = backend, resolution, device
        self._torch = _Net(backend.module, resolution, device)

    def prepare(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._torch.prepare(rgb)

    def forward(self, frame, queries, state):
        return self._torch.forward(frame, queries, state)

    @torch.no_grad()
    def forward_batch(self, frames, queries, state):
        if state is None:
            return self._torch.forward_batch(frames, queries, state)
        key = (queries.shape[1] - 1, frames.shape[0])
        engine = self.net.engines.get(key)
        if engine is None:
            raise RuntimeError(f"no TensorRT engine for {key[1]} trackers of {key[0]} slots (built: "
                               f"{sorted(self.net.engines)})")
        if state.step != 0:
            raise RuntimeError("TensorRT engines take the batch's state at step 0 (SlotTrackers)")
        tn = _tapnext_module()
        flat = [s.contiguous() for c in state.hidden_state for s in (c.rg_lru_state, c.conv1d_state)]
        out = engine(frames.contiguous(), queries.contiguous(), *flat)
        cache = tn.tapnext_lru_modules.RecurrentBlockCache
        hidden = [cache(out[2 + 2 * k], out[3 + 2 * k]) for k in range(len(flat) // 2)]
        return out[0], out[1] > 0, tn.TAPNextTrackingState(step=1, query_points=queries, hidden_state=hidden)
