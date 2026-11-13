"""Where new points go in an object's mask: inside it by a margin (a point on the mask's edge slides onto what is
behind the object), spread evenly (farthest point sampling: each new point the mask's pixel farthest from the points
it already has and the ones chosen before it)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

MAX_CANDIDATES = 20000  # mask pixels considered (a regular subset of a larger mask)


def shrink(mask: torch.Tensor, margin: int) -> torch.Tensor:
    """The mask's pixels at least ``margin`` pixels (a square's half side) inside it."""
    if margin <= 0:
        return mask
    outside = (~mask).float()[None, None]
    return F.max_pool2d(outside, 2 * margin + 1, stride=1, padding=margin)[0, 0] == 0


def grow(mask: torch.Tensor, margin: int) -> torch.Tensor:
    """The mask and the pixels up to ``margin`` pixels around it."""
    if margin <= 0:
        return mask
    return F.max_pool2d(mask.float()[None, None], 2 * margin + 1, stride=1, padding=margin)[0, 0] > 0


def sample(mask: torch.Tensor, n: int, have: torch.Tensor, margin: int) -> torch.Tensor:
    """Up to ``n`` new points (k, 2) pixels [x, y] in ``mask`` (bool (H, W)), ``margin`` inside it (the whole mask if
    nothing is that far inside), far from ``have`` (m, 2) and from each other. Fewer than ``n`` only if the mask has
    fewer pixels."""
    inner = shrink(mask, margin)
    if not bool(inner.any()):
        inner = mask
    ys, xs = torch.nonzero(inner, as_tuple=True)
    if len(xs) > MAX_CANDIDATES:
        step = -(-len(xs) // MAX_CANDIDATES)
        ys, xs = ys[::step], xs[::step]
    candidates = torch.stack([xs, ys], 1).float()
    if len(candidates) == 0 or n <= 0:
        return candidates[:0]
    chosen = []
    if len(have):
        nearest = torch.cdist(candidates, have.float()).min(dim=1).values
    else:  # the first: the pixel nearest the mask's centre
        chosen.append(int((candidates - candidates.mean(0)).norm(dim=1).argmin()))
        nearest = (candidates - candidates[chosen[0]]).norm(dim=1)
    for _ in range(min(n, len(candidates)) - len(chosen)):
        i = int(nearest.argmax())
        chosen.append(i)
        nearest = torch.minimum(nearest, (candidates - candidates[i]).norm(dim=1))
    return candidates[chosen]
