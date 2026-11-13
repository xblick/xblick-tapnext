"""Which slot of the tracker holds which track, and the rules of a track's life (README, "Tracks").

A track is a point followed from the frame it was given in: ``track_id`` (1, 2, ...: never used twice while the node
runs), the object it was seeded on (``object_id``, the segmenter's instance id; 0: given by hand), when it was last
visible. It ends

* when it has not been visible for ``forget`` seconds,
* when its object has not been in the segmenter's results for ``forget`` seconds,
* when it drifted: visible, but outside its object's mask (grown by a margin) in ``DRIFT_RESULTS`` results in a row,
* when new points need its slot and no slot is free: the tracks hidden longest first (``hidden_longest``),
* when the node is told to clear the tracks.

A segmenter may give an object a new id (its own tracker lost it and found it again). A track of an object not in a
result that the tracker saw inside the mask of an object in it then becomes that object's (``orphans``, ``adopt``):
the object keeps its points instead of getting new ones beside the old, which would wait ``forget`` seconds to end.

An object gets new points when fewer than half of ``per_object`` of its tracks are visible inside its mask: as many as
make ``per_object`` visible ones again. Its tracks then may be more than ``per_object`` (hidden ones wait to be seen
again); at more than twice ``per_object`` its longest hidden ones end.
"""

from __future__ import annotations

import numpy as np

DRIFT_RESULTS = 2  # results in a row a visible track lies outside its object's mask before it ends


class Tracks:
    def __init__(self, slots: int, per_object: int, forget_ns: int):
        self.slots = slots
        self.per_object = per_object
        self.forget_ns = forget_ns
        self.track_id = np.zeros(slots, np.uint32)  # 0: free
        self.object_id = np.zeros(slots, np.uint16)
        self.last_visible = np.zeros(slots, np.int64)
        self.started = np.zeros(slots, np.int64)  # the stamp of the frame the track was given in
        self.outside = np.zeros(slots, np.int32)  # results in a row outside the object's mask
        self.objects: dict[int, int] = {}  # object id -> stamp it was last in the segmenter's results
        self._next_id = 1
        self.newest = -1  # the stamp of the newest frame the tracker gave (update)

    @property
    def used(self) -> np.ndarray:
        return self.track_id != 0

    def free_slots(self) -> np.ndarray:
        return np.nonzero(~self.used)[0]

    def start(self, slots: np.ndarray, object_id: int, stamp: int) -> np.ndarray:
        """New tracks in free ``slots``; their track ids."""
        if self.used[slots].any():
            raise ValueError(f"slots {slots.tolist()}: not all free")
        ids = np.arange(self._next_id, self._next_id + len(slots), dtype=np.uint32)
        self._next_id += len(slots)
        self.track_id[slots], self.object_id[slots] = ids, object_id
        self.last_visible[slots], self.started[slots], self.outside[slots] = stamp, stamp, 0
        return ids

    def end(self, slots: np.ndarray):
        self.track_id[slots] = 0
        self.object_id[slots] = 0

    def update(self, visible: np.ndarray, stamp: int):
        """The tracker's visibility of each slot in the frame of ``stamp``."""
        self.last_visible[self.used & visible] = stamp
        self.newest = stamp

    def expired(self, now: int) -> np.ndarray:
        """Slots of tracks whose time is up at ``now`` (not visible, or their object not seen, for ``forget``)."""
        hidden = now - self.last_visible > self.forget_ns
        gone = np.array([o != 0 and now - self.objects.get(int(o), now) > self.forget_ns for o in self.object_id])
        return np.nonzero(self.used & (hidden | gone))[0]

    def orphans(self, present: list[int], visible: np.ndarray, track_id: np.ndarray) -> np.ndarray:
        """Slots of tracks the tracker saw in a result's frame (``track_id`` the slots' tracks then) whose object is not
        in that result (``present``); not those given by hand."""
        same = self.used & (self.track_id == track_id)
        return np.nonzero(same & visible & (self.object_id != 0) & ~np.isin(self.object_id, present))[0]

    def adopt(self, slots: np.ndarray, object_id: int):
        """Tracks in ``slots`` become ``object_id``'s (orphans inside its mask)."""
        self.object_id[slots] = object_id
        self.outside[slots] = 0

    def check(self, stamp: int, present: list[int], visible: np.ndarray, inside: np.ndarray, track_id: np.ndarray):
        """A segmenter's result of the frame ``stamp``: the objects in it (``present``), and for each slot as it was
        then (``track_id`` its track then) whether the tracker saw it then and whether it lay in its object's mask.
        Returns the slots of tracks that drifted, and for each present object how many new points it needs."""
        for o in present:
            self.objects[o] = max(self.objects.get(o, stamp), stamp)
        same = self.used & (self.track_id == track_id)  # the track that was in the slot then is still there
        checked = same & visible & np.isin(self.object_id, present)
        self.outside[checked & inside] = 0
        self.outside[checked & ~inside] += 1
        drifted = np.nonzero(checked & (self.outside >= DRIFT_RESULTS))[0]
        good = same & visible & inside
        good[drifted] = False
        good |= self.used & (self.started > stamp)  # given after that frame (from an earlier result): not yet seen
        needs = {}
        for o in present:
            have = int((good & (self.object_id == o)).sum())
            if have < self.per_object / 2:
                needs[o] = self.per_object - have
        return drifted, needs

    def hidden_longest(self, n: int) -> np.ndarray:
        """Slots of up to ``n`` tracks not visible in the newest frame, those hidden longest first."""
        hidden = np.nonzero(self.used & (self.last_visible < self.newest))[0]
        return hidden[np.argsort(self.last_visible[hidden], kind="stable")[:n]]

    def surplus(self, object_id: int, coming: int) -> np.ndarray:
        """Slots of the object's longest hidden tracks to end so that with ``coming`` new ones it has at most twice
        ``per_object``."""
        mine = np.nonzero(self.used & (self.object_id == object_id))[0]
        over = len(mine) + coming - 2 * self.per_object
        if over <= 0:
            return mine[:0]
        return mine[np.argsort(self.last_visible[mine], kind="stable")[:over]]
