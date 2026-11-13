"""The tracks' bookkeeping (tracks.py), the seeding (seeding.py), the lifting (lifting.py), the queryable's requests
(queries.py) and the published layout (node.cloud): no GPU."""

import numpy as np
import pytest
import torch

from xblick_tapnext.lifting import Lifter
from xblick_tapnext.node import FIELDS, POINT_STEP, cloud
from xblick_tapnext.queries import parse
from xblick_tapnext.seeding import grow, sample, shrink
from xblick_tapnext.tracks import DRIFT_RESULTS, Tracks

S = 1_000_000_000  # a second in ns


def test_tracks_get_new_ids_and_end_when_hidden_or_their_object_is_gone():
    t = Tracks(8, per_object=4, forget_ns=6 * S)
    first = t.start(np.array([0, 1]), 7, 10 * S)
    second = t.start(np.array([2]), 0, 10 * S)
    assert first.tolist() == [1, 2] and second.tolist() == [3]
    t.objects[7] = 12 * S
    t.update(np.array([True, False, True] + [False] * 5), 12 * S)
    assert t.expired(16 * S).tolist() == []
    assert t.expired(16 * S + 1).tolist() == [1]  # hidden since 10 s
    t.end(np.array([1]))
    assert t.start(np.array([1]), 7, 17 * S).tolist() == [4]  # never an id twice
    t.update(np.array([True] * 8), 18 * S)
    # object 7 last seen at 12 s: its tracks end after 18 s; the hand-given one (object 0) does not
    assert t.expired(18 * S).tolist() == []
    assert t.expired(18 * S + 1).tolist() == [0, 1]


def test_drift_ends_tracks_outside_their_mask_and_objects_get_points_back():
    t = Tracks(16, per_object=4, forget_ns=6 * S)
    t.start(np.arange(4), 5, 1 * S)
    visible = np.zeros(16, bool)
    visible[:4] = True
    inside = np.zeros(16, bool)
    inside[:3] = True  # slot 3 lies outside
    ids = t.track_id.copy()
    for k in range(DRIFT_RESULTS):
        drifted, needs = t.check((2 + k) * S, [5, 9], visible, inside, ids)
        if k < DRIFT_RESULTS - 1:
            assert drifted.tolist() == [] and needs == {9: 4}  # 3 of 4 good: enough; object 9 has none
    assert drifted.tolist() == [3]
    t.end(drifted)
    # now 3 good of 4: still at least half
    _, needs = t.check(5 * S, [5], visible, inside, ids)
    assert needs == {}
    inside[:3] = False
    _, needs = t.check(6 * S, [5], visible, inside, ids)
    assert needs == {5: 4}


def test_a_slot_given_to_another_track_since_is_not_judged_by_the_old_ones_position():
    t = Tracks(4, per_object=4, forget_ns=6 * S)
    t.start(np.array([0, 1]), 5, 1 * S)
    then = t.track_id.copy()
    t.end(np.array([1]))
    t.start(np.array([1]), 5, 3 * S)  # a new track in slot 1, given after the result's frame
    visible = np.array([True, True, False, False])
    for _ in range(DRIFT_RESULTS):
        drifted, needs = t.check(2 * S, [5], visible, np.zeros(4, bool), then)
    assert drifted.tolist() == [0]  # slot 0's track drifted; slot 1's new track is not that one
    assert needs == {5: 3}  # the new track counts as good (given after the frame): 1 of 4


def test_an_object_keeps_at_most_twice_its_points_the_longest_hidden_end_first():
    t = Tracks(16, per_object=2, forget_ns=6 * S)
    t.start(np.arange(4), 5, 1 * S)
    t.last_visible[:4] = [4 * S, 2 * S, 3 * S, 1 * S]
    assert t.surplus(5, 0).tolist() == []
    assert sorted(t.surplus(5, 2).tolist()) == [1, 3]


def test_an_object_under_a_new_id_adopts_its_old_tracks():
    """SAM 3 calls object 7 now 9: 7's tracks seen then are orphans, adopted by 9, which then needs no points; tracks
    given by hand, hidden ones and those of objects in the result are no orphans."""
    t = Tracks(8, per_object=4, forget_ns=6 * S)
    t.start(np.array([0, 1, 2, 3]), 7, 10 * S)
    t.start(np.array([4]), 0, 10 * S)
    t.start(np.array([5]), 5, 10 * S)
    seen = np.array([True, True, True, False, True, True, False, False])
    orphans = t.orphans([9, 5], seen, t.track_id.copy())
    assert orphans.tolist() == [0, 1, 2]
    t.adopt(orphans, 9)
    _, needs = t.check(11 * S, [9, 5], seen, np.ones(8, bool), t.track_id.copy())
    assert t.object_id[:4].tolist() == [9, 9, 9, 7] and 9 not in needs


def test_without_free_slots_the_tracks_hidden_longest_make_room():
    t = Tracks(4, per_object=4, forget_ns=6 * S)
    t.start(np.arange(4), 7, 10 * S)
    t.update(np.array([False, True, False, False]), 11 * S)
    t.update(np.array([False, True, True, False]), 12 * S)
    t.last_visible[3] = 10 * S + 1  # hidden a little shorter than slot 0
    assert t.hidden_longest(2).tolist() == [0, 3]
    assert t.hidden_longest(9).tolist() == [0, 3]  # the visible ones never


def test_new_points_lie_inside_the_mask_spread_and_away_from_existing_ones():
    mask = torch.zeros(100, 120, dtype=torch.bool)
    mask[20:80, 30:90] = True
    points = sample(mask, 16, torch.zeros(0, 2), margin=4)
    assert points.shape == (16, 2)
    assert ((points[:, 0] >= 34) & (points[:, 0] < 86) & (points[:, 1] >= 24) & (points[:, 1] < 76)).all()
    gaps = torch.cdist(points, points) + torch.eye(16) * 1e9
    assert gaps.min() > 8  # spread: 16 points in a 52 x 52 square
    have = torch.tensor([[60.0, 50.0]])
    more = sample(mask, 4, have, margin=4)
    assert (more - have).norm(dim=1).min() > 15
    # a mask thinner than the margin: its own pixels
    thin = torch.zeros(10, 10, dtype=torch.bool)
    thin[5, 2:8] = True
    assert len(sample(thin, 3, torch.zeros(0, 2), margin=4)) == 3
    assert not shrink(thin, 1).any() and grow(thin, 1).sum() == 3 * 8


def test_lifting_inverts_the_projection_with_and_without_distortion():
    from xblick import CameraInfo

    for d, model in (([], ""), ([0.1, -0.05, 0.001, 0.002, 0.0], "plumb_bob")):
        info = CameraInfo.pinhole(640, 480, 500.0, 510.0, 319.5, 239.5, frame_id="c")
        info.d, info.distortion_model = d, model
        lifter = Lifter(info)
        import cv2

        world = np.array([[0.1, -0.2, 1.5], [-0.3, 0.1, 2.0], [0.0, 0.0, 1.0]])
        uv = cv2.projectPoints(world, np.zeros(3), np.zeros(3), lifter.k, np.asarray(d, float))[0].reshape(-1, 2)
        assert np.allclose(lifter.lift(uv, world[:, 2]), world, atol=1e-4)
    assert np.isnan(lifter.lift(uv[:1], np.array([0.0]))).all()
    # a depth image of half the size: the nearest of its pixels
    rows, cols, inside = lifter.depth_pixels(np.array([[0.0, 0.0], [639.0, 479.0], [-5.0, 3.0]]), (240, 320))
    assert rows.tolist()[:2] == [0, 240 - 1] and cols.tolist()[:2] == [0, 320 - 1] and inside.tolist() == [True,
                                                                                                         True, False]


def test_requests_parse_and_bad_ones_say_why():
    assert parse({"points": [[1, 2.5]], "stamp": 7}) == {"points": [[1.0, 2.5]], "stamp": 7}
    assert parse({"points": [[1, 2]]})["stamp"] is None
    assert parse({"source": "none"}) == {"source": "none"}
    assert parse({"clear": True}) == {"clear": True}
    for request, message in (({}, "a request is"), ({"points": []}, "non-empty"), ({"points": [[1]]}, "non-empty"),
                             ({"points": [[1, 2]], "stamp": -1}, "nanoseconds"), ({"source": "grid"}, "one of"),
                             ({"clear": False}, "one of them"), ({"points": [[1, 2]], "x": 1}, "only with")):
        with pytest.raises(ValueError, match=message):
            parse(request)


def test_the_cloud_has_the_documented_layout():
    from xblick.points import points_dtype

    xyz = np.array([[0.1, 0.2, 0.3], [np.nan, np.nan, np.nan]])
    points = cloud(xyz, np.array([[1.5, 2.5], [3.0, 4.0]]), np.array([7, 9], np.uint32), np.array([3, 0], np.uint16),
                   np.array([1, 0], np.uint8))
    assert points.shape == (1, 2) and points.dtype == points_dtype(FIELDS, POINT_STEP) and POINT_STEP == 32
    assert points["track_id"].tolist() == [[7, 9]] and points["object_id"].tolist() == [[3, 0]]
    assert np.isnan(points["z"][0, 1]) and points["u"][0, 0] == 1.5 and points["visible"].tolist() == [[1, 0]]
    assert cloud(np.zeros((0, 3)), np.zeros((0, 2)), *(np.zeros(0, t) for t in (np.uint32, np.uint16, np.uint8))
                 ).shape == (1, 0)
