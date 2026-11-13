"""Tracked pixels to points in the camera: each pixel's ray (the colour camera's CameraInfo, its distortion undone)
times the depth at it (a depth image in the colour camera's geometry, e.g. the Azure Kinect's depth_to_color; at its
nearest pixel, no interpolation across edges). Pixels are ROS's (centres at integers); x right, y down, z along the
optical axis, metres (REP 103's optical frame). No depth there: NaN."""

from __future__ import annotations

import numpy as np


class Lifter:
    """The rays of one colour camera model (``key`` tells whether a CameraInfo is still that model)."""

    def __init__(self, info):
        self.key = self.key_of(info)
        self.k = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        self.d = np.asarray(info.d, dtype=np.float64)
        self.model = info.distortion_model
        self.size = (info.height, info.width)
        if self.model not in ("plumb_bob", "rational_polynomial", "equidistant", ""):
            raise ValueError(f"distortion model {self.model!r}: plumb_bob, rational_polynomial or equidistant")

    @staticmethod
    def key_of(info) -> tuple:
        return (tuple(info.k), tuple(info.d), info.distortion_model, info.width, info.height)

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) normalised image coordinates (x / z, y / z)."""
        import cv2

        if len(uv) == 0:
            return np.zeros((0, 2))
        pixels = np.ascontiguousarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == "equidistant":
            return cv2.fisheye.undistortPoints(pixels, self.k, self.d[:4]).reshape(-1, 2)
        if not self.d.size or not self.d.any():
            return np.stack([(uv[:, 0] - self.k[0, 2]) / self.k[0, 0], (uv[:, 1] - self.k[1, 2]) / self.k[1, 1]], 1)
        return cv2.undistortPoints(pixels, self.k, self.d).reshape(-1, 2)

    def depth_pixels(self, uv: np.ndarray, depth_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
        """The depth image's pixel (row, column) nearest each colour pixel (the same camera at another size) and
        whether it is inside."""
        height, width = self.size
        dh, dw = depth_shape
        cols = np.round((uv[:, 0] + 0.5) * dw / width - 0.5).astype(np.int64)
        rows = np.round((uv[:, 1] + 0.5) * dh / height - 0.5).astype(np.int64)
        inside = (cols >= 0) & (cols < dw) & (rows >= 0) & (rows < dh)
        return np.where(inside, rows, 0), np.where(inside, cols, 0), inside

    def lift(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """(N, 2) pixels and their depths (metres; 0 or NaN: none) -> (N, 3), NaN without depth."""
        xy = self.rays(uv)
        z = np.where(z > 0, z, np.nan)
        return np.stack([xy[:, 0] * z, xy[:, 1] * z, z], 1)
