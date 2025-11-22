from __future__ import annotations

import numpy as np


class FrenetFrame:
    """Frenet frame defined by a polyline centerline.
    由折线中心线定义的 Frenet 坐标系。

    Parameters
    ----------
    centerline_xy : array-like, shape (N,2)
        Polyline points in driving direction order.
        按行驶方向排序的折线点集。
    """

    def __init__(self, centerline_xy: np.ndarray) -> None:
        centerline_xy = np.asarray(centerline_xy, dtype=float)
        if centerline_xy.ndim != 2 or centerline_xy.shape[1] != 2:
            raise ValueError("centerline_xy must have shape (N, 2)")
        if len(centerline_xy) < 2:
            raise ValueError("centerline_xy must contain at least two points")

        diffs = np.diff(centerline_xy, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        if np.any(seg_len == 0):
            raise ValueError("centerline contains duplicate consecutive points")

        self.centerline = centerline_xy
        self.seg_len = seg_len
        self.s_seg = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.tangents = diffs / seg_len[:, None]
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)

    def _project_point(self, p: np.ndarray) -> tuple[float, float, int]:
        best_dist2 = np.inf
        best_s = 0.0
        best_n = 0.0
        best_seg_idx = 0

        for k, seg_len in enumerate(self.seg_len):
            p0 = self.centerline[k]
            t_vec = self.tangents[k]
            w = p - p0
            proj_along = float(np.dot(w, t_vec))
            proj_along_clamped = float(np.clip(proj_along, 0.0, seg_len))
            proj = p0 + proj_along_clamped * t_vec
            n = float(np.dot(p - proj, self.normals[k]))
            s = float(self.s_seg[k] + proj_along_clamped)
            dist2 = float(np.sum((p - proj) ** 2))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_s, best_n, best_seg_idx = s, n, k

        return best_s, best_n, best_seg_idx

    def to_frenet(
        self,
        xy: np.ndarray,
        v_xy: np.ndarray | None = None,
        a_xy: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Project world-frame states into the Frenet frame.

        Velocity and acceleration components follow the stored tangent
        orientation (i.e., positive ``v_s``/``a_s`` align with the polyline
        direction). Callers may flip the sign externally to ensure forward
        motion stays positive regardless of dataset drivingDirection metadata.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy[None, :]
        if xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")

        N = xy.shape[0]
        s = np.empty(N, dtype=float)
        n = np.empty(N, dtype=float)
        seg_idx = np.empty(N, dtype=int)

        for i in range(N):
            s[i], n[i], seg_idx[i] = self._project_point(xy[i])

        result: dict[str, np.ndarray] = {"s": s, "n": n, "seg_idx": seg_idx}

        if v_xy is not None:
            v_xy = np.asarray(v_xy, dtype=float)
            if v_xy.shape != xy.shape:
                raise ValueError("v_xy must match xy shape")
            v_s = np.empty(N, dtype=float)
            v_n = np.empty(N, dtype=float)
            for i in range(N):
                t_vec = self.tangents[seg_idx[i]]
                n_vec = self.normals[seg_idx[i]]
                v_s[i] = float(np.dot(v_xy[i], t_vec))
                v_n[i] = float(np.dot(v_xy[i], n_vec))
            result.update({"v_s": v_s, "v_n": v_n})

        if a_xy is not None:
            a_xy = np.asarray(a_xy, dtype=float)
            if a_xy.shape != xy.shape:
                raise ValueError("a_xy must match xy shape")
            a_s = np.empty(N, dtype=float)
            a_n = np.empty(N, dtype=float)
            for i in range(N):
                t_vec = self.tangents[seg_idx[i]]
                n_vec = self.normals[seg_idx[i]]
                a_s[i] = float(np.dot(a_xy[i], t_vec))
                a_n[i] = float(np.dot(a_xy[i], n_vec))
            result.update({"a_s": a_s, "a_n": a_n})

        return result
