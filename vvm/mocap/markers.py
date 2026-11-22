"""Load marker trajectories from C3D (exported by every optical mocap system).

Units come from POINT:UNITS, the up axis is detected from the data, subject
prefixes are stripped and unlabelled markers dropped. Output: metres, Y-up,
NaN for missing samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_UNIT_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}


@dataclass
class MarkerData:
    names: list[str]
    xyz: np.ndarray  # [T, M, 3] metres, Y-up, NaN = missing
    rate: float
    first_frame: int = 0
    source: str = ""

    @property
    def num_frames(self) -> int:
        return self.xyz.shape[0]

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.num_frames) / self.rate

    def get(self, name: str) -> np.ndarray:
        return self.xyz[:, self.names.index(name)]

    def fill_gaps(self, max_gap_s: float = 1.0) -> "MarkerData":
        """Interpolate gaps up to ``max_gap_s`` seconds (longer gaps stay NaN).

        Short occlusions (e.g. ASIS markers hidden while sitting) would otherwise
        leave a segment with too few markers, making its orientation ambiguous.
        """
        max_gap = int(round(max_gap_s * self.rate))
        xyz = self.xyz.copy()
        t = np.arange(self.num_frames)
        for m in range(xyz.shape[1]):
            missing = np.isnan(xyz[:, m, 0])
            if missing.all() or not missing.any():
                continue
            ok = ~missing
            # length of the run each missing sample belongs to
            edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
            run = np.zeros(len(t), int)
            for a, b in zip(edges[::2], edges[1::2]):
                run[a:b] = b - a
            fill = missing & (run <= max_gap) & (t > t[ok][0]) & (t < t[ok][-1])
            for k in range(3):
                xyz[fill, m, k] = np.interp(t[fill], t[ok], xyz[ok, m, k])
        return MarkerData(self.names, xyz, self.rate, self.first_frame, self.source)

    def slice(self, start: int, stop: int) -> "MarkerData":
        return MarkerData(self.names, self.xyz[start:stop], self.rate, self.first_frame + start, self.source)


def _clean_label(label: str) -> str:
    return label.split(":")[-1].strip()


def _is_unlabelled(label: str) -> bool:
    low = label.lower()
    return not label or label.startswith("*") or low.startswith(("unlabeled", "unlabelled", "unnamed"))


def to_y_up(xyz: np.ndarray, up_axis: str | None = None) -> tuple[np.ndarray, str]:
    """Rotate [T,M,3] so the vertical axis becomes +Y.

    ``up_axis`` is e.g. "+z"; None detects it as the axis of largest body spread,
    signed away from the floor.
    """
    if up_axis is None:
        spread = np.nanmedian(np.nanmax(xyz, 1) - np.nanmin(xyz, 1), axis=0)  # [3]
        a = int(np.nanargmax(spread))
        # head is farther from the floor than the feet: compare upper/lower tails
        h = xyz[..., a]
        sign = 1.0 if abs(np.nanpercentile(h, 95)) > abs(np.nanpercentile(h, 5)) else -1.0
        up_axis = ("+" if sign > 0 else "-") + "xyz"[a]
    sign = 1.0 if up_axis[0] == "+" else -1.0
    a = "xyz".index(up_axis[1])
    up = np.zeros(3)
    up[a] = sign
    # pick a forward-ish horizontal axis to complete a right-handed frame
    h1 = np.zeros(3)
    h1[(a + 1) % 3] = 1.0
    h2 = np.cross(h1, up)
    R = np.stack([h1, up, h2])  # rows: new x, new y (=up), new z
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return xyz @ R.T, up_axis


def load_c3d(path: str | Path, up_axis: str | None = None) -> MarkerData:
    import ezc3d

    c = ezc3d.c3d(str(path))
    point = c["parameters"]["POINT"]
    labels = [_clean_label(x) for x in point["LABELS"]["value"]]
    # >255 markers spill into LABELS2, LABELS3, ...
    for k in range(2, 10):
        if f"LABELS{k}" in point:
            labels += [_clean_label(x) for x in point[f"LABELS{k}"]["value"]]
    pts = c["data"]["points"]  # [4, M, T]
    xyz = np.transpose(pts[:3], (2, 1, 0)).astype(np.float64)
    residual = np.transpose(pts[3], (1, 0))
    xyz[residual < 0] = np.nan
    xyz[(xyz == 0).all(-1)] = np.nan
    unit = str(point.get("UNITS", {}).get("value", ["mm"])[0]).strip().lower()
    xyz *= _UNIT_TO_M.get(unit, 1e-3)

    keep = [i for i, n in enumerate(labels[: xyz.shape[1]]) if not _is_unlabelled(n)]
    xyz, _ = to_y_up(xyz[:, keep], up_axis)
    rate = float(point["RATE"]["value"][0])
    first = int(c["header"]["points"]["first_frame"])
    return MarkerData([labels[i] for i in keep], xyz, rate, first, str(path))
