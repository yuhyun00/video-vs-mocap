"""SomaMotion: the single output format of every source (video models and mocap).

Stores SOMA parameters and FK joint positions. Metres, right-handed, Y-up;
``times`` are seconds on the source's own clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SomaMotion:
    source: str  # model key ("gem-x", ...) or "mocap"
    times: np.ndarray  # [T] seconds
    joints: np.ndarray  # [T, J, 3] SOMA public joints, metres, Y-up
    joint_names: list[str]
    valid: np.ndarray  # [T] bool
    poses: np.ndarray | None = None  # [T, 77, 3] SOMA axis-angle (layer convention)
    transl: np.ndarray | None = None  # [T, 3] Hips translation, metres
    identity: np.ndarray | None = None  # [K] identity coefficients (sequence constant)
    scale_params: np.ndarray | None = None  # [S] backend scale parameters
    identity_model: str = "soma"
    frame: str = "world"  # "world" (gravity aligned) or "camera" (camera axes, Y-up)
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- props
    @property
    def num_frames(self) -> int:
        return len(self.times)

    @property
    def fps(self) -> float:
        return float(1.0 / np.median(np.diff(self.times))) if len(self.times) > 1 else 0.0

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def joint(self, name: str) -> np.ndarray:
        return self.joints[:, self.joint_index(name)]

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in ("times", "joints", "valid", "poses", "transl",
                                                  "identity", "scale_params")
                  if getattr(self, k) is not None}
        np.savez_compressed(path, **arrays, joint_names=np.array(self.joint_names),
                            source=self.source, identity_model=self.identity_model, frame=self.frame,
                            meta=json.dumps(self.meta, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SomaMotion":
        d = np.load(path, allow_pickle=False)
        get = lambda k: d[k] if k in d.files else None  # noqa: E731
        return cls(source=str(d["source"]), times=d["times"], joints=d["joints"],
                   joint_names=[str(x) for x in d["joint_names"]], valid=d["valid"].astype(bool),
                   poses=get("poses"), transl=get("transl"), identity=get("identity"),
                   scale_params=get("scale_params"), identity_model=str(d["identity_model"]),
                   frame=str(d["frame"]), meta=json.loads(str(d["meta"])))

    # ------------------------------------------------------------ slicing
    def subset(self, mask: np.ndarray) -> "SomaMotion":
        sl = lambda a: None if a is None else a[mask]  # noqa: E731
        return SomaMotion(self.source, self.times[mask], self.joints[mask], self.joint_names,
                          self.valid[mask], sl(self.poses), sl(self.transl), self.identity,
                          self.scale_params, self.identity_model, self.frame, dict(self.meta))

    def resample(self, times: np.ndarray) -> "SomaMotion":
        """Linear resampling of joint positions onto ``times`` (parameters dropped)."""
        T, J, _ = self.joints.shape
        flat = self.joints.reshape(T, -1)
        ok = self.valid
        out = np.stack([np.interp(times, self.times[ok], flat[ok, k]) for k in range(flat.shape[1])], -1)
        inside = (times >= self.times[ok][0]) & (times <= self.times[ok][-1])
        # a resampled frame is valid only if both neighbours were valid
        idx = np.clip(np.searchsorted(self.times, times), 1, T - 1)
        neigh_ok = ok[idx] & ok[idx - 1]
        return SomaMotion(self.source, times, out.reshape(len(times), J, 3), self.joint_names,
                          inside & neigh_ok, identity_model=self.identity_model, frame=self.frame,
                          meta=dict(self.meta))
