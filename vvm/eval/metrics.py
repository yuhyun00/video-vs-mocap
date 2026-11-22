"""Quantitative comparison between a video-model motion and the mocap reference.

Both inputs must already be on the same time grid (see ``vvm.sync``) and use
SOMA joints. Positions are compared on ``EVAL_JOINTS``; angles use
``vvm.eval.angles``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vvm.eval.angles import ANGLE_NAMES, angle_matrix
from vvm.soma.skeleton import EVAL_JOINTS, indices


# ----------------------------------------------------------------- alignment
def similarity_transform(src: np.ndarray, dst: np.ndarray, scale: bool = True):
    """Umeyama: returns (s, R, t) minimising |s R src + t - dst| for [N,3] point sets."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / max(xs.var(0).sum(), 1e-12) if scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply(s, R, t, x: np.ndarray) -> np.ndarray:
    return s * x @ R.T + t


def align_sequence(pred: np.ndarray, ref: np.ndarray, scale: bool = False) -> np.ndarray:
    """One rigid (optionally similarity) transform for the whole sequence [T,J,3]."""
    s, R, t = similarity_transform(pred.reshape(-1, 3), ref.reshape(-1, 3), scale)
    return apply(s, R, t, pred)


def align_per_frame(pred: np.ndarray, ref: np.ndarray, scale: bool = True) -> np.ndarray:
    out = np.empty_like(pred)
    for i in range(len(pred)):
        s, R, t = similarity_transform(pred[i], ref[i], scale)
        out[i] = apply(s, R, t, pred[i])
    return out


# ------------------------------------------------------------------- results
@dataclass
class Evaluation:
    model: str
    num_frames: int
    position: dict[str, float] = field(default_factory=dict)  # summary, mm
    per_joint: dict[str, float] = field(default_factory=dict)  # MPJPE (root aligned) per joint, mm
    angles: dict[str, dict[str, float]] = field(default_factory=dict)  # angle -> {rmse, mae, bias, r}
    smoothness: dict[str, float] = field(default_factory=dict)
    # time series kept for plotting
    series: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def summary_row(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {"model": self.model, "frames": self.num_frames, **self.position}
        lower = [v for k, v in self.angles.items() if "elbow" not in k]
        if lower:
            row["lower-limb angle RMSE (deg)"] = float(np.mean([a["rmse"] for a in lower]))
            row["lower-limb angle r"] = float(np.mean([a["r"] for a in lower]))
        row.update(self.smoothness)
        return row


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(pred_joints: np.ndarray, ref_joints: np.ndarray, joint_names, model: str,
             fps: float, times: np.ndarray | None = None) -> Evaluation:
    """pred/ref: [T, J, 3] on the same timestamps (invalid frames removed).

    ``times`` marks where frames were removed, so accelerations are only taken
    over three consecutive frames, never across a gap.
    """
    idx = indices(joint_names, EVAL_JOINTS)
    hips = list(joint_names).index("Hips")
    P, G = pred_joints[:, idx], ref_joints[:, idx]
    T = len(P)
    ev = Evaluation(model, T)

    root_p = P - pred_joints[:, hips:hips + 1]
    root_g = G - ref_joints[:, hips:hips + 1]
    # Root-aligned error still depends on each source's world heading; remove a
    # single sequence rotation first so it measures pose, not facing direction.
    root_p = align_sequence(root_p, root_g, scale=False)
    err_root = np.linalg.norm(root_p - root_g, axis=-1) * 1000
    err_pa = np.linalg.norm(align_per_frame(P, G, True) - G, axis=-1) * 1000
    err_w = np.linalg.norm(align_sequence(P, G, False) - G, axis=-1) * 1000
    ev.position = {"MPJPE (mm)": float(err_root.mean()), "PA-MPJPE (mm)": float(err_pa.mean()),
                   "W-MPJPE (mm)": float(err_w.mean())}
    ev.per_joint = {n: float(e) for n, e in zip(EVAL_JOINTS, err_root.mean(0))}

    ang_p = angle_matrix(pred_joints, joint_names)
    ang_g = angle_matrix(ref_joints, joint_names)
    for k, name in enumerate(ANGLE_NAMES):
        d = ang_p[:, k] - ang_g[:, k]
        ev.angles[name] = {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.abs(d).mean()),
                           "bias": float(d.mean()), "rmse_debiased": float(d.std()),
                           "r": _pearson(ang_p[:, k], ang_g[:, k]),
                           "rom_pred": float(np.ptp(ang_p[:, k])), "rom_ref": float(np.ptp(ang_g[:, k]))}

    step = np.ones(max(T - 1, 0), bool) if times is None else np.isclose(np.diff(times), 1 / fps, rtol=0.25)
    triple = step[1:] & step[:-1]
    if triple.any():
        acc_p = (np.diff(P, 2, axis=0) * fps ** 2)[triple]
        acc_g = (np.diff(G, 2, axis=0) * fps ** 2)[triple]
        ev.smoothness = {"accel error (m/s^2)": float(np.linalg.norm(acc_p - acc_g, axis=-1).mean()),
                         "jitter pred (m/s^2)": float(np.linalg.norm(acc_p, axis=-1).mean()),
                         "jitter ref (m/s^2)": float(np.linalg.norm(acc_g, axis=-1).mean())}
    ev.series = {"err_root": err_root, "err_pa": err_pa, "angles_pred": ang_p, "angles_ref": ang_g}
    return ev
