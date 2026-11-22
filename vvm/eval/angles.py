"""Anatomical joint angles computed from SOMA joint positions.

Angles are derived from positions only, so they are independent of each
source's world frame and of SOMA's internal joint-orient conventions; the same
function is applied to mocap and to every video model.

Sign conventions (degrees): hip flexion +, hip abduction +, knee flexion +
(0 = straight), ankle dorsiflexion +, elbow flexion + (0 = straight).
"""

from __future__ import annotations

import numpy as np

ANGLE_NAMES: tuple[str, ...] = (
    "L hip flexion", "R hip flexion", "L hip abduction", "R hip abduction",
    "L knee flexion", "R knee flexion", "L ankle dorsiflexion", "R ankle dorsiflexion",
    "L elbow flexion", "R elbow flexion",
)
LOWER_LIMB_ANGLES = ANGLE_NAMES[:8]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1.0, 1.0)))


def pelvis_frame(joints: np.ndarray, names) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(left, up, anterior) unit axes [T,3] from hip centres and lower spine."""
    j = lambda n: joints[:, list(names).index(n)]  # noqa: E731
    left = _unit(j("LeftLeg") - j("RightLeg"))
    up = j("Spine1") - j("Hips")
    up = _unit(up - (up * left).sum(-1, keepdims=True) * left)
    anterior = np.cross(left, up)
    return left, up, anterior


def joint_angles(joints: np.ndarray, names) -> dict[str, np.ndarray]:
    """Dict of angle name -> [T] degrees."""
    j = lambda n: joints[:, list(names).index(n)]  # noqa: E731
    left, up, ant = pelvis_frame(joints, names)
    out: dict[str, np.ndarray] = {}
    for side, s, sign in (("Left", "L", 1.0), ("Right", "R", -1.0)):
        thigh = j(f"{side}Shin") - j(f"{side}Leg")
        shank = j(f"{side}Foot") - j(f"{side}Shin")
        foot = j(f"{side}ToeBase") - j(f"{side}Foot")
        down = -(thigh * up).sum(-1)
        # flexion: thigh angle in the sagittal plane; abduction: thigh angle out of the
        # sagittal plane (arcsin stays well-conditioned at 90 deg flexion, e.g. sitting)
        out[f"{s} hip flexion"] = np.degrees(np.arctan2((thigh * ant).sum(-1), down))
        out[f"{s} hip abduction"] = np.degrees(np.arcsin(np.clip(sign * (_unit(thigh) * left).sum(-1), -1, 1)))
        out[f"{s} knee flexion"] = _between(thigh, shank)
        out[f"{s} ankle dorsiflexion"] = 90.0 - _between(-shank, foot)
        upper = j(f"{side}ForeArm") - j(f"{side}Arm")
        fore = j(f"{side}Hand") - j(f"{side}ForeArm")
        out[f"{s} elbow flexion"] = _between(upper, fore)
    return {k: out[k] for k in ANGLE_NAMES}


def angle_matrix(joints: np.ndarray, names, which=ANGLE_NAMES) -> np.ndarray:
    a = joint_angles(joints, names)
    return np.stack([a[k] for k in which], -1)
