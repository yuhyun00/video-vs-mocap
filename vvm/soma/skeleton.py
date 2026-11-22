"""SOMA skeleton vocabulary used by evaluation, synchronisation and drawing.

Joint names follow the SOMA public skeleton (77 joints, no Root). Only the
names are hard-coded here; indices are always looked up in the motion's own
``joint_names`` so the code stays valid if the layer adds joints.
"""

from __future__ import annotations

# Body joints compared between sources (fingers, eyes, jaw and end sites are
# excluded: mocap does not observe them and video models are unreliable there).
EVAL_JOINTS: tuple[str, ...] = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head",
    "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)

LOWER_BODY_JOINTS: tuple[str, ...] = (
    "Hips", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)

# Bones drawn in skeleton renders (parent, child) over EVAL_JOINTS.
BONES: tuple[tuple[str, str], ...] = (
    ("Hips", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Chest"), ("Chest", "Neck1"), ("Neck1", "Head"),
    ("Chest", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Chest", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftLeg"), ("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightLeg"), ("RightLeg", "RightShin"), ("RightShin", "RightFoot"), ("RightFoot", "RightToeBase"),
)

# SOMA joint -> anatomy: LeftLeg = hip, LeftShin = knee, LeftFoot = ankle,
# LeftArm = shoulder, LeftForeArm = elbow, LeftHand = wrist.
ANATOMY = {
    "LeftLeg": "L hip", "RightLeg": "R hip", "LeftShin": "L knee", "RightShin": "R knee",
    "LeftFoot": "L ankle", "RightFoot": "R ankle", "LeftToeBase": "L toe", "RightToeBase": "R toe",
    "LeftArm": "L shoulder", "RightArm": "R shoulder", "LeftForeArm": "L elbow",
    "RightForeArm": "R elbow", "LeftHand": "L wrist", "RightHand": "R wrist",
}


def indices(joint_names, subset=EVAL_JOINTS) -> list[int]:
    return [list(joint_names).index(n) for n in subset]
