"""Marker-label resolution and anatomical joint-centre estimation.

Labels are resolved through the alias table in ``configs/markersets.yaml``,
so any protocol works as long as its labels are listed there. Joint centres
are estimated with standard regression / midpoint rules and serve as SOMA
joint targets for the static calibration.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import yaml

from vvm.mocap.markers import MarkerData
from vvm.paths import CONFIGS


def _norm(label: str) -> str:
    return re.sub(r"[\s_\-.]", "", label).upper()


@lru_cache(maxsize=1)
def markerset_config() -> dict:
    return yaml.safe_load((CONFIGS / "markersets.yaml").read_text())


def resolve(names: list[str]) -> dict[str, str]:
    """landmark -> actual marker label present in ``names``."""
    lookup = {_norm(n): n for n in names}
    out = {}
    for landmark, aliases in markerset_config()["landmarks"].items():
        for a in aliases:
            if _norm(a) in lookup:
                out[landmark] = lookup[_norm(a)]
                break
    return out


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def joint_centres(md: MarkerData) -> dict[str, np.ndarray]:
    """SOMA joint name -> [T,3] target position (NaN where not computable)."""
    lm = resolve(md.names)
    d = markerset_config()["defaults"]
    g = lambda k: md.get(lm[k]) if k in lm else None  # noqa: E731
    out: dict[str, np.ndarray] = {}

    # ---- pelvis frame and hip joint centres (Harrington et al. 2007) ----
    lasi, rasi, lpsi, rpsi = g("L_ASIS"), g("R_ASIS"), g("L_PSIS"), g("R_PSIS")
    sacr = g("SACRUM")
    if lasi is not None and rasi is not None and (lpsi is not None and rpsi is not None or sacr is not None):
        mid_asis = (lasi + rasi) / 2
        mid_psis = (lpsi + rpsi) / 2 if lpsi is not None and rpsi is not None else sacr
        left = _unit(lasi - rasi)
        fwd = mid_asis - mid_psis
        fwd = _unit(fwd - (fwd * left).sum(-1, keepdims=True) * left)
        up = np.cross(fwd, left)
        pw = np.linalg.norm(lasi - rasi, axis=-1, keepdims=True)
        pd = np.linalg.norm(mid_asis - mid_psis, axis=-1, keepdims=True)
        x = -0.24 * pd - 0.0099
        y = 0.33 * pw + 0.0073
        z = -0.30 * pw - 0.0109
        out["LeftLeg"] = mid_asis + x * fwd + y * left + z * up
        out["RightLeg"] = mid_asis + x * fwd - y * left + z * up
        out["Hips"] = (out["LeftLeg"] + out["RightLeg"]) / 2 + 0.25 * pw * up
        pelvis_left, pelvis_up = left, up
    else:
        pelvis_left = pelvis_up = None

    def pair(lat, med, half, medial_dir):
        a, b = g(lat), g(med)
        if a is not None and b is not None:
            return (a + b) / 2
        if a is not None and medial_dir is not None:
            return a + half * medial_dir
        return None

    for side, s, sgn in (("Left", "L", 1.0), ("Right", "R", -1.0)):
        medial = None if pelvis_left is None else -sgn * pelvis_left
        k = pair(f"{s}_KNEE_LAT", f"{s}_KNEE_MED", d["knee_half_width"], medial)
        a = pair(f"{s}_ANK_LAT", f"{s}_ANK_MED", d["ankle_half_width"], medial)
        if k is not None:
            out[f"{side}Shin"] = k
        if a is not None:
            out[f"{side}Foot"] = a
        toes = [x for x in (g(f"{s}_TOE"), g(f"{s}_TOE5")) if x is not None]
        if toes:
            out[f"{side}ToeBase"] = np.mean(toes, 0)
        sho = g(f"{s}_SHO")
        if sho is not None:
            down = -pelvis_up if pelvis_up is not None else np.array([0.0, -1.0, 0.0])
            out[f"{side}Arm"] = sho + d["shoulder_drop"] * down
        e = pair(f"{s}_ELB_LAT", f"{s}_ELB_MED", d["elbow_half_width"], medial)
        if e is not None:
            out[f"{side}ForeArm"] = e
        wa, wb = g(f"{s}_WRIST_A"), g(f"{s}_WRIST_B")
        if wa is not None and wb is not None:
            out[f"{side}Hand"] = (wa + wb) / 2

    head = [g(k) for k in ("L_FHEAD", "R_FHEAD", "L_BHEAD", "R_BHEAD") if g(k) is not None]
    if len(head) >= 3:
        out["HeadCentre"] = np.mean(head, 0)  # not a SOMA joint; used with a low weight
    return out
