"""Marker trajectories -> SOMA motion (any marker set).

1. Static calibration: fit SOMA identity, bone scales and pose to marker-derived
   joint centres, then attach every labelled marker rigidly to a SOMA segment.
2. Trial fit: per-frame pose and root translation fitted to the markers (bony
   landmarks weighted higher) and to joint centres, with temporal smoothing.
   Joints are solved first with the pelvis fixed, then everything is refined.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vvm.mocap.landmarks import joint_centres, resolve
from vvm.mocap.markers import MarkerData
from vvm.soma import body
from vvm.soma.motion import SomaMotion

# Joints whose rotation is estimated from markers; every other joint (fingers,
# eyes, jaw, toes, end sites) stays at the reference pose.
FREE_JOINTS = ("Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
               "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
               "RightShoulder", "RightArm", "RightForeArm", "RightHand",
               "LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot")

# Segment of known landmarks (anatomy, independent of the lab's labels).
LANDMARK_SEGMENT = {
    "L_ASIS": "Hips", "R_ASIS": "Hips", "L_PSIS": "Hips", "R_PSIS": "Hips", "SACRUM": "Hips",
    "L_KNEE_LAT": "LeftLeg", "L_KNEE_MED": "LeftLeg", "R_KNEE_LAT": "RightLeg", "R_KNEE_MED": "RightLeg",
    "L_ANK_LAT": "LeftShin", "L_ANK_MED": "LeftShin", "R_ANK_LAT": "RightShin", "R_ANK_MED": "RightShin",
    "L_TOE": "LeftFoot", "L_TOE5": "LeftFoot", "L_HEEL": "LeftFoot",
    "R_TOE": "RightFoot", "R_TOE5": "RightFoot", "R_HEEL": "RightFoot",
    "L_SHO": "LeftShoulder", "R_SHO": "RightShoulder",
    "L_ELB_LAT": "LeftArm", "L_ELB_MED": "LeftArm", "R_ELB_LAT": "RightArm", "R_ELB_MED": "RightArm",
    "L_WRIST_A": "LeftForeArm", "L_WRIST_B": "LeftForeArm", "R_WRIST_A": "RightForeArm",
    "R_WRIST_B": "RightForeArm",
    "L_FHEAD": "Head", "R_FHEAD": "Head", "L_BHEAD": "Head", "R_BHEAD": "Head",
    "C7": "Chest", "CLAV": "Chest", "STRN": "Chest", "T10": "Spine2", "BACK_OFFSET": "Chest",
    "L_FINGER": "LeftHand", "R_FINGER": "RightHand",
}

# Bones used to attach unknown markers: (segment that owns the marker, from, to)
ATTACH_BONES = (
    ("Hips", "Hips", "LeftLeg"), ("Hips", "Hips", "RightLeg"), ("Hips", "Hips", "Spine1"),
    ("Spine1", "Spine1", "Spine2"), ("Spine2", "Spine2", "Chest"), ("Chest", "Chest", "Neck1"),
    ("Chest", "Chest", "LeftShoulder"), ("Chest", "Chest", "RightShoulder"), ("Neck1", "Neck1", "Head"),
    ("Head", "Head", "HeadEnd"),
    ("LeftArm", "LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftForeArm", "LeftHand"),
    ("LeftHand", "LeftHand", "LeftHandMiddle2"),
    ("RightArm", "RightArm", "RightForeArm"), ("RightForeArm", "RightForeArm", "RightHand"),
    ("RightHand", "RightHand", "RightHandMiddle2"),
    ("LeftLeg", "LeftLeg", "LeftShin"), ("LeftShin", "LeftShin", "LeftFoot"),
    ("LeftFoot", "LeftFoot", "LeftToeEnd"),
    ("RightLeg", "RightLeg", "RightShin"), ("RightShin", "RightShin", "RightFoot"),
    ("RightFoot", "RightFoot", "RightToeEnd"),
)

# Trial-fit weight of non-landmark (soft-tissue) markers.
SOFT_TISSUE_WEIGHT = 0.3

# Joint-centre targets for the static and trial fits (they also stop the thigh twisting).
JOINT_TARGET_WEIGHTS = {"LeftLeg": 1.0, "RightLeg": 1.0, "LeftShin": 1.0, "RightShin": 1.0,
                         "LeftFoot": 1.0, "RightFoot": 1.0, "LeftToeBase": 0.5, "RightToeBase": 0.5,
                         "LeftArm": 0.5, "RightArm": 0.5, "LeftForeArm": 0.7, "RightForeArm": 0.7,
                         "LeftHand": 0.7, "RightHand": 0.7, "Hips": 0.2}


# --------------------------------------------------------------- rotations
def _aa_to_mat(aa: torch.Tensor) -> torch.Tensor:
    from gem.common.geometry.rotations import axis_angle_to_matrix

    return axis_angle_to_matrix(aa)


def _mat_to_aa(m: torch.Tensor) -> torch.Tensor:
    from gem.common.geometry.rotations import matrix_to_axis_angle

    return matrix_to_axis_angle(m)


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with R @ src + t ~= dst for [N,3] point sets (no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, md - R @ ms


def _frame_from_pelvis(left: np.ndarray, up: np.ndarray) -> np.ndarray:
    left = left / np.linalg.norm(left)
    up = up - left * (up @ left)
    up /= np.linalg.norm(up)
    return np.stack([left, up, np.cross(left, up)], 1)  # columns: left, up, anterior


# ------------------------------------------------------------------- model
class _Fk:
    """Thin wrapper: full SOMA FK returning joints and joint world transforms."""

    def __init__(self, device: str):
        self.device = device
        self.names = body.joint_names()
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.free = torch.tensor([self.idx[n] - 1 for n in FREE_JOINTS], device=device)  # body_pose index

    def __call__(self, go, free_aa, base_pose, ident, scale, transl):
        bp = base_pose.clone() if base_pose.shape[0] == go.shape[0] else base_pose.expand(go.shape[0], -1, -1).clone()
        bp[:, self.free] = free_aa
        out = body.soma_body(self.device)(global_orient=go, body_pose=bp, identity_coeffs=ident.expand(len(go), -1),
                                          scale_params=scale.expand(len(go), -1), transl=transl,
                                          return_joints_only=True)
        return out["joints"], out["transforms"][:, 1:], bp  # drop virtual Root -> public order


@dataclass
class Calibration:
    identity: np.ndarray  # [128]
    scale: np.ndarray  # [57]
    static_pose: np.ndarray  # [76,3] body pose of the static trial (reference for priors)
    markers: list[str]
    segments: list[str]
    offsets: np.ndarray  # [M,3] marker position in its segment frame (m)
    static_residual_mm: float = 0.0
    jc_residual_mm: float = 0.0

    def save(self, path: str | Path) -> None:
        np.savez(path, identity=self.identity, scale=self.scale, static_pose=self.static_pose,
                 offsets=self.offsets, meta=json.dumps({"markers": self.markers, "segments": self.segments,
                                                        "static_residual_mm": self.static_residual_mm,
                                                        "jc_residual_mm": self.jc_residual_mm}))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = np.load(path)
        m = json.loads(str(d["meta"]))
        return cls(d["identity"], d["scale"], d["static_pose"], m["markers"], m["segments"], d["offsets"],
                   m["static_residual_mm"], m["jc_residual_mm"])


def _segment_point_distance(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def calibrate(static: MarkerData, device: str = "cuda", iters: int = 600, log=print) -> Calibration:
    md = static.fill_gaps()
    mid = slice(md.num_frames // 4, max(md.num_frames * 3 // 4, md.num_frames // 4 + 1))
    md_mean = MarkerData(md.names, np.nanmedian(md.xyz[mid], 0, keepdims=True), md.rate, 0, md.source)
    jc = {k: v[0] for k, v in joint_centres(md_mean).items() if np.isfinite(v[0]).all()}
    targets = {k: v for k, v in jc.items() if k in JOINT_TARGET_WEIGHTS}
    if not {"LeftLeg", "RightLeg"} <= targets.keys():
        raise ValueError("static calibration needs pelvis markers (ASIS + PSIS or sacrum); "
                         f"resolved landmarks: {sorted(resolve(static.names))}")
    log(f"[calib] joint-centre targets: {sorted(targets)}")

    fk = _Fk(device)
    dev = dict(device=device, dtype=torch.float32)
    # ---- rest-pose facing -> subject facing (initial global orientation)
    ident0, scale0 = body.neutral_shape(device)
    zero = torch.zeros(1, 3, **dev)
    with torch.no_grad():
        rest_j, _, _ = fk(zero, torch.zeros(1, len(FREE_JOINTS), 3, **dev), torch.zeros(1, 76, 3, **dev), ident0,
                          scale0, zero)
    rj = rest_j[0].cpu().numpy()
    F_rest = _frame_from_pelvis(rj[fk.idx["LeftLeg"]] - rj[fk.idx["RightLeg"]], rj[fk.idx["Spine1"]] - rj[fk.idx["Hips"]])
    hip_mid = (targets["LeftLeg"] + targets["RightLeg"]) / 2
    up_guess = (targets.get("LeftArm", hip_mid + [0, 0.5, 0]) + targets.get("RightArm", hip_mid + [0, 0.5, 0])) / 2 - hip_mid
    F_subj = _frame_from_pelvis(targets["LeftLeg"] - targets["RightLeg"], up_guess)
    R0 = F_subj @ F_rest.T

    go = _mat_to_aa(torch.as_tensor(R0[None], **dev)).clone().requires_grad_(True)
    free = torch.zeros(1, len(FREE_JOINTS), 3, **dev, requires_grad=True)
    transl = torch.as_tensor((hip_mid - R0 @ (rj[fk.idx["LeftLeg"]] + rj[fk.idx["RightLeg"]]) / 2)[None],
                             **dev).requires_grad_(True)
    ident = torch.zeros(body.IDENTITY_DIM, **dev, requires_grad=True)
    log_scale = torch.zeros(body.SCALE_DIM - 1, **dev, requires_grad=True)
    base = torch.zeros(1, 76, 3, **dev)
    names = list(targets)
    tgt = torch.as_tensor(np.stack([targets[n] for n in names]), **dev)
    w = torch.as_tensor([JOINT_TARGET_WEIGHTS[n] for n in names], **dev)
    jidx = torch.tensor([fk.idx[n] for n in names], device=device)

    def scale_vec():
        return torch.cat([torch.ones(1, **dev), log_scale.exp()])

    opt = torch.optim.Adam([{"params": [go, free, transl], "lr": 0.02},
                            {"params": [ident, log_scale], "lr": 0.02}])
    for it in range(iters):
        opt.zero_grad()
        j, _, _ = fk(go, free, base, ident, scale_vec(), transl)
        err = ((j[0, jidx] - tgt) * 100).pow(2).sum(-1)  # cm^2
        loss = (w * err).sum() / w.sum() + 0.05 * ident.pow(2).mean() * 100 + 5.0 * log_scale.pow(2).mean() * 100 \
            + 0.5 * free.pow(2).mean()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == iters - 1:
            log(f"[calib] iter {it:4d}  joint-centre rms {err.mean().sqrt().item() * 10:.1f} mm")

    with torch.no_grad():
        j, T, bp = fk(go, free, base, ident, scale_vec(), transl)
        jc_rms = float(((j[0, jidx] - tgt).pow(2).sum(-1).mean().sqrt()) * 1000)
        joints = j[0].cpu().numpy()
        trans = T[0].cpu().numpy()  # [77,4,4]

    # ---- attach every labelled marker to a segment
    lm_of = {v: k for k, v in resolve(md.names).items()}
    markers, segs, offs = [], [], []
    for mi, name in enumerate(md.names):
        p = md_mean.xyz[0, mi]
        if not np.isfinite(p).all():
            continue
        if name in lm_of and lm_of[name] in LANDMARK_SEGMENT:
            seg = LANDMARK_SEGMENT[lm_of[name]]
        else:
            d = [(_segment_point_distance(p, joints[fk.idx[a]], joints[fk.idx[b]]), s) for s, a, b in ATTACH_BONES]
            dist, seg = min(d)
            if dist > 0.25:
                log(f"[calib] marker {name} is {dist:.2f} m from the body; ignored")
                continue
        M = trans[fk.idx[seg]]
        markers.append(name)
        segs.append(seg)
        offs.append(M[:3, :3].T @ (p - M[:3, 3]))
    calib = Calibration(ident.detach().cpu().numpy(), scale_vec().detach().cpu().numpy(),
                        bp[0].detach().cpu().numpy(), markers, segs, np.array(offs), 0.0, jc_rms)
    log(f"[calib] {len(markers)} markers attached; joint-centre residual {jc_rms:.1f} mm")
    return calib


def fit_trial(trial: MarkerData, calib: Calibration, device: str = "cuda", iters: int = 800,
              smooth: float = 1.0, log=print) -> SomaMotion:
    md = trial.fill_gaps()
    fk = _Fk(device)
    dev = dict(device=device, dtype=torch.float32)
    use = [i for i, m in enumerate(calib.markers) if m in md.names]
    if len(use) < 6:
        raise ValueError(f"only {len(use)} calibrated markers found in the trial")
    names = [calib.markers[i] for i in use]
    segs = [calib.segments[i] for i in use]
    offsets = torch.as_tensor(calib.offsets[use], **dev)
    seg_idx = torch.tensor([fk.idx[s] for s in segs], device=device)
    X = np.stack([md.get(n) for n in names], 1)  # [T,M,3]
    valid = np.isfinite(X).all(-1)
    T = len(X)
    Xt = torch.as_tensor(np.nan_to_num(X), **dev)
    landmarks = set(resolve(names).values())
    weight = torch.tensor([1.0 if n in landmarks else SOFT_TISSUE_WEIGHT for n in names], **dev)
    Vt = torch.as_tensor(valid, **dev) * weight
    jc = {k: v for k, v in joint_centres(md).items() if k in JOINT_TARGET_WEIGHTS}
    jnames = list(jc)
    J = np.stack([jc[n] for n in jnames], 1)  # [T,K,3]
    Jt = torch.as_tensor(np.nan_to_num(J), **dev)
    Jw = torch.as_tensor(np.isfinite(J).all(-1), **dev) * torch.tensor([JOINT_TARGET_WEIGHTS[n] for n in jnames], **dev)
    jidx = torch.tensor([fk.idx[n] for n in jnames], device=device)

    ident = torch.as_tensor(calib.identity, **dev)
    scale = torch.as_tensor(calib.scale, **dev)
    base = torch.as_tensor(calib.static_pose[None], **dev)
    free0 = base[:, fk.free].expand(T, -1, -1).clone()

    # ---- initial root pose per frame: rigid fit of the pelvis markers
    with torch.no_grad():
        _, Ts, _ = fk(torch.zeros(1, 3, **dev), free0[:1], base, ident, scale, torch.zeros(1, 3, **dev))
    Ts = Ts[0].cpu().numpy()
    hip_m = [k for k, s in enumerate(segs) if s == "Hips"]
    pel_local = np.stack([Ts[fk.idx["Hips"], :3, :3] @ calib.offsets[use][k] + Ts[fk.idx["Hips"], :3, 3]
                          for k in hip_m])
    go0, tr0 = np.zeros((T, 3)), np.zeros((T, 3))
    last_R, last_t = np.eye(3), X[valid[:, hip_m].all(1)][0, hip_m].mean(0) if valid[:, hip_m].all(1).any() else 0
    for t in range(T):
        ok = valid[t, hip_m]
        if ok.sum() >= 3:
            last_R, last_t = _kabsch(pel_local[ok], X[t, hip_m][ok])
        go0[t] = _mat_to_aa(torch.as_tensor(last_R[None], dtype=torch.float32))[0].numpy()
        tr0[t] = last_t
    go = torch.as_tensor(go0, **dev).requires_grad_(True)
    transl = torch.as_tensor(tr0, **dev).requires_grad_(True)
    free = free0.clone().requires_grad_(True)

    def optimise(params, n, stage):
        opt = torch.optim.Adam(params, lr=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n)
        for it in range(n):
            opt.zero_grad()
            joints, Tw, _ = fk(go, free, base, ident, scale, transl)
            Tm = Tw[:, seg_idx]  # [T,M,4,4]
            pred = (Tm[..., :3, :3] @ offsets[None, :, :, None])[..., 0] + Tm[..., :3, 3]
            d2 = ((pred - Xt) * 100).pow(2).sum(-1)  # cm^2
            data = (d2 * Vt).sum() / Vt.sum()
            j2 = ((joints[:, jidx] - Jt) * 100).pow(2).sum(-1)
            centres = (j2 * Jw).sum() / Jw.sum().clamp_min(1e-6)
            acc = (free[2:] - 2 * free[1:-1] + free[:-2]).pow(2).mean() + \
                  ((transl[2:] - 2 * transl[1:-1] + transl[:-2]) * 100).pow(2).mean() * 0.01
            prior = (free - base[:, fk.free]).pow(2).mean()
            loss = data + centres + smooth * 100 * acc + 0.1 * prior
            loss.backward()
            opt.step()
            sched.step()
            if it % 100 == 0 or it == n - 1:
                log(f"[fit] {stage} iter {it:4d}  marker rms {data.sqrt().item() * 10:.1f} mm")

    # Pelvis fixed first, so the fit flexes the hips instead of tilting the pelvis (e.g. sitting).
    optimise([free], iters // 2, "joints")
    optimise([go, transl, free], iters - iters // 2, "all")

    with torch.no_grad():
        joints, Tw, bp = fk(go, free, base, ident, scale, transl)
        Tm = Tw[:, seg_idx]
        pred = (Tm[..., :3, :3] @ offsets[None, :, :, None])[..., 0] + Tm[..., :3, 3]
        seen = torch.as_tensor(valid, **dev)  # unweighted, for reporting
        res = ((pred - Xt).norm(dim=-1) * seen).sum(1) / seen.sum(1).clamp_min(1)
        jseen = (Jw > 0).float()
        jres = ((joints[:, jidx] - Jt).norm(dim=-1) * jseen).sum(1) / jseen.sum(1).clamp_min(1)
    frame_ok = valid.sum(1) >= max(6, int(0.6 * len(use)))
    return SomaMotion(
        source="mocap", times=md.times, joints=joints.cpu().numpy(), joint_names=fk.names, valid=frame_ok,
        poses=torch.cat([go[:, None], bp], 1).detach().cpu().numpy(), transl=transl.detach().cpu().numpy(),
        identity=calib.identity, scale_params=calib.scale, identity_model="soma", frame="world",
        meta={"marker_rms_mm": float((res[torch.as_tensor(frame_ok, device=device)].mean()) * 1000),
              "joint_centre_rms_mm": float((jres[torch.as_tensor(frame_ok, device=device)].mean()) * 1000),
              "markers_used": names, "rate": md.rate, "file": md.source})
