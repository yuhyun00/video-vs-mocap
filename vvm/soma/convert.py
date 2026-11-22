"""Conversions from other body models to SOMA, shared by all adapters.

Follows SOMA-X's reference converters: pose the source model, then invert its
vertices onto a SOMA layer that uses the source model as identity backend.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from vvm import weights


def _fp32(fn):
    """Run without autocast/grad: some upstream code (SAM 3) enables bf16 autocast process-wide."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad(), torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            return fn(*args, **kwargs)

    return wrapper


def _soma_layer(identity_model_type: str, device: str, unit: str, **kwargs):
    from soma import SOMALayer
    from soma.units import Unit

    return SOMALayer(str(weights.path("soma_assets")), identity_model_type=identity_model_type, device=device,
                     mode="warp", output_unit=Unit.CENTIMETERS if unit == "cm" else Unit.METERS, **kwargs)


def _fk_joints(layer, rotations: torch.Tensor, root_translation: torch.Tensor) -> torch.Tensor:
    """FK of PoseInversion output on the identity already prepared on ``layer``.

    PoseInversion returns joint-orient-inclusive ("absolute") local rotations for
    the full rig, virtual Root first; ``pose`` takes the 77 public joints.
    """
    rot = rotations[:, 1:] if rotations.shape[1] == 78 else rotations
    out = layer.pose(rot, transl=root_translation, pose2rot=False, absolute_pose=True, fk_only=True)
    return out["joints"]


@_fp32
def from_mhr(shape_params: np.ndarray, model_params: np.ndarray, device: str = "cuda",
             batch_size: int = 64) -> dict[str, np.ndarray]:
    """MHR -> SOMA.

    Args:
        shape_params: [T,45] MHR identity coefficients.
        model_params: [T,204] MHR model parameters (translation cm, pose, scales).
    Returns dict with ``joints`` [T,77,3] metres (MHR world frame, Y-up),
    ``rotations`` [T,J,3,3] SOMA local rotations, ``root_translation`` [T,3] m,
    ``fit_error_cm`` [T] mean vertex error of the inversion.
    """
    from soma.fitting.pose_inversion import PoseInversion

    mhr_jit = torch.jit.load(str(weights.path("soma_assets") / "MHR" / "mhr_model_lod1.pt"), map_location=device)
    layer = _soma_layer("mhr", device, "cm")
    # low_lod=True would build an internal low-LOD layer, which hits a broken relative
    # import in py-soma-x 0.3.3; refitting on the mid LOD is equivalent, only slower.
    inv = PoseInversion(layer, low_lod=False)
    ic_all = torch.as_tensor(shape_params, dtype=torch.float32, device=device)
    mp_all = torch.as_tensor(model_params, dtype=torch.float32, device=device)
    face = torch.zeros(1, 72, device=device)
    joints, rots, transl, err = [], [], [], []
    for s in range(0, len(ic_all), batch_size):
        ic, mp = ic_all[s:s + batch_size], mp_all[s:s + batch_size]
        scale, flex = mp[:, 136:], mp[:, 130:136]
        inv.prepare_identity(ic, scale, kwargs={"bone_length_flexibles": flex})
        verts_cm, _ = mhr_jit(ic, mp, face.expand(len(ic), -1))
        res = inv.fit(verts_cm)
        layer.prepare_identity(ic, scale, kwargs={"bone_length_flexibles": flex})
        j = _fk_joints(layer, res["rotations"], res["root_translation"])
        joints.append(j.cpu() / 100.0)
        rots.append(res["rotations"].cpu())
        transl.append(res["root_translation"].cpu() / 100.0)
        err.append(res["per_vertex_error"].mean(-1).cpu())
    return {"joints": torch.cat(joints).numpy(), "rotations": torch.cat(rots).numpy(),
            "root_translation": torch.cat(transl).numpy(), "fit_error_cm": torch.cat(err).numpy()}


@_fp32
def from_mhr_lod6_vertices(verts_m: np.ndarray, device: str = "cuda", batch_size: int = 128) -> dict[str, np.ndarray]:
    """Posed MHR LOD6 meshes [T,595,3] (metres) -> SOMA, used for DuoMo.

    Without MHR parameters the mean identity is used: joint angles are recovered,
    bone lengths follow the mean body.
    """
    from soma.fitting.pose_inversion import PoseInversion

    layer = _soma_layer("mhr", device, "cm", lod="low")
    inv = PoseInversion(layer, low_lod=True)
    ident = torch.zeros(1, 45, device=device)
    scale = torch.zeros(1, 68, device=device)
    inv.prepare_identity(ident, scale)
    layer.prepare_identity(ident, scale)
    V = torch.as_tensor(verts_m, dtype=torch.float32, device=device) * 100.0  # -> cm
    joints, err = [], []
    for s in range(0, len(V), batch_size):
        res = inv.fit(V[s:s + batch_size])
        joints.append(_fk_joints(layer, res["rotations"], res["root_translation"]).cpu() / 100.0)
        err.append(res["per_vertex_error"].mean(-1).cpu())
    return {"joints": torch.cat(joints).numpy(), "fit_error_cm": torch.cat(err).numpy()}


@_fp32
def from_smpl_family(model_type: str, model_path: str, poses: np.ndarray, betas: np.ndarray, trans: np.ndarray,
                     device: str = "cuda", batch_size: int = 128) -> dict[str, np.ndarray]:
    """SMPL / SMPL-H / SMPL-X axis-angle parameters -> SOMA.

    Args:
        model_type: "smpl", "smplh" or "smplx" (SOMA identity backend name).
        model_path: licensed model file (e.g. SMPLH neutral model.npz).
        poses: [T, J*3] full axis-angle pose including global orient.
        betas: [B] or [T,B] shape coefficients.
        trans: [T,3] metres.
    """
    from soma.fitting.pose_inversion import PoseInversion

    layer = _soma_layer(model_type, device, "m", identity_model_kwargs={"model_path": model_path})
    inv = PoseInversion(layer, low_lod=False)  # see from_mhr
    smpl = layer.identity_model  # SOMA's SMPL-family backend exposes the source model
    T = len(poses)
    betas = np.broadcast_to(np.atleast_2d(betas), (T, np.atleast_2d(betas).shape[-1]))
    joints, rots, transl, err = [], [], [], []
    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        b = torch.as_tensor(betas[s:e], dtype=torch.float32, device=device)
        p = torch.as_tensor(poses[s:e], dtype=torch.float32, device=device)
        t = torch.as_tensor(trans[s:e], dtype=torch.float32, device=device)
        verts = smpl.forward_native(betas=b, poses=p, transl=t)  # [N,V,3] metres
        inv.prepare_identity(b)
        res = inv.fit(verts)
        layer.prepare_identity(b)
        joints.append(_fk_joints(layer, res["rotations"], res["root_translation"]).cpu())
        rots.append(res["rotations"].cpu())
        transl.append(res["root_translation"].cpu())
        err.append(res["per_vertex_error"].mean(-1).cpu())
    return {"joints": torch.cat(joints).numpy(), "rotations": torch.cat(rots).numpy(),
            "root_translation": torch.cat(transl).numpy(), "fit_error_m": torch.cat(err).numpy()}
