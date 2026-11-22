"""The SOMA body model used everywhere: GEM-X's ``SomaFixedGs1BodyAdapter`` (SOMA-X 0.3.3).

Differentiable, and skips skinning when only joints are needed. Parameters:
global_orient [T,3], body_pose [T,76,3] (axis-angle), identity_coeffs [128],
scale_params [57], transl [T,3] metres.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from vvm import weights

NUM_BODY_POSE = 76
IDENTITY_DIM = 128
SCALE_DIM = 57


@lru_cache(maxsize=4)
def soma_body(device: str = "cuda", lod: str = "mid", apply_correctives: bool = False):
    from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig

    cfg = SomaRuntimeConfig(data_root=str(weights.path("soma_assets")), device=torch.device(device), lod=lod,
                            apply_correctives=apply_correctives, enable_procedural_transforms=True,
                            procedural_reference="runtime")
    return SomaFixedGs1BodyAdapter(cfg).to(device)


@lru_cache(maxsize=1)
def joint_names() -> list[str]:
    from soma import SOMALayer
    from soma.units import Unit

    layer = SOMALayer(identity_model_type="soma", data_root=str(weights.path("soma_assets")), device="cpu",
                      output_unit=Unit.METERS)
    # public_joint_names includes the virtual "Root"; FK joints (77) do not.
    return [n for n in layer.public_joint_names if n != "Root"]


def neutral_shape(device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(IDENTITY_DIM, device=device), torch.ones(SCALE_DIM, device=device)


def forward(global_orient, body_pose, identity, scale, transl, *, device="cuda", joints_only=True,
            chunk: int = 512, grad: bool = False) -> dict[str, torch.Tensor]:
    """Batched SOMA FK. Inputs are tensors or numpy arrays; ``grad=True`` keeps autograd."""
    with torch.set_grad_enabled(grad):
        return _forward(global_orient, body_pose, identity, scale, transl, device, joints_only, chunk)


def _forward(global_orient, body_pose, identity, scale, transl, device, joints_only, chunk):
    body = soma_body(device)
    t = lambda x: torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x,  # noqa: E731
                                  dtype=torch.float32, device=device)
    go, bp, tr = t(global_orient), t(body_pose).reshape(-1, NUM_BODY_POSE, 3), t(transl)
    ident, sc = t(identity), t(scale)
    outs = []
    for s in range(0, len(go), chunk):
        e = s + chunk
        n = len(go[s:e])
        out = body(global_orient=go[s:e], body_pose=bp[s:e],
                   identity_coeffs=ident.expand(n, -1) if ident.ndim == 1 else ident[s:e],
                   scale_params=sc.expand(n, -1) if sc.ndim == 1 else sc[s:e],
                   transl=tr[s:e], return_joints_only=joints_only)
        outs.append(out)
    keys = [k for k in ("joints", "vertices") if outs[0].get(k) is not None]
    return {k: torch.cat([o[k] for o in outs]) for k in keys}


def faces() -> np.ndarray:
    return soma_body("cpu").runtime.faces.detach().cpu().numpy()
