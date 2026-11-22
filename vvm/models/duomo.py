"""DuoMo (Meta) -> SOMA.

DuoMo's preprocessing + two-stage diffusion are called directly with the
shared subject boxes. Its native output is a posed MHR LOD6 mesh (595
vertices) in a gravity-aligned world frame, converted with the shared
MHR-LOD6 -> SOMA converter (no SMPL needed for the conversion).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from vvm import weights
from vvm.models.base import PreparedVideo, VideoModel
from vvm.paths import CHECKPOINTS
from vvm.soma import body
from vvm.soma.convert import from_mhr_lod6_vertices
from vvm.soma.motion import SomaMotion


class DuoMo(VideoModel):
    key = "duomo"
    name = "DuoMo"
    native_format = "MHR LOD6 mesh (world, gravity aligned)"
    repo = "DuoMo"

    def run(self, video: PreparedVideo, out_dir: Path, device: str) -> SomaMotion:
        d = CHECKPOINTS / "duomo"
        links = {f"data/checkpoints/{n}": d / "checkpoints" / n
                 for n in ("converter", "dense_kpt", "duomo_cam", "duomo_world")}
        links.update({
            "data/third_party/phmr": d / "third_party" / "phmr",
            "data/third_party/camcalib_sa_biased_l2.ckpt": "duomo_camcalib",
            "data/third_party/vitpose-h-multi-coco.pth": "duomo_vitpose",
            "data/body_models/smpl": CHECKPOINTS / "body_models" / "smpl",
            "data/body_models/smplx": CHECKPOINTS / "body_models" / "smplx",
        })
        for f in (d / "body_models").glob("*"):  # smplx2smpl.pkl, J_regressor_h36m.npy, ...
            links[f"data/body_models/{f.name}"] = f
        self.link_weights(links)
        # DuoMo's extra SMPL-X array file lives next to the licensed SMPL-X models
        smplx_dir = CHECKPOINTS / "body_models" / "smplx"
        for f in (d / "supplementary_smplx").glob("*.npz"):
            if smplx_dir.is_dir() and not (smplx_dir / f.name).exists():
                (smplx_dir / f.name).symlink_to(f)

        root = self.add_repo_to_path("third_party/PromptHMR")
        os.chdir(root)  # PromptHMR resolves "data/body_models/..." relative to the working directory
        from src.inference import MotionDiffusionPipeline

        pipe = MotionDiffusionPipeline(str(root / "configs" / "inference.yaml"))
        work = out_dir / "work"
        work.mkdir(exist_ok=True)
        boxes = torch.as_tensor(video.boxes, dtype=torch.float32)
        camera, dkp, imgfeat, valid = pipe.run_preprocess(str(video.path), str(work), boxes=boxes)
        results = pipe.run_duomo(camera, dkp, imgfeat, valid, cam_gravity=True)
        verts = results["pred_lod6_world"].float().cpu().numpy()
        np.save(out_dir / "native_lod6_world.npy", verts)
        conv = from_mhr_lod6_vertices(verts, device=device)
        n = len(verts)
        return SomaMotion(source=self.key, times=video.times[:n], joints=conv["joints"],
                          joint_names=body.joint_names(), valid=valid.cpu().numpy()[:n].astype(bool),
                          identity_model="mhr", frame="world",
                          meta={"lod6_to_soma_error_cm": float(conv["fit_error_cm"].mean())})
