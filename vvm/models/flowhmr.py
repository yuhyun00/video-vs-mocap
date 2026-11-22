"""FlowHMR -> SOMA.

FlowHMR's own pipeline is reused unchanged; the shared subject boxes are
injected through its bbox cache file (``<video>_bbox.npz``), which makes it
skip its own YOLOX tracking. The camera is treated as static (no VGGT).
Native output is SMPL-H (AMASS-style ``poses/betas/trans``), converted with
the shared SMPL-family -> SOMA converter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vvm import weights
from vvm.models.base import PreparedVideo, VideoModel
from vvm.soma import body
from vvm.soma.convert import from_smpl_family
from vvm.soma.motion import SomaMotion


class FlowHMR(VideoModel):
    key = "flowhmr"
    name = "FlowHMR"
    native_format = "SMPL-H (world, 30 fps)"
    repo = "flowhmr"
    static_camera = True

    def run(self, video: PreparedVideo, out_dir: Path, device: str) -> SomaMotion:
        self.link_weights({
            "ckpts/yolox/yolox_l.pth": "flowhmr_yolox_l",
            "ckpts/sam-3d-body-dinov3/model.ckpt": "sam3db_model.ckpt",
            "ckpts/sam-3d-body-dinov3/model_config.yaml": "sam3db_model_config.yaml",
            "ckpts/sam-3d-body-dinov3/assets/mhr_model.pt": "sam3db_assets/mhr_model.pt",
            "assets/body_models/smplh/neutral/model.npz": "smplh",
            "assets/body_models/smpl_neutral_J_regressor.pt": "smplh_regressor",
        })
        self.add_repo_to_path("third_party/YOLOX", "third_party/sam-3d-body")
        from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

        work = out_dir / "work"
        stem = video.path.stem
        sub_dir = work / stem
        sub_dir.mkdir(parents=True, exist_ok=True)
        T = len(video.boxes)
        np.savez(sub_dir / f"{stem}_bbox.npz", bbox=video.boxes.astype(np.float32), start_end=np.array([0, T - 1]))

        ckpt = weights.path("flowhmr_flowhmr_latest/flowhmr_latest.ckpt")
        engine = V2MInferenceEngine.build(str(ckpt), str(ckpt.parent / "config.yml"), device=device)
        engine.process_video(str(video.path), str(work), static_camera=self.static_camera)
        res = np.load(sub_dir / f"{stem}_seed0_origin.npz")
        poses = res["poses"].reshape(len(res["poses"]), -1)  # [T, 52*3]
        conv = from_smpl_family("smplh", str(weights.path("smplh")), poses, res["betas"].reshape(-1), res["trans"],
                                device=device)
        n = len(poses)
        return SomaMotion(source=self.key, times=video.times[:n], joints=conv["joints"],
                          joint_names=body.joint_names(), valid=np.ones(n, bool), identity_model="smplh",
                          frame="world", meta={"smplh_to_soma_error_m": float(conv["fit_error_m"].mean()),
                                               "static_camera": self.static_camera})
