"""Manifest of every weight / asset file the project needs.

The download script and every model adapter read this one table, so a file is
declared exactly once: where it comes from, where it lives under
``checkpoints/``, which models need it and whether the user must fetch it by
hand (licence-gated files).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vvm.paths import BODY_MODELS, CHECKPOINTS


@dataclass(frozen=True)
class Weight:
    key: str
    dest: str  # path relative to CHECKPOINTS (file or directory)
    source: str  # "hf", "hf_snapshot", "url", "url_zip_member", "github_tar", "gdrive_file", "gdrive_folder", "manual"
    models: tuple[str, ...]
    repo: str = ""  # HF repo id, URL or Google-Drive id depending on source
    filename: str = ""  # file inside the HF repo / zip member suffix
    revision: str | None = None
    gated: bool = False  # HF repo requires access approval + login
    note: str = ""
    allow_patterns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def path(self) -> Path:
        return CHECKPOINTS / self.dest

    def exists(self) -> bool:
        p = self.path
        if p.is_dir():
            return any(p.iterdir())
        return p.is_file() and p.stat().st_size > 0


_GEMX = ("gem-x",)
_S3DB = ("sam3d-body",)
_B4D = ("sam-body4d",)
_DUOMO = ("duomo",)
_FLOW = ("flowhmr",)
_ALL = ("gem-x", "sam3d-body", "sam-body4d", "duomo", "flowhmr", "mocap")

WEIGHTS: tuple[Weight, ...] = (
    # --- SOMA geometry (shared by every converter and the mocap fitter) ---
    Weight("soma_assets", "soma", "hf_snapshot", _ALL, repo="nvidia/SOMA-X", revision="v0.3.3",
           allow_patterns=("SOMA_neutral.npz", "SOMA_template_rig.usda", "SOMA_procedural_transforms.json",
                           "correctives_model.pt", "SOMAHand.npz", "MHR/*", "SMPL/*", "SMPLX/*", "LICENSE")),
    # --- shared person detector (also used by GEM-X) ---
    Weight("yolox_humanart", "detector/yolox_humanart.onnx", "url_zip_member", _ALL,
           repo="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                "yolox_x_8xb8-300e_humanart-a39d44ed.zip",
           filename="end2end.onnx"),
    # --- GEM-X ---
    *(Weight(f"gemx_{f}", f"gemx/{f}", "hf", _GEMX, repo="nvidia/GEM-X", filename=f)
      for f in ("body_s240000.ckpt", "body_config.json", "vitpose.pth",
                "motion_native77_e774_s292175.ckpt", "motion_stats_raw.npz", "motion_native77.yaml")),
    # --- SAM 3D Body (also used by SAM-Body4D and FlowHMR) ---
    *(Weight(f"sam3db_{f}", f"sam-3d-body-dinov3/{f}", "hf", _S3DB + _B4D + _FLOW,
             repo="facebook/sam-3d-body-dinov3", filename=f, gated=True)
      for f in ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt")),
    Weight("moge2", "moge-2-vitl-normal/model.pt", "hf", _S3DB + _B4D,
           repo="Ruicheng/moge-2-vitl-normal", filename="model.pt"),
    # --- SAM-Body4D extras ---
    Weight("sam3", "sam3/sam3.pt", "hf", _B4D, repo="facebook/sam3", filename="sam3.pt", gated=True),
    Weight("vas_amodal", "diffusion-vas-amodal-segmentation", "hf_snapshot", _B4D,
           repo="kaihuac/diffusion-vas-amodal-segmentation"),
    Weight("vas_completion", "diffusion-vas-content-completion", "hf_snapshot", _B4D,
           repo="kaihuac/diffusion-vas-content-completion"),
    Weight("depth_anything_v2", "depth_anything_v2_vitl.pth", "hf", _B4D,
           repo="depth-anything/Depth-Anything-V2-Large", filename="depth_anything_v2_vitl.pth"),
    # --- FlowHMR ---
    *(Weight(f"flowhmr_{f}", f"flowhmr/{f}", "hf", _FLOW, repo="fafsaf1/flowhmr-0.46B", filename=f)
      for f in ("config.json", "flowhmr_latest/config.yml", "flowhmr_latest/flowhmr_latest.ckpt")),
    Weight("flowhmr_yolox_l", "flowhmr/yolox/yolox_l.pth", "url", _FLOW,
           repo="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"),
    Weight("smplh_regressor", "body_models/smpl_neutral_J_regressor.pt", "url", _FLOW,
           repo="https://raw.githubusercontent.com/zju3dv/GVHMR/main/hmr4d/utils/body_model/"
                "smpl_neutral_J_regressor.pt"),
    # --- DuoMo ---
    *(Weight(f"duomo_{n}", f"duomo/checkpoints/{n}", "github_tar", _DUOMO,
             repo=f"https://github.com/facebookresearch/DuoMo/releases/download/v1/{n}.tar.gz")
      for n in ("converter", "dense_kpt", "duomo_cam", "duomo_world")),
    # Google-Drive files (DuoMo's download scripts); YOLO / keypoint-RCNN / SAM2 are not
    # needed because the subject boxes are supplied.
    Weight("duomo_phmr_ckpt", "duomo/third_party/phmr/checkpoint.ckpt", "gdrive_file", _DUOMO,
           repo="1uQwMCkkqtyQIwuBeWwE83XcOlISKbTHB"),
    Weight("duomo_phmr_cfg", "duomo/third_party/phmr/config.yaml", "gdrive_file", _DUOMO,
           repo="1P3EEmBDeRRORhBkUhwPZclZxd7U3Z-yo"),
    Weight("duomo_camcalib", "duomo/third_party/camcalib_sa_biased_l2.ckpt", "gdrive_file", _DUOMO,
           repo="1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO"),
    # same ViTPose-H file as DuoMo's Drive link, from the public GVHMR checkpoint mirror
    Weight("duomo_vitpose", "duomo/third_party/vitpose-h-multi-coco.pth", "hf", _DUOMO,
           repo="camenduru/GVHMR", filename="vitpose/vitpose-h-multi-coco.pth"),
    *(Weight(f"duomo_bm_{n}", f"duomo/body_models/{n}", "gdrive_file", _DUOMO, repo=i)
      for n, i in (("J_regressor_h36m.npy", "1gfJohwV63O9cvCo9oHWtQgilEqoBBsg1"),
                   ("smpl_mean_params.npz", "1O1--UkEFxCH8IwsYecdFuhjSVH0LlA7x"),
                   ("smplx2smpl.pkl", "1B9LtDH_mMWJZn5P2QnG_3PeQtHSIKOPg"),
                   ("smplx2smpl_joints.npy", "14wB62zX-OMF98xQegYsKEEi84W8gA9D0"))),
    Weight("duomo_supp_smplx", "duomo/supplementary_smplx", "gdrive_file", _DUOMO,
           repo="1v9Qy7ZXWcTM8_a9K2nSLyyVrJMFYcUOk", note="directory: keeps the Drive file name"),
    # --- Licence-gated body models: the user downloads these by hand ---
    Weight("smpl", "body_models/smpl/SMPL_NEUTRAL.pkl", "manual", _DUOMO,
           note="https://smpl.is.tue.mpg.de  ->  SMPL_python_v.1.1.0.zip"),
    Weight("smplx", "body_models/smplx/SMPLX_NEUTRAL.npz", "manual", _DUOMO,
           note="https://smpl-x.is.tue.mpg.de  ->  models_smplx_v1_1.zip"),
    Weight("smplh", "body_models/smplh/neutral/model.npz", "manual", _FLOW,
           note="https://mano.is.tue.mpg.de  ->  Extended SMPL+H model (AMASS), smplh.tar.xz"),
)

BY_KEY = {w.key: w for w in WEIGHTS}


def path(key: str) -> Path:
    return BY_KEY[key].path


def for_model(model: str) -> list[Weight]:
    return [w for w in WEIGHTS if model in w.models]


def missing(model: str) -> list[Weight]:
    return [w for w in for_model(model) if not w.exists()]


def describe_missing(model: str) -> str:
    lines = []
    for w in missing(model):
        how = w.note if w.source == "manual" else f"python scripts/download_weights.py --only {w.key}"
        gate = " [gated: accept access on huggingface.co + hf auth login]" if w.gated else ""
        lines.append(f"  - {w.path.relative_to(CHECKPOINTS.parent)}  <- {how}{gate}")
    return "\n".join(lines)


__all__ = ["Weight", "WEIGHTS", "BY_KEY", "path", "for_model", "missing", "describe_missing", "BODY_MODELS"]
