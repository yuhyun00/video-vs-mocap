"""GEM-X Body Motion (NVIDIA) -> SOMA. Native output is already SOMA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from vvm import weights
from vvm.models.base import PreparedVideo, VideoModel
from vvm.paths import repo
from vvm.soma import body
from vvm.soma.motion import SomaMotion


def _config(camera_mode: str) -> dict:
    """Release YAML with checkpoint paths pointed at checkpoints/gemx."""
    cfg = OmegaConf.to_container(OmegaConf.load(repo("GEM-X") / "docs/configs/body_video_native77.yaml"))
    cfg["pose"]["checkpoint"] = str(weights.path("gemx_body_s240000.ckpt"))
    cfg["observations"]["checkpoint"] = str(weights.path("gemx_vitpose.pth"))
    cfg["detector"]["checkpoint"] = str(weights.path("yolox_humanart"))
    cfg["motion"]["checkpoint"] = str(weights.path("gemx_motion_native77_e774_s292175.ckpt"))
    cfg["motion"]["stats"] = str(weights.path("gemx_motion_stats_raw.npz"))
    cfg["camera"]["mode"] = camera_mode
    return cfg


def _track_from_boxes(boxes: np.ndarray, observed: np.ndarray, bbox_enlarge: float = 1.2):
    """Our subject boxes in GEM-X's TrackResult form (same crop policy as GEM-X)."""
    from gem.common.detectors.tracks import TrackResult, smooth_boxes, square_crops

    T = len(boxes)
    valid = np.flatnonzero(observed) if observed.any() else np.arange(T)
    empty = [np.zeros((0, 4), np.float32) for _ in range(T)]
    return TrackResult(
        np.arange(T), boxes.astype(np.float32), square_crops(smooth_boxes(boxes), bbox_enlarge),
        np.zeros(T, np.int32), observed.astype(bool), (int(valid[0]), int(valid[-1])),
        [b[None] for b in boxes.astype(np.float32)], empty, empty, [np.zeros(0, np.float32)] * T,
        ["primary_track" if o else "primary_interpolated" for o in observed])


class GemX(VideoModel):
    key = "gem-x"
    name = "GEM-X"
    native_format = "SOMA (identity128 + scale57, 77 joints)"
    repo = "GEM-X"
    camera_mode = "static"  # lab clips: fixed camera

    def run(self, video: PreparedVideo, out_dir: Path, device: str) -> SomaMotion:
        import gem.inference.body_video as bv
        from gem.inference.motion import load_predictor, run_motion
        from gem.inference.video import VideoFrames

        native = out_dir / "native.pt"
        if self.native_cache_valid(video, out_dir) and native.exists():
            result = torch.load(native, weights_only=False)  # same subject: only redo the SOMA conversion
        else:
            cfg = _config(self.camera_mode)
            # Hand GEM-X the shared subject track instead of its "largest person" rule.
            bv.detect_track = lambda frames, detector, **kw: _track_from_boxes(video.boxes, video.observed)
            with VideoFrames(video.path) as frames:
                sequence, _, _ = bv.build_body_conditions(frames, cfg, device=device, progress=print)
            predictor = load_predictor(cfg["motion"]["checkpoint"], cfg["motion"],
                                       stats_path=cfg["motion"]["stats"], device=device)
            result = run_motion(predictor, sequence, device=device)
            torch.save(result, native)
            self.mark_native_cache(video, out_dir)

        p = {k: v[0].float() for k, v in result["pred_soma_params_global"].items()}  # drop batch dim
        T = p["global_orient"].shape[0]
        ident = p["identity_coeffs"].reshape(-1, body.IDENTITY_DIM).mean(0)
        scale = p["scale_params"].reshape(-1, body.SCALE_DIM).mean(0)
        joints = body.forward(p["global_orient"], p["body_pose"], ident, scale, p["transl"], device=device)["joints"]
        return SomaMotion(
            source=self.key, times=video.times[:T], joints=joints.cpu().numpy(), joint_names=body.joint_names(),
            valid=np.ones(T, bool), poses=torch.cat([p["global_orient"][:, None],
                                                     p["body_pose"].reshape(T, -1, 3)], 1).cpu().numpy(),
            transl=p["transl"].cpu().numpy(), identity=ident.cpu().numpy(), scale_params=scale.cpu().numpy(),
            identity_model="soma", frame="world", meta={"camera_mode": self.camera_mode})
