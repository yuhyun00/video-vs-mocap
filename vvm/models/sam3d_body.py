"""SAM 3D Body (Meta) -> SOMA.

Per-frame image model: every frame of the normalised clip is processed with
the shared subject box. Native output is MHR (shape 45 + model params 204) in
camera space; it is converted with the shared MHR -> SOMA converter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vvm import weights
from vvm.models.base import PreparedVideo, VideoModel
from vvm.soma import body
from vvm.soma.convert import from_mhr
from vvm.soma.motion import SomaMotion
from vvm.video.io import iter_frames

# Camera-space joints in our Y-up camera frame: joints_mhr + pred_cam_t * CAM_TO_YUP.
CAM_TO_YUP = np.array([1.0, -1.0, -1.0])


def mhr_outputs_to_soma(source: str, times: np.ndarray, shape: np.ndarray, model_params: np.ndarray,
                        cam_t: np.ndarray, valid: np.ndarray, device: str, meta: dict) -> SomaMotion:
    """Shared by SAM 3D Body and SAM-Body4D: per-frame MHR (camera space) -> SomaMotion."""
    T = len(times)
    joints = np.zeros((T, 77, 3), np.float32)
    if valid.any():
        conv = from_mhr(shape[valid], model_params[valid], device=device)
        joints[valid] = conv["joints"] + (cam_t[valid] * CAM_TO_YUP)[:, None]
        meta = {**meta, "mhr_to_soma_error_cm": float(np.mean(conv["fit_error_cm"]))}
    return SomaMotion(source=source, times=times, joints=joints, joint_names=body.joint_names(), valid=valid,
                      identity=np.median(shape[valid], 0) if valid.any() else None, identity_model="mhr",
                      frame="camera", meta=meta)


def static_intrinsics(video: PreparedVideo, device: str, fov_estimator_cls) -> np.ndarray:
    """One MoGe-2 intrinsics estimate per video, cached so both MHR models share the camera."""
    path = video.workdir / "intrinsics_moge2.npy"
    if path.exists():
        return np.load(path)
    first = next(iter_frames(video.path, rgb=True))
    fov = fov_estimator_cls(name="moge2", device=device, path=str(weights.path("moge2")))
    K = np.asarray(fov.get_cam_intrinsics(first), np.float32)
    del fov
    np.save(path, K)
    return K


class Sam3DBody(VideoModel):
    key = "sam3d-body"
    name = "SAM 3D Body"
    native_format = "MHR (per frame, camera space)"
    repo = "sam-3d-body"
    inference_type = "body"  # body decoder only: full-body output, skips the hand refinement

    def run(self, video: PreparedVideo, out_dir: Path, device: str) -> SomaMotion:
        native = out_dir / "native_mhr.npz"
        if self.native_cache_valid(video, out_dir) and native.exists():
            d = np.load(native)
            return mhr_outputs_to_soma(self.key, video.times[:len(d["valid"])], d["shape"], d["model_params"],
                                       d["cam_t"], d["valid"], device, {"intrinsics": d["K"].tolist()})
        self.add_repo_to_path()
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from tools.build_fov_estimator import FOVEstimator

        K = static_intrinsics(video, device, FOVEstimator)
        model, cfg = load_sam_3d_body(str(weights.path("sam3db_model.ckpt")), device=device,
                                      mhr_path=str(weights.path("sam3db_assets/mhr_model.pt")))
        estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
        K_t = torch.as_tensor(K)

        T = len(video.boxes)
        shape = np.zeros((T, 45), np.float32)
        params = np.zeros((T, 204), np.float32)
        cam_t = np.zeros((T, 3), np.float32)
        valid = np.zeros(T, bool)
        for t, frame in enumerate(iter_frames(video.path, rgb=True)):
            if t >= T:
                break
            out = estimator.process_one_image(frame, bboxes=video.boxes[t][None], cam_int=K_t,
                                              inference_type=self.inference_type)
            if out:
                shape[t], params[t], cam_t[t] = out[0]["shape_params"], out[0]["mhr_model_params"], out[0]["pred_cam_t"]
                valid[t] = True
            if t % 100 == 0:
                print(f"[sam3d-body] frame {t}/{T}", flush=True)
        np.savez(native, shape=shape, model_params=params, cam_t=cam_t, valid=valid, K=K)
        self.mark_native_cache(video, out_dir)
        del estimator, model
        return mhr_outputs_to_soma(self.key, video.times[:T], shape, params, cam_t, valid, device,
                                   {"intrinsics": K.tolist()})
