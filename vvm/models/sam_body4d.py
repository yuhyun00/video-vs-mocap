"""SAM-Body4D -> SOMA.

Runs the upstream offline pipeline (SAM 3 masklets + SAM 3D Body) with these
changes, all made from outside the repo:

* the masklet is prompted with the shared subject box;
* camera intrinsics are shared with SAM 3D Body (one estimate per video);
* the video is processed in chunks so SAM 3 fits in host RAM;
* MHR parameters are captured instead of meshes, then converted to SOMA.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from vvm import weights
from vvm.models.base import PreparedVideo, VideoModel
from vvm.models.sam3d_body import mhr_outputs_to_soma, static_intrinsics
from vvm.soma.motion import SomaMotion
from vvm.video.io import iter_frames

CHUNK = 180  # frames per SAM 3 run (6 s at 30 fps), limited by host RAM
# Frames per SAM 3D Body batch (upstream: 64; 16 runs out of 24 GB). Smoothing is per batch,
# so depth jumps at batch boundaries; raise it on larger GPUs.
BATCH = 8


def _config(out_dir: Path, completion: bool) -> Path:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "third_party/sam-body4d/configs/body4d.yaml")
    cfg.paths.ckpt_root = str(weights.path("sam3").parents[1])
    cfg.sam3.ckpt_path = str(weights.path("sam3"))
    cfg.sam_3d_body.ckpt_path = str(weights.path("sam3db_model.ckpt"))
    cfg.sam_3d_body.mhr_path = str(weights.path("sam3db_assets/mhr_model.pt"))
    cfg.sam_3d_body.fov_path = str(weights.path("moge2"))
    cfg.sam_3d_body.batch_size = BATCH
    cfg.runtime.output_dir = str(out_dir / "body4d")
    cfg.completion.enable = completion
    cfg.completion.model_path_mask = str(weights.path("vas_amodal"))
    cfg.completion.model_path_rgb = str(weights.path("vas_completion"))
    cfg.completion.model_path_depth = str(weights.path("depth_anything_v2"))
    path = out_dir / "body4d.yaml"
    OmegaConf.save(cfg, path)
    return path


class SamBody4D(VideoModel):
    key = "sam-body4d"
    name = "SAM-Body4D"
    native_format = "MHR (per frame via masklet-prompted SAM 3D Body)"
    repo = "sam-body4d"
    completion = True  # Diffusion-VAS occlusion completion (the method's core feature)

    def run(self, video: PreparedVideo, out_dir: Path, device: str) -> SomaMotion:
        native = out_dir / "native_mhr.npz"
        if self.native_cache_valid(video, out_dir) and native.exists() and int(np.load(native)["batch"]) == BATCH:
            d = np.load(native)
            return mhr_outputs_to_soma(self.key, video.times[:len(d["valid"])], d["shape"], d["model_params"],
                                       d["cam_t"], d["valid"], device,
                                       {"completion": self.completion, "intrinsics": d["K"].tolist()})
        import gc
        import hashlib

        import torch

        self.add_repo_to_path("scripts")
        import offline_app as app
        from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

        K = static_intrinsics(video, device, FOVEstimator)
        K_t = torch.as_tensor(K)

        T = len(video.boxes)
        captured: dict[int, dict] = {}
        chunk_start = [0]  # SAM-Body4D numbers frames 0.. inside each chunk

        def capture(outputs, faces, save_dir, focal_dir, image_path, id_current):
            if outputs:
                captured[chunk_start[0] + int(Path(image_path).stem)] = outputs[0]

        # Replace upstream side effects (PLY export, rendering, mp4 writing, ViTDet detector).
        app.save_mesh_results = capture
        app.visualize_sample_together = lambda img, *a, **k: img
        app.visualize_sample = lambda img, *a, **k: []
        app.jpg_folder_to_mp4 = lambda *a, **k: None
        app.build_sam3_3d_body_config = _estimator_without_detector
        process_with_mask = app.process_image_with_mask
        app.process_image_with_mask = lambda *a, **kw: process_with_mask(*a, **{**kw, "cam_int": K_t})

        frames_dir = out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        paths = []
        for t, frame in enumerate(iter_frames(video.path, rgb=False)):
            p = frames_dir / f"{t:08d}.jpg"
            if not p.exists():
                cv2.imwrite(str(p), frame)
            paths.append(str(p))
        h, w = cv2.imread(paths[0]).shape[:2]

        # equal-length chunks: SAM-Body4D's occlusion completion fails on very short tails
        n_chunks = -(-T // CHUNK)
        size = -(-T // n_chunks)
        key = video.boxes.tobytes() + K.tobytes() + str(BATCH).encode()
        subject = hashlib.sha1(key).hexdigest()[:10]  # chunks depend on subject, camera and batch size
        chunk_dir = out_dir / "chunks" / subject
        chunk_dir.mkdir(parents=True, exist_ok=True)
        runner = None
        for start in range(0, T, size):
            chunk = paths[start:start + size]
            done = chunk_dir / f"{start:06d}_{len(chunk):04d}.npz"
            if done.exists():  # finished in an earlier (interrupted) run
                captured.update(np.load(done, allow_pickle=True)["captured"].item())
                continue
            if runner is None:  # load SAM 3 / SAM 3D Body / Diffusion-VAS only if needed
                runner = app.OfflineApp(str(_config(out_dir, self.completion)))
            seen = np.flatnonzero(video.observed[start:start + len(chunk)])
            k = int(seen[0]) if len(seen) else 0
            print(f"[sam-body4d] frames {start}-{start + len(chunk) - 1}", flush=True)
            runner.OUTPUT_DIR = str(out_dir / "body4d" / f"chunk_{start:06d}")
            Path(runner.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            state = runner.predictor.init_state(video_path=chunk, offload_video_to_cpu=True)
            runner.predictor.clear_all_points_in_video(state)
            runner.RUNTIME["inference_state"] = state
            x1, y1, x2, y2 = video.boxes[start + k]
            box = np.array([[x1 / w, y1 / h, x2 / w, y2 / h]], np.float32)
            _, runner.RUNTIME["out_obj_ids"], _, _ = runner.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=k, obj_id=1, box=box)
            chunk_start[0] = start
            runner.on_mask_generation(start_frame_idx=0)  # local numbering: upstream assumes it
            with torch.autocast("cuda", enabled=False):
                runner.on_4d_generation()
            keep = ("shape_params", "mhr_model_params", "pred_cam_t")
            mine = {t: {f: np.asarray(o[f]) for f in keep} for t, o in captured.items()
                    if start <= t < start + len(chunk)}
            np.savez(done, captured=np.array(mine, dtype=object))
            del state
            runner.RUNTIME.pop("inference_state", None)
            gc.collect()
            torch.cuda.empty_cache()
        del runner

        shape = np.zeros((T, 45), np.float32)
        params = np.zeros((T, 204), np.float32)
        cam_t = np.zeros((T, 3), np.float32)
        valid = np.zeros(T, bool)
        for t, o in captured.items():
            if t < T:
                shape[t], params[t], cam_t[t], valid[t] = o["shape_params"], o["mhr_model_params"], o["pred_cam_t"], True
        np.savez(native, shape=shape, model_params=params, cam_t=cam_t, valid=valid, K=K, batch=BATCH)
        self.mark_native_cache(video, out_dir)
        return mhr_outputs_to_soma(self.key, video.times[:T], shape, params, cam_t, valid, device,
                                   {"completion": self.completion, "chunk": CHUNK, "batch": BATCH,
                                    "intrinsics": K.tolist()})


def _estimator_without_detector(cfg):
    """SAM 3D Body estimator as built by SAM-Body4D, minus the ViTDet detector (boxes come from masks)."""
    import torch
    from models.sam_3d_body.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

    device = torch.device("cuda")
    model, model_cfg = load_sam_3d_body(cfg.sam_3d_body["ckpt_path"], device=device,
                                        mhr_path=cfg.sam_3d_body["mhr_path"])
    fov = FOVEstimator(name="moge2", device=device, path=cfg.sam_3d_body["fov_path"])
    return SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg, human_detector=None,
                              human_segmentor=None, fov_estimator=fov)
