"""Person detection + tracking and subject selection, shared by every model.

Detection runs once per video; the chosen subject's per-frame boxes are then
handed to every model, so all models reconstruct the same person.

The YOLOX-HumanArt ONNX detector and ByteTrack implementation are reused from
GEM-X (installed in the unified environment) instead of being duplicated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vvm import weights
from vvm.video.io import iter_frames, probe


@dataclass
class Tracks:
    num_frames: int
    width: int
    height: int
    # track id -> (frame indices [N], boxes xyxy [N,4], scores [N])
    items: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        payload = {"num_frames": self.num_frames, "width": self.width, "height": self.height,
                   "tracks": {str(k): {"frames": f.tolist(), "boxes": b.round(2).tolist(),
                                       "scores": s.round(3).tolist()}
                              for k, (f, b, s) in self.items.items()}}
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "Tracks":
        d = json.loads(Path(path).read_text())
        items = {int(k): (np.asarray(v["frames"], int), np.asarray(v["boxes"], np.float32).reshape(-1, 4),
                          np.asarray(v["scores"], np.float32)) for k, v in d["tracks"].items()}
        return cls(d["num_frames"], d["width"], d["height"], items)

    # ------------------------------------------------------------ selection
    def ranking(self) -> list[tuple[int, float]]:
        """Subject score per track, best first (the subject stays visible, in place and inside the frame)."""
        out = []
        for tid, (f, b, _) in self.items.items():
            coverage = len(f) / max(self.num_frames, 1)
            h = np.maximum(b[:, 3] - b[:, 1], 1.0)
            centers = (b[:, :2] + b[:, 2:]) / 2
            motion = np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1) / h[1:]) if len(f) > 1 else 1.0
            margin = 2.0
            touches = ((b[:, 0] < margin) | (b[:, 1] < margin) | (b[:, 2] > self.width - margin)
                       | (b[:, 3] > self.height - margin)).mean()
            score = coverage * (1.0 - 0.5 * touches) / (1.0 + 20.0 * motion)
            out.append((tid, float(score)))
        return sorted(out, key=lambda x: -x[1])

    def recommended(self) -> int:
        ranked = self.ranking()
        if not ranked:
            raise ValueError("no person detected")
        return ranked[0][0]

    def stitched(self, max_gap: int = 150, max_jump: float = 1.5) -> "Tracks":
        """Join track fragments split by occlusions (disjoint in time, close in space).

        A fragment joins the chain whose last box is nearest its first box if the gap is
        at most ``max_gap`` frames and the jump is below ``max_jump`` box heights.
        """
        order = sorted(self.items, key=lambda k: self.items[k][0][0])
        chains: dict[int, list] = {}
        for tid in order:
            f, b, s = self.items[tid]
            best, best_d = None, max_jump
            for cid, parts in chains.items():
                lf, lb = parts[-1][0][-1], parts[-1][1][-1]
                if lf >= f[0] or f[0] - lf > max_gap:
                    continue
                h = max(lb[3] - lb[1], 1.0)
                d = np.linalg.norm((lb[:2] + lb[2:]) / 2 - (b[0, :2] + b[0, 2:]) / 2) / h
                d += 0.5 * abs(np.log(max(b[0, 3] - b[0, 1], 1.0) / h))  # size change (occlusion halves boxes)
                if d < best_d:
                    best, best_d = cid, d
            chains.setdefault(best if best is not None else tid, []).append((f, b, s))
        items = {cid: tuple(np.concatenate(x) for x in zip(*parts)) for cid, parts in chains.items()}
        return Tracks(self.num_frames, self.width, self.height, items)

    def subject_boxes(self, tid: int) -> tuple[np.ndarray, np.ndarray]:
        """Gap-filled xyxy boxes [T,4] and observed mask [T] for one track."""
        f, b, _ = self.items[tid]
        T = self.num_frames
        boxes = np.zeros((T, 4), np.float32)
        observed = np.zeros(T, bool)
        observed[f] = True
        for k in range(4):
            boxes[:, k] = np.interp(np.arange(T), f, b[:, k])  # holds endpoints
        return boxes, observed

    def boxes_at(self, frame: int) -> list[tuple[int, np.ndarray]]:
        out = []
        for tid, (f, b, _) in self.items.items():
            i = np.searchsorted(f, frame)
            if i < len(f) and f[i] == frame:
                out.append((tid, b[i]))
        return out


def detect_and_track(video_path: str | Path, device: str = "cuda", stride: int = 1,
                     min_track_frames: int = 15, progress=None) -> Tracks:
    import onnxruntime

    onnxruntime.preload_dlls()  # CUDA/cuDNN from the pip nvidia-* wheels that torch installed
    from gem.common.detectors import YOLOXONNX
    from gem.common.detectors.bytetrack import ByteTracker

    info = probe(video_path)
    detector = YOLOXONNX(weights.path("yolox_humanart"), score_thr=0.5, device=device)
    tracker = ByteTracker()
    rows: dict[int, list] = {}
    for t, frame in enumerate(iter_frames(video_path, rgb=False)):
        if t % stride:
            continue
        boxes, scores = detector.detect(frame)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        for box, tid, score in tracker.update(boxes, np.asarray(scores, np.float32)):
            rows.setdefault(int(tid), []).append((t, box, score))
        if progress and t % 100 == 0:
            progress(t / max(info.num_frames, 1))
    items = {}
    for tid, r in rows.items():
        if len(r) * stride < min_track_frames:
            continue
        items[tid] = (np.array([x[0] for x in r]), np.array([x[1] for x in r], np.float32),
                      np.array([x[2] for x in r], np.float32))
    return Tracks(info.num_frames, info.width, info.height, items)


def draw_tracks(frame_rgb: np.ndarray, tracks: Tracks, frame: int, highlight: int | None = None) -> np.ndarray:
    img = frame_rgb.copy()
    scale = max(img.shape[:2]) / 1000
    for tid, b in tracks.boxes_at(frame):
        color = (255, 60, 60) if tid == highlight else (60, 200, 255)
        x1, y1, x2, y2 = b.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(3 * scale)))
        cv2.putText(img, f"ID {tid}", (x1 + 4, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2 * scale, color, max(2, int(3 * scale)))
    return img


def best_preview_frame(tracks: Tracks) -> int:
    """Frame where the most tracks are visible at once (for the picker UI)."""
    counts = np.zeros(tracks.num_frames, int)
    for f, _, _ in tracks.items.values():
        counts[f] += 1
    return int(np.argmax(counts))
