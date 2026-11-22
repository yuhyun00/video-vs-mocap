"""Side-by-side comparison video: source frame | front view | side view.

Skeletons are drawn with OpenCV using an orthographic projection after the
model motion has been rigidly aligned to the mocap motion (one transform for
the whole sequence), so what differs on screen is pose, not world placement.
"""

from __future__ import annotations

import cv2
import numpy as np

from vvm.eval.angles import pelvis_frame
from vvm.models import display_name
from vvm.soma.skeleton import BONES
from vvm.video.io import read_frames, write_video
from vvm.viz.style import SURFACE, GRID, TEXT, bgr, color


def _heading_rotation(ref: np.ndarray, names) -> np.ndarray:
    """Rotation about Y making the mocap subject face +Z on average (front view)."""
    _, _, ant = pelvis_frame(ref, names)
    a = ant.mean(0)
    yaw = np.arctan2(a[0], a[2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _panel(skeletons: list[tuple[np.ndarray, str, int]], names, view: str, size: int, center: np.ndarray,
           span: float, label: str, legend: list[tuple[str, str]] | None = None) -> np.ndarray:
    img = np.full((size, size, 3), bgr(SURFACE), np.uint8)
    ax = (0, 1) if view == "front" else (2, 1)  # x-y or z-y plane
    sgn = (1, -1) if view == "front" else (-1, -1)
    px = lambda p: (int(size / 2 + sgn[0] * (p[ax[0]] - center[ax[0]]) / span * size),  # noqa: E731
                    int(size / 2 + sgn[1] * (p[ax[1]] - center[ax[1]]) / span * size))
    floor_y = px(np.array([0, center[1] - span * 0.45, 0]))[1]
    cv2.line(img, (0, floor_y), (size, floor_y), bgr(GRID), 1)
    idx = {n: i for i, n in enumerate(names)}
    for joints, hex_color, thick in skeletons:
        col = bgr(hex_color)
        for a, b in BONES:
            cv2.line(img, px(joints[idx[a]]), px(joints[idx[b]]), col, thick, cv2.LINE_AA)
        for a, b in BONES:
            cv2.circle(img, px(joints[idx[b]]), thick + 1, col, -1, cv2.LINE_AA)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr(TEXT), 1, cv2.LINE_AA)
    for i, (text, key) in enumerate(legend or []):  # swatch + name in text ink (identity not colour-only)
        y = 48 + 20 * i
        cv2.line(img, (10, y - 5), (34, y - 5), bgr(color(key)), 3, cv2.LINE_AA)
        cv2.putText(img, text, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr(TEXT), 1, cv2.LINE_AA)
    return img


def comparison_video(path, video_path: str, frame_ids: np.ndarray, ref: np.ndarray | None,
                     preds: dict[str, np.ndarray], names, fps: float, size: int = 480,
                     max_frames: int | None = None) -> str:
    """ref/preds: [N,J,3] on the same timestamps (preds already aligned to ref).

    Without ``ref`` (no mocap yet) the first model is the anchor for heading and
    placement, and only the models are drawn.
    """
    anchor = ref if ref is not None else next(iter(preds.values()))
    n = len(anchor) if max_frames is None else min(len(anchor), max_frames)
    R = _heading_rotation(anchor[:n], names)
    rot = lambda x: x @ R.T  # noqa: E731
    ref_r = rot(anchor[:n])
    hips = list(names).index("Hips")
    # display only: put every model's pelvis on the anchor pelvis so poses overlap
    preds_r = {k: rot(v[:n]) - rot(v[:n])[:, hips:hips + 1] + ref_r[:, hips:hips + 1] for k, v in preds.items()}
    span = 2.2
    frames = read_frames(video_path, frame_ids[:n], max_side=size)
    legend = ([("mocap", "mocap")] if ref is not None else []) + [(display_name(k), k) for k in preds]

    def gen():
        for i in range(n):
            center = ref_r[i].mean(0)
            skel = ([(ref_r[i], color("mocap"), 3)] if ref is not None else []) + \
                [(p[i], color(k), 2) for k, p in preds_r.items()]
            front = _panel(skel, names, "front", size, center, span, "front", legend)
            side = _panel(skel, names, "side", size, center, span, "side")
            src = frames[i] if i < len(frames) else np.zeros((size, size, 3), np.uint8)
            src = cv2.resize(src, (int(src.shape[1] * size / src.shape[0]), size))
            yield np.concatenate([src, front[..., ::-1], side[..., ::-1]], 1)

    return str(write_video(path, gen(), fps))
