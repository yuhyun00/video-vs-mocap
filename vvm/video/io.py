"""Video probing, normalisation and frame access shared by every model.

All models receive the *same* normalised clip (upright, constant frame rate,
H.264), so their outputs share one timeline and differences come from the
models, not from decoding.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FPS = 30.0


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    num_frames: int
    rotation: int  # degrees stored in the container metadata

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    info = VideoInfo(
        path=str(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        rotation=int(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
    )
    cap.release()
    return info


def video_id(path: str | Path) -> str:
    """Stable short id: file stem + hash of name/size/mtime."""
    p = Path(path)
    st = p.stat()
    h = hashlib.sha1(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:8]
    return f"{p.stem}_{h}"


def normalize(src: str | Path, dst: str | Path, fps: float = DEFAULT_FPS,
              start: float = 0.0, duration: float | None = None) -> VideoInfo:
    """Transcode to an upright, constant-``fps`` H.264 clip.

    ffmpeg applies the container rotation while re-encoding, which fixes the
    phone clips stored sideways with a 90 degree flag.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-an", "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-metadata:s:v:0", "rotate=0", str(dst)]
    subprocess.run(cmd, check=True)
    info = probe(dst)
    meta = {"source": str(src), "start": start, "duration": duration, **asdict(info)}
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return info


def read_frames(path: str | Path, indices=None, rgb: bool = True, max_side: int | None = None):
    """Decode frames (all, or the sorted ``indices``) as uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    wanted = None if indices is None else set(int(i) for i in indices)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if wanted is None or i in wanted:
            if max_side:
                s = max_side / max(frame.shape[:2])
                if s < 1:
                    frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if rgb else frame)
            if wanted is not None and len(frames) == len(wanted):
                break
        i += 1
    cap.release()
    return frames


def iter_frames(path: str | Path, rgb: bool = True):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if rgb else frame
    finally:
        cap.release()


def write_video(path: str | Path, frames, fps: float) -> Path:
    """Write RGB frames to an H.264 mp4 (browser playable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w2}x{h2}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-crf", "20", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in _chain(first, frames):
        proc.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    return path


def _chain(first, rest):
    yield first
    yield from rest
