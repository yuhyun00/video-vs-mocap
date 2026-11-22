"""Common interface for video -> SOMA models.

Each adapter only knows how to call its own upstream code and which SOMA
converter to use; everything else (normalised video, subject boxes, output
format, weight checks, subprocess launch) is shared.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vvm import weights
from vvm.soma.motion import SomaMotion
from vvm.video.io import VideoInfo, probe


@dataclass
class PreparedVideo:
    """A normalised clip plus the selected subject's boxes (shared by all models)."""

    workdir: Path  # outputs/<video id>/
    path: Path  # normalised upright constant-fps mp4
    boxes: np.ndarray  # [T,4] xyxy, gap filled
    observed: np.ndarray  # [T] detector actually saw the subject
    track_id: int

    @property
    def info(self) -> VideoInfo:
        return probe(self.path)

    @property
    def fps(self) -> float:
        return self.info.fps

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.boxes)) / self.fps

    def save(self) -> None:
        np.save(self.workdir / "subject_boxes.npy", self.boxes)
        np.save(self.workdir / "subject_observed.npy", self.observed)
        (self.workdir / "prepared.json").write_text(json.dumps(
            {"path": str(self.path), "track_id": self.track_id}, indent=2))

    @classmethod
    def load(cls, workdir: str | Path) -> "PreparedVideo":
        workdir = Path(workdir)
        meta = json.loads((workdir / "prepared.json").read_text())
        return cls(workdir, Path(meta["path"]), np.load(workdir / "subject_boxes.npy"),
                   np.load(workdir / "subject_observed.npy"), int(meta["track_id"]))

    def model_dir(self, key: str) -> Path:
        d = self.workdir / "models" / key
        d.mkdir(parents=True, exist_ok=True)
        return d


class VideoModel(ABC):
    key: str = ""
    name: str = ""
    native_format: str = ""  # what the upstream model outputs before SOMA conversion
    repo: str = ""  # directory under third_party/

    def missing_weights(self) -> list[weights.Weight]:
        return weights.missing(self.key)

    def check_ready(self) -> None:
        if self.missing_weights():
            raise FileNotFoundError(f"{self.name}: missing weights\n{weights.describe_missing(self.key)}")

    def link_weights(self, links: dict[str, str | Path]) -> None:
        """Expose files from checkpoints/ at the repo-relative paths upstream code expects.

        ``links`` maps a path inside the upstream repo to a manifest key or an
        absolute path. Symlinks keep a single copy of every weight on disk.
        """
        from vvm.paths import repo

        root = repo(self.repo)
        for rel, src in links.items():
            target = Path(src) if isinstance(src, Path) else weights.path(src)
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                if dst.resolve() == target.resolve():
                    continue
                dst.unlink()
            dst.symlink_to(target)

    # Native (pre-SOMA) outputs are cached per subject so a converter change
    # does not require re-running the network.
    @staticmethod
    def native_cache_valid(video: PreparedVideo, out_dir: Path) -> bool:
        f = out_dir / "native_boxes.npy"
        return f.exists() and np.array_equal(np.load(f), video.boxes)

    @staticmethod
    def mark_native_cache(video: PreparedVideo, out_dir: Path) -> None:
        np.save(out_dir / "native_boxes.npy", video.boxes)

    def add_repo_to_path(self, *subdirs: str) -> Path:
        """Put the upstream repo (and optional sub-directories) first on sys.path."""
        from vvm.paths import repo

        root = repo(self.repo)
        for p in (root, *(root / s for s in subdirs)):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        return root

    @abstractmethod
    def run(self, video: PreparedVideo, out_dir: Path, device: str) -> SomaMotion:
        """Run the upstream model and return its result converted to SOMA."""
