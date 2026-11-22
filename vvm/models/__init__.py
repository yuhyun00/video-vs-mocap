"""Model registry and the subprocess launcher shared by the CLI and the web UI.

Every model runs in the same (unified) Python environment, but in its own
child process: upstream repositories use clashing top-level module names
(``src``, ``configs``, ``sam_3d_body`` in three variants...) and each holds
several GB of GPU memory, so process isolation keeps runs independent.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from vvm.paths import ROOT, VENV_PYTHON

# key -> (module, class). Imported lazily so listing models needs no torch.
REGISTRY: dict[str, tuple[str, str, str]] = {
    "gem-x": ("vvm.models.gemx", "GemX", "GEM-X (NVIDIA)"),
    "sam3d-body": ("vvm.models.sam3d_body", "Sam3DBody", "SAM 3D Body (Meta)"),
    "sam-body4d": ("vvm.models.sam_body4d", "SamBody4D", "SAM-Body4D"),
    "duomo": ("vvm.models.duomo", "DuoMo", "DuoMo (Meta)"),
    "flowhmr": ("vvm.models.flowhmr", "FlowHMR", "FlowHMR"),
}


def display_name(key: str) -> str:
    return REGISTRY[key][2]


def get(key: str):
    import importlib

    module, cls, _ = REGISTRY[key]
    return getattr(importlib.import_module(module), cls)()


def output_path(workdir: str | Path, key: str) -> Path:
    return Path(workdir) / "models" / key / "soma.npz"


def run_subprocess(key: str, workdir: str | Path, gpu: int | None = None, log_path: str | Path | None = None,
                   python: str | Path = VENV_PYTHON) -> Path:
    """Run one model on a prepared video in a child process; returns the SOMA npz path."""
    workdir = Path(workdir)
    log_path = Path(log_path or workdir / "models" / key / "run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYOPENGL_PLATFORM", "egl")  # upstream visualisation modules import pyrender headless
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on 24 GB GPUs
    cmd = [str(python), "-m", "vvm.models.runner", "--model", key, "--workdir", str(workdir)]
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    out = output_path(workdir, key)
    if proc.returncode != 0 or not out.exists():
        tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-25:])
        raise RuntimeError(f"{display_name(key)} failed (exit {proc.returncode}). Log: {log_path}\n{tail}")
    return out
