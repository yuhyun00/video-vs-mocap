"""Single source of truth for project directories.

Every module resolves files through these constants, so moving a directory
means editing this file only.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
CHECKPOINTS = Path(os.environ.get("VVM_CHECKPOINTS", ROOT / "checkpoints"))
BODY_MODELS = CHECKPOINTS / "body_models"
OUTPUTS = Path(os.environ.get("VVM_OUTPUTS", ROOT / "outputs"))
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"

# Interpreter of the unified environment. Model adapters launch their runner
# with it so that every model executes in the same environment.
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def repo(name: str) -> Path:
    """Path of a cloned third-party repository."""
    return THIRD_PARTY / name
