"""Mocap (C3D) -> SOMA, auto-sync with each model's SOMA output, metrics + report.

    .venv/bin/python scripts/compare.py VIDEO TRIAL.c3d STATIC.c3d [--sync-ref gem-x]

VIDEO must have been processed with scripts/run_video.py first. Results go
to outputs/videos/<id>/comparison/ (summary.json, *.html plots, comparison.mp4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vvm import pipeline  # noqa: E402
from vvm.models import REGISTRY, output_path  # noqa: E402
from vvm.soma.motion import SomaMotion  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("trial")
    ap.add_argument("static")
    ap.add_argument("--sync-ref", choices=sorted(REGISTRY), help="model whose sync offset is shared")
    ap.add_argument("--up-axis", help="override vertical axis detection, e.g. +z")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()

    wd = pipeline.workdir_for(a.video)
    motions = {k: SomaMotion.load(output_path(wd, k)) for k in REGISTRY if output_path(wd, k).exists()}
    if not motions:
        sys.exit(f"no model outputs under {wd}; run scripts/run_video.py first")
    mocap = pipeline.mocap_to_soma(a.trial, a.static, up_axis=a.up_axis)
    cmp = pipeline.compare(wd, motions, mocap, a.sync_ref, render=not a.no_render)
    import pandas as pd

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.DataFrame(cmp.table).to_string(index=False))
    print(f"\nreport: {wd / 'comparison'}")


if __name__ == "__main__":
    main()
