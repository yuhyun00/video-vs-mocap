"""Video -> SOMA for one or more models (CLI twin of the web UI's first tab).

    .venv/bin/python scripts/run_video.py VIDEO --models gem-x sam3d-body [--track ID] [--gpu 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vvm import pipeline  # noqa: E402
from vvm.models import REGISTRY  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--models", nargs="+", default=["gem-x"], choices=sorted(REGISTRY))
    ap.add_argument("--track", type=int, help="subject track id (default: recommended)")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--rerun", action="store_true", help="ignore cached model outputs")
    a = ap.parse_args()

    wd, tracks = pipeline.prepare_video(a.video)
    print("tracks (id, score):", [(t, round(s, 3)) for t, s in tracks.ranking()])
    pv = pipeline.select_subject(wd, tracks, a.track)
    print(f"subject track {pv.track_id}; workdir {wd}")
    for key, res in pipeline.run_models(wd, a.models, gpu=a.gpu, reuse=not a.rerun).items():
        print(f"{key}: {'FAILED ' + str(res) if isinstance(res, Exception) else 'ok'}")


if __name__ == "__main__":
    main()
