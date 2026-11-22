"""Child-process entry point: ``python -m vvm.models.runner --model KEY --workdir DIR``."""

from __future__ import annotations

import argparse
import time

from vvm.models import REGISTRY, get, output_path
from vvm.models.base import PreparedVideo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    video = PreparedVideo.load(args.workdir)
    model = get(args.model)
    model.check_ready()
    out_dir = video.model_dir(args.model)
    t0 = time.time()
    print(f"[runner] {model.name} on {video.path} ({len(video.boxes)} frames)", flush=True)
    motion = model.run(video, out_dir, args.device)
    motion.meta.update({"runtime_s": round(time.time() - t0, 1), "video": str(video.path),
                        "track_id": video.track_id})
    motion.save(output_path(args.workdir, args.model))
    print(f"[runner] done in {time.time() - t0:.1f}s -> {output_path(args.workdir, args.model)}", flush=True)


if __name__ == "__main__":
    main()
