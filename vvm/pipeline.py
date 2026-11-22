"""End-to-end steps shared by the CLI scripts and the Gradio app.

    prepare_video  -> normalised clip + person tracks (cached per video)
    select_subject -> PreparedVideo with the chosen track's boxes
    run_models     -> SOMA motion per model (subprocess each)
    mocap_to_soma  -> SOMA motion from C3D (+ static calibration)
    compare        -> sync + metrics + figures + comparison video per model
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vvm.models import REGISTRY, output_path, run_subprocess
from vvm.models.base import PreparedVideo
from vvm.paths import OUTPUTS
from vvm.soma.motion import SomaMotion
from vvm.video.io import normalize, read_frames, video_id
from vvm.video.tracking import Tracks, detect_and_track, draw_tracks, best_preview_frame


# --------------------------------------------------------------------- video
def workdir_for(src: str | Path) -> Path:
    return OUTPUTS / "videos" / video_id(src)


def prepare_video(src: str | Path, device: str = "cuda", progress=None) -> tuple[Path, Tracks]:
    """Normalise (cached) and detect/track people (cached)."""
    wd = workdir_for(src)
    wd.mkdir(parents=True, exist_ok=True)
    norm = wd / "video_30fps.mp4"
    if not norm.exists():
        normalize(src, norm)
    tracks_file = wd / "tracks.json"
    if tracks_file.exists():
        tracks = Tracks.load(tracks_file)
    else:
        tracks = detect_and_track(norm, device=device, progress=progress)
        tracks.save(tracks_file)
    return wd, tracks.stitched()


def track_preview(wd: Path, tracks: Tracks, highlight: int | None = None) -> np.ndarray:
    f = best_preview_frame(tracks)
    frame = read_frames(wd / "video_30fps.mp4", [f], max_side=960)[0]
    scale = frame.shape[0] / tracks.height
    scaled = Tracks(tracks.num_frames, int(tracks.width * scale), int(tracks.height * scale),
                    {k: (fr, b * scale, s) for k, (fr, b, s) in tracks.items.items()})
    return draw_tracks(frame, scaled, f, highlight)


def select_subject(wd: Path, tracks: Tracks, track_id: int | None = None) -> PreparedVideo:
    tid = tracks.recommended() if track_id is None else int(track_id)
    boxes, observed = tracks.subject_boxes(tid)
    pv = PreparedVideo(wd, wd / "video_30fps.mp4", boxes, observed, tid)
    pv.save()
    return pv


# -------------------------------------------------------------------- models
def run_models(wd: Path, models: list[str], gpu: int | None = None, log=print,
               reuse: bool = True) -> dict[str, SomaMotion | Exception]:
    results: dict[str, SomaMotion | Exception] = {}
    for key in models:
        out = output_path(wd, key)
        if reuse and out.exists():
            results[key] = SomaMotion.load(out)
            log(f"{key}: cached result")
            continue
        log(f"{key}: running ...")
        try:
            results[key] = SomaMotion.load(run_subprocess(key, wd, gpu=gpu))
            log(f"{key}: done ({results[key].meta.get('runtime_s')} s)")
        except Exception as e:  # report and continue with the other models
            results[key] = e
            log(f"{key}: FAILED - {str(e).splitlines()[0]}")
    return results


def export_soma(wd: Path, key: str) -> Path:
    """Copy of a model's SOMA result with a self-describing name, for download."""
    import shutil

    src = output_path(wd, key)
    dst = wd / "exports" / f"{wd.name}__{key}.soma.npz"
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    return dst


def preview_video(wd: Path, motions: dict[str, SomaMotion], max_frames: int = 450) -> str | None:
    """Skeleton video of the models' SOMA output (no mocap needed); first model is the anchor."""
    from vvm.eval.metrics import align_sequence
    from vvm.viz.render import comparison_video

    if not motions:
        return None
    keys = list(motions)
    anchor = motions[keys[0]]
    grid = anchor.times[anchor.valid][:max_frames]
    res = {k: m.resample(grid) for k, m in motions.items()}
    ok = np.logical_and.reduce([r.valid for r in res.values()])
    aligned = {k: align_sequence(r.joints[ok], res[keys[0]].joints[ok]) for k, r in res.items()}
    frame_ids = np.round(grid[ok] * anchor.fps).astype(int)
    return comparison_video(wd / "soma_preview.mp4", str(wd / "video_30fps.mp4"), frame_ids, None, aligned,
                            anchor.joint_names, anchor.fps)


# --------------------------------------------------------------------- mocap
def mocap_to_soma(trial_c3d: str | Path, static_c3d: str | Path, device: str = "cuda", log=print,
                  up_axis: str | None = None) -> SomaMotion:
    from vvm.mocap.fit_soma import Calibration, calibrate, fit_trial
    from vvm.mocap.markers import load_c3d

    out_dir = OUTPUTS / "mocap"
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_file = out_dir / f"{Path(static_c3d).stem}_{video_id(static_c3d)[-8:]}_calib.npz"
    if calib_file.exists():
        calib = Calibration.load(calib_file)
    else:
        calib = calibrate(load_c3d(static_c3d, up_axis), device=device, log=log)
        calib.save(calib_file)
    motion_file = out_dir / f"{video_id(trial_c3d)}_soma.npz"
    if motion_file.exists():
        return SomaMotion.load(motion_file)
    motion = fit_trial(load_c3d(trial_c3d, up_axis), calib, device=device, log=log)
    motion.meta["static"] = str(static_c3d)
    motion.save(motion_file)
    return motion


# ------------------------------------------------------------------- compare
@dataclass
class Comparison:
    syncs: dict = field(default_factory=dict)
    evals: dict = field(default_factory=dict)
    times: np.ndarray | None = None  # common video-time grid used for metrics
    figures: dict = field(default_factory=dict)
    video: str | None = None
    table: list[dict] = field(default_factory=list)
    angle_table: list[dict] = field(default_factory=list)


def compare(wd: Path, motions: dict[str, SomaMotion], mocap: SomaMotion, reference_model: str | None = None,
            render: bool = True, log=print) -> Comparison:
    """Synchronise every model to mocap, evaluate on a common grid and plot.

    The time offset is estimated per model; the one with the strongest
    correlation peak (or ``reference_model``) defines the common offset, so
    all models are scored on the same frames.
    """
    from vvm.eval.metrics import align_sequence, evaluate
    from vvm.sync.auto_sync import auto_sync, combine, common_grid
    from vvm.viz import plots
    from vvm.viz.render import comparison_video

    cmp = Comparison()
    for key, m in motions.items():
        cmp.syncs[key] = auto_sync(m, mocap)
        log(f"sync {key}: {cmp.syncs[key].describe()}")
    if reference_model:
        offset, source = cmp.syncs[reference_model].offset, reference_model
    else:  # consensus of all models' correlation curves
        consensus = combine(cmp.syncs)
        offset, source = consensus.offset, "consensus of " + ", ".join(cmp.syncs)
        log(f"sync consensus: {consensus.describe()}")
    best = reference_model or max(cmp.syncs, key=lambda k: cmp.syncs[k].score)
    log(f"common offset ({source}): {offset:+.3f} s")

    grid = common_grid(motions[best], mocap, offset)
    ref = mocap.resample(grid - offset)
    keep = ref.valid.copy()
    res = {k: m.resample(grid) for k, m in motions.items()}
    for r in res.values():
        keep &= r.valid
    cmp.times = grid[keep]
    names = mocap.joint_names
    fps = motions[best].fps
    aligned = {}
    for key, r in res.items():
        ev = evaluate(r.joints[keep], ref.joints[keep], names, key, fps, times=cmp.times)
        cmp.evals[key] = ev
        aligned[key] = align_sequence(r.joints[keep], ref.joints[keep], scale=False)
        cmp.table.append({"sync offset (s)": round(cmp.syncs[key].offset, 3),
                          "sync corr": round(cmp.syncs[key].score, 3),
                          **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in ev.summary_row().items()}})
        for a, s in ev.angles.items():
            cmp.angle_table.append({"model": key, "angle": a, **{k: round(v, 2) for k, v in s.items()}})

    cmp.figures = {"angles": plots.angle_curves(cmp.times, cmp.evals),
                   "per_joint": plots.per_joint_heatmap(cmp.evals),
                   "error_time": plots.error_over_time(cmp.times, cmp.evals),
                   "sync": plots.sync_curve(cmp.syncs)}
    report = wd / "comparison"
    report.mkdir(exist_ok=True)
    for k, fig in cmp.figures.items():
        fig.write_html(report / f"{k}.html", include_plotlyjs="cdn")
    (report / "summary.json").write_text(json.dumps({"offset_s": offset, "offset_source": source,
                                                     "table": cmp.table, "angles": cmp.angle_table}, indent=2))
    if render and len(cmp.times):
        frame_ids = np.round(cmp.times * fps).astype(int)
        cmp.video = comparison_video(report / "comparison.mp4", str(wd / "video_30fps.mp4"), frame_ids,
                                     ref.joints[keep], aligned, names, fps)
    return cmp


def available_models() -> list[str]:
    return list(REGISTRY)
