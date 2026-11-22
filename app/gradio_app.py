"""Web UI: video -> SOMA with several models, then comparison against mocap.

    .venv/bin/python app/gradio_app.py  [--port 7860] [--no-share]

All logic lives in ``vvm.pipeline``; this file only wires widgets to it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vvm import pipeline, weights  # noqa: E402
from vvm.models import REGISTRY, display_name, output_path  # noqa: E402
from vvm.paths import DATA  # noqa: E402
from vvm.soma.motion import SomaMotion  # noqa: E402
from vvm.video.tracking import Tracks  # noqa: E402


def _samples(patterns: tuple[str, ...]) -> list[tuple[str, str]]:
    """(label relative to data/, absolute path) for every matching sample file."""
    files = sorted(p for pat in patterns for p in DATA.rglob(pat))
    return [(str(p.relative_to(DATA)), str(p)) for p in files]


def _model_choices() -> list[tuple[str, str]]:
    out = []
    for key in REGISTRY:
        miss = weights.missing(key)
        label = display_name(key) + ("" if not miss else f"  (weights missing: {len(miss)})")
        out.append((label, key))
    return out


def _resolve(upload, sample):
    path = upload or sample
    if not path:
        raise gr.Error("Upload a file or pick a sample.")
    return str(path)


# ----------------------------------------------------------------- step 1
def step_prepare(upload, sample, progress=gr.Progress()):
    src = _resolve(upload, sample)
    progress(0.05, desc="Normalising video (upright, 30 fps)")
    wd, tracks = pipeline.prepare_video(
        src, progress=lambda f: progress(0.1 + 0.8 * f, desc="Detecting and tracking people"))
    if not tracks.items:
        raise gr.Error("No person detected.")
    rec = tracks.recommended()
    ranking = "\n".join(f"- ID {tid}: score {s:.3f}, {len(tracks.items[tid][0])} frames"
                        for tid, s in tracks.ranking())
    choices = [str(t) for t, _ in tracks.ranking()]
    pv = pipeline.select_subject(wd, tracks, rec)
    return (str(wd), pipeline.track_preview(wd, tracks, rec), gr.update(choices=choices, value=str(rec)),
            f"Recommended subject: **ID {rec}** ({len(choices)} tracks)\n\n{ranking}", _status_table(pv.workdir))


def step_pick(wd, track_id):
    if not wd:
        raise gr.Error("Run step ① first.")
    tracks = Tracks.load(Path(wd) / "tracks.json").stitched()
    pipeline.select_subject(Path(wd), tracks, int(track_id))
    return pipeline.track_preview(Path(wd), tracks, int(track_id))


# ----------------------------------------------------------------- step 2
def _status_table(wd) -> pd.DataFrame:
    rows = []
    for key in REGISTRY:
        out = output_path(wd, key) if wd else None
        miss = weights.missing(key)
        state = "done" if out is not None and out.exists() else ("weights missing" if miss else "ready")
        rows.append({"model": display_name(key), "status": state,
                     "missing weights": ", ".join(w.key for w in miss)})
    return pd.DataFrame(rows)


def step_run(wd, models, gpu, rerun):
    if not wd:
        raise gr.Error("Run step ① first.")
    if not models:
        raise gr.Error("Select at least one model.")
    log = []
    files = []
    for key in models:
        log.append(f"▶ {display_name(key)} running ...")
        yield "\n".join(log), _status_table(wd), files, None
        res = pipeline.run_models(Path(wd), [key], gpu=None if gpu == "auto" else int(gpu),
                                  log=lambda m: None, reuse=not rerun)[key]
        if isinstance(res, Exception):
            log[-1] = f"✖ {display_name(key)} failed: {str(res).splitlines()[0]}  (log: models/{key}/run.log)"
        else:
            log[-1] = (f"✔ {display_name(key)}: {res.num_frames} frames, valid {res.valid.mean() * 100:.0f}%, "
                       f"{res.meta.get('runtime_s', '?')} s")
            files.append(str(pipeline.export_soma(Path(wd), key)))
        yield "\n".join(log), _status_table(wd), files, None
    motions = {k: SomaMotion.load(output_path(wd, k)) for k in models if output_path(wd, k).exists()}
    yield "\n".join(log + ["Rendering SOMA preview ..."]), _status_table(wd), files, None
    yield "\n".join(log), _status_table(wd), files, pipeline.preview_video(Path(wd), motions)


# ----------------------------------------------------------------- step 3
def step_compare(wd, trial_up, trial_sample, static_up, static_sample, sync_ref, progress=gr.Progress()):
    if not wd:
        raise gr.Error("Run steps ① and ② first.")
    trial, static = _resolve(trial_up, trial_sample), _resolve(static_up, static_sample)
    motions = {k: SomaMotion.load(output_path(wd, k)) for k in REGISTRY if output_path(wd, k).exists()}
    if not motions:
        raise gr.Error("No model has a SOMA result yet; run step ② first.")
    progress(0.1, desc="Mocap → SOMA (static calibration + trial fit)")
    mocap = pipeline.mocap_to_soma(trial, static, log=lambda m: None)
    progress(0.6, desc="Auto sync + evaluation")
    lines = []
    cmp = pipeline.compare(Path(wd), motions, mocap, None if sync_ref == "auto" else sync_ref,
                           log=lines.append)
    sync_md = "\n".join(f"- {l}" for l in lines) + \
        f"\n- mocap fit: marker residual {mocap.meta.get('marker_rms_mm', float('nan')):.1f} mm, " \
        f"joint-centre residual {mocap.meta.get('joint_centre_rms_mm', float('nan')):.1f} mm, " \
        f"{len(mocap.meta.get('markers_used', []))} markers, compared frames: {len(cmp.times)}"
    return (sync_md, pd.DataFrame(cmp.table), pd.DataFrame(cmp.angle_table), cmp.figures["angles"],
            cmp.figures["per_joint"], cmp.figures["error_time"], cmp.figures["sync"], cmp.video)


# --------------------------------------------------------------------- UI
def build() -> gr.Blocks:
    videos = _samples(("*.mp4", "*.mov", "*.avi"))
    c3ds = _samples(("*.c3d",))
    static_default = next((v for _, v in c3ds if "static" in Path(v).name.lower()), None)
    with gr.Blocks(title="Video vs Mocap (SOMA)") as demo:
        gr.Markdown("# Video motion capture vs. marker mocap, unified in SOMA")
        wd = gr.State("")
        with gr.Tab("1. Video → SOMA"):
            with gr.Row():
                with gr.Column(scale=1):
                    v_up = gr.Video(label="Upload video", sources=["upload"])
                    v_sample = gr.Dropdown(videos, label="Or a sample from data/", value=None)
                    b_prep = gr.Button("① Preprocess: normalise + detect people", variant="primary")
                    track = gr.Dropdown([], label="Subject (track ID)")
                    track_md = gr.Markdown()
                with gr.Column(scale=1):
                    preview = gr.Image(label="Detected people (red = selected)", interactive=False)
            with gr.Row():
                with gr.Column(scale=1):
                    models = gr.CheckboxGroup(_model_choices(), value=["gem-x"], label="Models")
                    with gr.Row():
                        gpu = gr.Dropdown(["auto", "0", "1"], value="auto", label="GPU")
                        rerun = gr.Checkbox(False, label="Ignore cache and rerun")
                    b_run = gr.Button("② Run selected models → SOMA", variant="primary")
                with gr.Column(scale=1):
                    status = gr.Dataframe(_status_table(None), label="Model status", interactive=False)
            run_log = gr.Textbox(label="Run log", lines=6)
            with gr.Row():
                soma_preview = gr.Video(label="SOMA preview: source | front | side", scale=2)
                soma_files = gr.File(label="SOMA results (.npz)", file_count="multiple", scale=1)

        with gr.Tab("2. Mocap comparison"):
            with gr.Row():
                t_up = gr.File(label="Trial C3D", file_types=[".c3d"])
                t_sample = gr.Dropdown([c for c in c3ds if c[1] != static_default], label="Or a sample trial")
                s_up = gr.File(label="Static C3D (body shape and marker calibration)", file_types=[".c3d"])
                s_sample = gr.Dropdown(c3ds, value=static_default, label="Or a sample static trial")
            with gr.Row():
                sync_ref = gr.Dropdown(["auto", *REGISTRY], value="auto", label="Sync reference model")
                b_cmp = gr.Button("③ Mocap → SOMA + auto sync + evaluate", variant="primary")
            sync_md = gr.Markdown()
            table = gr.Dataframe(label="Summary metrics (mm, deg)", interactive=False)
            cmp_video = gr.Video(label="Side by side: source | front | side (grey = mocap)")
            p_angles = gr.Plot(label="Joint angles")
            with gr.Row():
                p_joint = gr.Plot(label="Per-joint error")
                p_sync = gr.Plot(label="Synchronisation")
            p_time = gr.Plot(label="Error over time")
            angle_table = gr.Dataframe(label="Joint-angle metrics", interactive=False)

        b_prep.click(step_prepare, [v_up, v_sample], [wd, preview, track, track_md, status])
        track.change(step_pick, [wd, track], [preview])
        b_run.click(step_run, [wd, models, gpu, rerun], [run_log, status, soma_files, soma_preview])
        b_cmp.click(step_compare, [wd, t_up, t_sample, s_up, s_sample, sync_ref],
                    [sync_md, table, angle_table, p_angles, p_joint, p_time, p_sync, cmp_video])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve on the LAN")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-share", dest="share", action="store_false", help="local only, no public gradio.live link")
    a = ap.parse_args()
    build().queue().launch(theme=gr.themes.Base(), server_name=a.host, server_port=a.port, share=a.share,
                           allowed_paths=[str(Path(__file__).resolve().parents[1] / "outputs")])
