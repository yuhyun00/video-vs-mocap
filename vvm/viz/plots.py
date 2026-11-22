"""Plotly figures for the comparison report (rendered by Gradio or saved as HTML)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vvm.eval.angles import ANGLE_NAMES
from vvm.eval.metrics import Evaluation
from vvm.models import display_name
from vvm.soma.skeleton import ANATOMY, EVAL_JOINTS
from vvm.sync.auto_sync import SyncResult
from vvm.viz.style import GRID, MUTED, REFERENCE_COLOR, SEQUENTIAL_BLUE, SURFACE, TEXT, color


def _layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=TEXT)), height=height,
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=TEXT, size=12),
                      hovermode="x unified", margin=dict(l=60, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    return fig


def angle_curves(times: np.ndarray, evals: dict[str, Evaluation], angles=ANGLE_NAMES[:8]) -> go.Figure:
    """Small multiples: one panel per angle, mocap vs every model."""
    rows = (len(angles) + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=[f"{a} (deg)" for a in angles],
                        shared_xaxes=True, vertical_spacing=0.07)
    for k, name in enumerate(angles):
        r, c = k // 2 + 1, k % 2 + 1
        ai = ANGLE_NAMES.index(name)
        for key, ev in evals.items():
            fig.add_trace(go.Scatter(x=times, y=ev.series["angles_pred"][:, ai], name=display_name(key),
                                     legendgroup=key, showlegend=(k == 0), mode="lines",
                                     line=dict(color=color(key), width=2)), row=r, col=c)
        ref = next(iter(evals.values())).series["angles_ref"][:, ai]
        fig.add_trace(go.Scatter(x=times, y=ref, name="Mocap (reference)", legendgroup="mocap",
                                 showlegend=(k == 0), mode="lines",
                                 line=dict(color=REFERENCE_COLOR, width=2, dash="dash")), row=r, col=c)
    fig.update_xaxes(title_text="video time (s)", row=rows)
    return _layout(fig, "Lower-limb joint angles: video models vs mocap", 230 * rows + 80)


def per_joint_heatmap(evals: dict[str, Evaluation]) -> go.Figure:
    keys = list(evals)
    z = np.array([[evals[k].per_joint[j] for k in keys] for j in EVAL_JOINTS])
    labels = [ANATOMY.get(j, j) for j in EVAL_JOINTS]
    scale = [[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)]
    fig = go.Figure(go.Heatmap(z=z, x=[display_name(k) for k in keys], y=labels, colorscale=scale,
                               text=np.round(z).astype(int), texttemplate="%{text}", xgap=2, ygap=2,
                               hovertemplate="%{y} · %{x}: %{z:.1f} mm<extra></extra>",
                               colorbar=dict(title="mm")))
    fig.update_yaxes(autorange="reversed")
    fig = _layout(fig, "Per-joint position error (root-aligned MPJPE)", 40 + 26 * len(labels) + 80)
    fig.update_layout(hovermode="closest")
    return fig


def error_over_time(times: np.ndarray, evals: dict[str, Evaluation]) -> go.Figure:
    fig = go.Figure()
    for key, ev in evals.items():
        fig.add_trace(go.Scatter(x=times, y=ev.series["err_pa"].mean(1), name=display_name(key), mode="lines",
                                 line=dict(color=color(key), width=2)))
    fig.update_xaxes(title_text="video time (s)")
    fig.update_yaxes(title_text="PA-MPJPE (mm)")
    return _layout(fig, "Per-frame pose error (Procrustes aligned)", 320)


def sync_curve(results: dict[str, SyncResult]) -> go.Figure:
    fig = go.Figure()
    for key, s in results.items():
        fig.add_trace(go.Scatter(x=s.lags, y=s.scores, name=display_name(key), mode="lines",
                                 line=dict(color=color(key), width=2)))
        fig.add_trace(go.Scatter(x=[s.offset], y=[s.score], mode="markers", showlegend=False,
                                 marker=dict(color=color(key), size=10, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{display_name(key)}: offset %{{x:.3f}} s, r=%{{y:.3f}}"
                                               "<extra></extra>"))
    fig.update_xaxes(title_text="offset (s): video time = mocap time + offset")
    fig.update_yaxes(title_text="mean joint-angle correlation")
    return _layout(fig, "Automatic synchronisation (cross-correlation)", 320)
