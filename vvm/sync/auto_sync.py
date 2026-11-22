"""Automatic time alignment of video-model motion with mocap motion.

Cross-correlates joint angles (independent of either world frame) over all
offsets, refined to sub-sample precision. Convention: ``t_video = t_mocap + offset``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vvm.eval.angles import LOWER_LIMB_ANGLES, angle_matrix
from vvm.soma.motion import SomaMotion

# knee and hip flexion are the most reliable channels for walking; ankle and
# abduction are added with lower weight.
_CHANNELS = LOWER_LIMB_ANGLES
_WEIGHTS = np.array([1.0, 1.0, 0.3, 0.3, 1.0, 1.0, 0.5, 0.5])
# below this flexion std (quiet standing) correlation peaks are noise
MIN_MOTION_DEG = 3.0
_FLEXION = [i for i, n in enumerate(_CHANNELS) if "flexion" in n]


@dataclass
class SyncResult:
    offset: float  # seconds, t_video = t_mocap + offset
    score: float  # mean correlation at the optimum (-1..1)
    peak_ratio: float  # best / second-best separated peak; >1.2 is a confident match
    lags: np.ndarray
    scores: np.ndarray
    per_channel: dict[str, float]
    motion_deg: float = float("nan")  # std of the mocap hip/knee flexion (signal strength)

    @property
    def reliable(self) -> bool:
        return self.motion_deg >= MIN_MOTION_DEG

    def describe(self) -> str:
        if not self.reliable:
            return (f"mocap trial has almost no joint motion (flexion std {self.motion_deg:.1f} deg): "
                    f"the time offset cannot be determined from motion; offset {self.offset:+.3f} s is arbitrary")
        # periodic motion (gait) produces near-equal peaks one cycle apart, so the
        # peak ratio is reported but confidence is judged on the correlation itself
        conf = "high" if self.score > 0.7 else "medium" if self.score > 0.5 else "low"
        return (f"offset = {self.offset:+.3f} s (video = mocap + offset), "
                f"corr = {self.score:.3f}, peak ratio = {self.peak_ratio:.2f} -> {conf} confidence")


def _signals(m: SomaMotion, rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ang = angle_matrix(m.joints, m.joint_names, _CHANNELS)
    t0, t1 = m.times[m.valid][0], m.times[m.valid][-1]
    grid = np.arange(t0, t1, 1.0 / rate)
    ok = m.valid & np.isfinite(ang).all(1)
    sig = np.stack([np.interp(grid, m.times[ok], ang[ok, k]) for k in range(ang.shape[1])], -1)
    valid = np.interp(grid, m.times, ok.astype(float)) > 0.99
    return grid, sig, valid


def _corr_at(a, va, b, vb, shift: int, min_overlap: int) -> np.ndarray | None:
    """Per-channel Pearson r between a[i+shift] and b[i] (b = mocap)."""
    lo, hi = max(0, -shift), min(len(b), len(a) - shift)
    if hi - lo < min_overlap:
        return None
    x, y = a[lo + shift:hi + shift], b[lo:hi]
    m = va[lo + shift:hi + shift] & vb[lo:hi]
    if m.sum() < min_overlap:
        return None
    x, y = x[m], y[m]
    x = x - x.mean(0)
    y = y - y.mean(0)
    den = np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0))
    return np.where(den > 1e-9, (x * y).sum(0) / np.maximum(den, 1e-9), 0.0)


def auto_sync(video: SomaMotion, mocap: SomaMotion, rate: float = 50.0,
              min_overlap_s: float = 3.0, search: tuple[float, float] | None = None) -> SyncResult:
    tv, sv, vv = _signals(video, rate)
    tm, sm, vm = _signals(mocap, rate)
    w = _WEIGHTS / _WEIGHTS.sum()
    min_overlap = int(min_overlap_s * rate)
    # shift k means video sample (i + k) pairs with mocap sample i
    k_lo, k_hi = -len(sm) + min_overlap, len(sv) - min_overlap
    if search is not None:  # restrict offsets (seconds) if the user knows roughly
        k_lo = max(k_lo, int(np.floor((search[0] - tv[0] + tm[0]) * rate)))
        k_hi = min(k_hi, int(np.ceil((search[1] - tv[0] + tm[0]) * rate)))
    ks = np.arange(k_lo, k_hi + 1)
    scores = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        r = _corr_at(sv, vv, sm, vm, int(k), min_overlap)
        if r is not None:
            scores[i] = float((r * w).sum())
    if np.all(np.isnan(scores)):
        raise ValueError("motions do not overlap enough to synchronise")
    best = int(np.nanargmax(scores))
    # parabolic refinement
    frac = 0.0
    if 0 < best < len(ks) - 1 and np.isfinite(scores[best - 1:best + 2]).all():
        y0, y1, y2 = scores[best - 1:best + 2]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    k_best = ks[best] + frac
    offset = tv[0] + k_best / rate - tm[0]
    # second peak at least 0.5 s away
    far = np.abs(ks - ks[best]) > 0.5 * rate
    second = np.nanmax(scores[far]) if np.any(far & np.isfinite(scores)) else np.nan
    ratio = float(scores[best] / second) if np.isfinite(second) and second > 0 else float("inf")
    r = _corr_at(sv, vv, sm, vm, int(ks[best]), min_overlap)
    lags = tv[0] + ks / rate - tm[0]
    motion = float(sm[vm][:, _FLEXION].std(0).max())
    return SyncResult(float(offset), float(scores[best]), ratio, lags, scores,
                      {n: float(x) for n, x in zip(_CHANNELS, r)}, motion)


def combine(results: dict[str, SyncResult]) -> SyncResult:
    """Consensus offset: maximum of the mean correlation curve over models.

    All results come from the same video, so their lag grids coincide up to
    each model's valid range; curves are averaged where they overlap.
    """
    lags = min((r.lags for r in results.values()), key=len)
    stack = np.stack([np.interp(lags, r.lags, r.scores, left=np.nan, right=np.nan) for r in results.values()])
    mean = np.nanmean(stack, 0)
    best = int(np.nanargmax(mean))
    frac = 0.0
    if 0 < best < len(lags) - 1 and np.isfinite(mean[best - 1:best + 2]).all():
        y0, y1, y2 = mean[best - 1:best + 2]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    step = float(np.median(np.diff(lags)))
    far = np.abs(lags - lags[best]) > 0.5
    second = np.nanmax(mean[far]) if np.any(far & np.isfinite(mean)) else np.nan
    return SyncResult(float(lags[best] + frac * step), float(mean[best]),
                      float(mean[best] / second) if np.isfinite(second) and second > 0 else float("inf"),
                      lags, mean, {}, min(r.motion_deg for r in results.values()))


def common_grid(video: SomaMotion, mocap: SomaMotion, offset: float) -> np.ndarray:
    """Video frame times that fall inside the (shifted) mocap trial."""
    t0 = mocap.times[mocap.valid][0] + offset
    t1 = mocap.times[mocap.valid][-1] + offset
    t = video.times[video.valid]
    return t[(t >= t0) & (t <= t1)]
