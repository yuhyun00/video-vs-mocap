"""Colours for every plot and render (one place, fixed per entity).

Model colours follow the validated categorical order of the reference palette
and are bound to the model key, never to rank, so filtering models never
repaints the survivors. The mocap reference is drawn in neutral ink, not a
categorical hue.
"""

from __future__ import annotations

MODEL_ORDER = ("gem-x", "sam3d-body", "sam-body4d", "duomo", "flowhmr")
_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

MODEL_COLORS = {k: _CATEGORICAL[i] for i, k in enumerate(MODEL_ORDER)}
REFERENCE_COLOR = "#52514e"  # mocap (neutral secondary ink)
TEXT = "#0b0b0b"
MUTED = "#898781"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def color(source: str) -> str:
    return REFERENCE_COLOR if source == "mocap" else MODEL_COLORS.get(source, _CATEGORICAL[-1])


def bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def rgb(hex_color: str) -> tuple[int, int, int]:
    b, g, r = bgr(hex_color)
    return r, g, b
