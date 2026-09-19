"""Shared helpers for the circuit-render scripts (`visualize_t1_circuit.py`,
`visualize_all_leg_circuits.py`) — extracted because both independently duplicated
the same VNC-cropping and theme logic.
"""

import pandas as pd

THEME_COLORS = {
    "dark": ("#1a1a19", "#ffffff"),
    "light": ("#fcfcfb", "#0b0b0b"),
}


def theme_colors(theme: str) -> tuple[str, str]:
    """Returns (background, foreground/text) hex colors for a theme."""
    return THEME_COLORS[theme]


def vnc_crop_bounds(skeletons, highlight_mask, margin_frac: float = 0.35):
    """Bounding box (in plot x / z coords) of the non-highlighted (motor
    neuron) skeletons — i.e. the VNC leg neuropil, excluding the brain
    arbor that only DNg100 has. Expanded by a margin so the thin CPG
    neurons (which sit inside this volume) aren't clipped at the edge.

    Note: `view=("x", "-z")` in navis.plot2d inverts the *displayed* axis
    direction, it does not negate the underlying data — so bounds here
    must be computed from raw `z`, not `-z`, to match what's on screen.
    Callers should set `ax.set_ylim(z1, z0)` (swapped) to match.
    """
    vnc_skels = [s for s, hl in zip(skeletons, highlight_mask) if not hl]
    xs = pd.concat([s.nodes["x"] for s in vnc_skels])
    zs = pd.concat([s.nodes["z"] for s in vnc_skels])
    x_margin = (xs.max() - xs.min()) * margin_frac
    z_margin = (zs.max() - zs.min()) * margin_frac
    return (xs.min() - x_margin, xs.max() + x_margin,
            zs.min() - z_margin, zs.max() + z_margin)
