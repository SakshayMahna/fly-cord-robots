"""Render the T1/T2/T3 CPG circuits + all leg motor neurons (MaleCNS) as a
3D skeleton plot — final Phase 1 circuit map.

Status as of 2026-09-19 (see CHANGELOG): T1's circuit is Pugliese's
published, validated result. T2/T3's circuits were identified by us from
static connectivity alone (`identify_all_leg_circuits.py`), then confirmed
rhythmically active by checking these exact neurons against Pugliese's
real published full-VNC simulation output (104-124/128 replicates active,
scores in the same range as T1). All three segments are now confirmed,
not hypothesized — colors are still kept distinct per segment (not by
confidence level) so the three leg circuits stay visually distinguishable
in the render.

Anatomical sanity check this plot provides: three separate bilateral
neuropil clusters of motor neurons along the VNC (T1/T2/T3, in that
anterior-to-posterior order), each with its own small CPG cluster sitting
inside it — not, say, T2/T3 "CPG" neurons that turn out to sit outside leg
neuropil entirely (which would suggest a mis-identification).

Color encoding: one distinct hue per LEG SEGMENT (T1/T2/T3), each shaded
into 3 tints for the 3 CPG roles (command DN darkest, excitatory hub
mid-tone, inhibitory lightest). Segment identity is the primary thing a
viewer needs to read off this plot (there was an earlier version of this
render, still in `docs/logs/2026-09-18.md`, that colored by
confirmed-vs-hypothesis status instead — that distinction no longer
applies now that all three segments are confirmed, and it had the side
effect of making T2 and T3 indistinguishable from each other by color).
"""

import argparse
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
import pandas as pd

from fly_robot.connectome.client import get_client
from fly_robot.analysis.render_utils import theme_colors, vnc_crop_bounds

COLOR_MOTOR_NEURON = "#8a8a86"

# One base hue per leg segment (first three categorical slots from the
# project's palette reference — validated for all-pairs colorblind-safe
# comparison, which matters here since a viewer needs to tell all three
# segments apart at a glance, not just adjacent ones).
SEGMENT_BASE_HUE = {
    "T1": "#2a78d6",  # blue
    "T2": "#eb6834",  # orange
    "T3": "#1baf7a",  # aqua
}


def _shade(hex_color: str, lightness_delta: float) -> str:
    """Lighten (positive) or darken (negative) a hex color in HSL space,
    keeping hue/saturation fixed — used to derive the 3 per-segment role
    tints from one base hue."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + lightness_delta))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


ROLE_LIGHTNESS_DELTA = {
    "command_DN": -0.12,     # darkest — the descending command neuron
    "CPG_excit_hub": 0.0,    # base hue
    "CPG_excit2": 0.0,       # same tint as the hub (both excitatory)
    "CPG_inhib": 0.20,       # lightest — the inhibitory neuron
}

ROLE_COLOR_BY_LEG = {
    (leg, role): _shade(hue, delta)
    for leg, hue in SEGMENT_BASE_HUE.items()
    for role, delta in ROLE_LIGHTNESS_DELTA.items()
}

LEGEND_ENTRIES = [
    (f"{leg} DNg100" if role == "command_DN" else
     f"{leg} CPG, inhibitory" if role == "CPG_inhib" else
     f"{leg} CPG (excitatory)",
     ROLE_COLOR_BY_LEG[(leg, role)])
    for leg in ("T1", "T2", "T3")
    for role in ("command_DN", "CPG_excit_hub", "CPG_inhib")
] + [("leg motor neurons (all legs)", COLOR_MOTOR_NEURON)]


def load_circuit(circuit_csv: str) -> pd.DataFrame:
    df = pd.read_csv(circuit_csv)
    return df[df["malecns_bodyId"].notna()].copy()


def fetch_and_color(df: pd.DataFrame):
    client = get_client()
    neu.set_default_client(client)

    # A DN/CPG neuron can appear once per (leg, side) but only needs
    # fetching once per malecns_bodyId — drop_duplicates keeps the color
    # assignment simple (a given bodyId always maps to one role/leg).
    df = df.drop_duplicates(subset=["malecns_bodyId"])
    ids = df["malecns_bodyId"].astype(int).tolist()
    skeletons = neu.fetch_skeletons(ids, missing_swc="raise")

    colors, is_highlight = {}, {}
    for _, row in df.iterrows():
        bid = int(row["malecns_bodyId"])
        if row["role"] == "leg_motor_neuron":
            colors[bid] = COLOR_MOTOR_NEURON
            is_highlight[bid] = False
        else:
            colors[bid] = ROLE_COLOR_BY_LEG[(row["leg"], row["role"])]
            is_highlight[bid] = True

    color_list = [colors[n.id] for n in skeletons]
    highlight_list = [is_highlight[n.id] for n in skeletons]
    return skeletons, color_list, highlight_list


def render(skeletons, color_list, highlight_list, out_dir: Path, theme: str):
    bg, fg = theme_colors(theme)

    fig, ax = navis.plot2d(
        skeletons, color=color_list, linewidth=0.5,
        method="2d", view=("x", "-z"), figsize=(9, 10),
    )

    highlight_skels = [s for s, hl in zip(skeletons, highlight_list) if hl]
    highlight_colors = [c for c, hl in zip(color_list, highlight_list) if hl]
    navis.plot2d(
        highlight_skels, color=highlight_colors, linewidth=2.0,
        method="2d", view=("x", "-z"), ax=ax,
    )

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    x0, x1, z0, z1 = vnc_crop_bounds(skeletons, highlight_list, margin_frac=0.15)
    ax.set_xlim(x0, x1)
    ax.set_ylim(z1, z0)  # inverted, see vnc_crop_bounds docstring

    handles = [plt.Line2D([0], [0], color=c, lw=2.5, label=label)
               for label, c in LEGEND_ENTRIES]
    ax.legend(handles=handles, loc="lower center", frameon=False,
              fontsize=7.5, labelcolor=fg, ncol=2, bbox_to_anchor=(0.5, -0.08))

    out_path = out_dir / f"all_legs_circuit_{theme}.png"
    fig.savefig(out_path, dpi=200, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-csv", default="data/circuit_map/all_legs_circuit.csv")
    parser.add_argument("--out-dir", default="media")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    circuit_df = load_circuit(args.circuit_csv)
    n_cpg = (circuit_df["role"] != "leg_motor_neuron").sum()
    n_mn = (circuit_df["role"] == "leg_motor_neuron").sum()
    print(f"Rendering {len(circuit_df)} rows ({n_cpg} CPG/DN role-instances, {n_mn} motor neurons)")

    skels, colors, highlights = fetch_and_color(circuit_df)
    print(f"Fetched {len(skels)} unique skeletons")

    for theme in ("dark", "light"):
        path = render(skels, colors, highlights, out_dir, theme)
        print(f"Wrote {path}")
