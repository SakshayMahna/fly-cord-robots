"""Render the DNg100 T1 walking circuit (MaleCNS) as a 3D skeleton plot.

Purpose: this is a Phase 0 sanity check as much as it is video footage. If
the neurons we resolved via `identify_circuit.py` are correctly identified,
the leg motor neurons should cluster spatially in the T1 (front) leg
neuropil/nerve, and the CPG neurons + DNg100 should sit in the central VNC
neuropil, not scattered randomly.

Color roles (not connectome data, purely a plotting choice):
  - leg motor neurons: muted gray (background structure, 138 neurons)
  - DNg100: red        (command/descending neuron)
  - CPG excitatory:    blue    (IN17A001, INXXX466 — acetylcholine)
  - CPG inhibitory:    orange  (IN16B036 — glutamate)
These follow the categorical hues from the project's palette reference,
using only 3 saturated colors against a neutral background to stay clear
of adjacent-pair confusability.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
import pandas as pd

from fly_robot.connectome.client import get_client

COLOR_MOTOR_NEURON = "#8a8a86"
COLOR_DN = "#e34948"
COLOR_CPG_EXCITATORY = "#2a78d6"
COLOR_CPG_INHIBITORY = "#eb6834"

CPG_ROLE_COLORS = {
    "DNg100": COLOR_DN,
    "IN17A001": COLOR_CPG_EXCITATORY,
    "INXXX466": COLOR_CPG_EXCITATORY,
    "IN16B036": COLOR_CPG_INHIBITORY,
}


def load_circuit(circuit_csv: str) -> pd.DataFrame:
    df = pd.read_csv(circuit_csv)
    return df[df["malecns_bodyId"].notna()].copy()


def fetch_and_color(df: pd.DataFrame):
    client = get_client()
    neu.set_default_client(client)

    ids = df["malecns_bodyId"].astype(int).tolist()
    skeletons = neu.fetch_skeletons(ids, missing_swc="raise")

    colors = {}
    is_highlight = {}
    for _, row in df.iterrows():
        bid = int(row["malecns_bodyId"])
        if row["role"] == "CPG_or_command":
            colors[bid] = CPG_ROLE_COLORS.get(row["type_x"], COLOR_DN)
            is_highlight[bid] = True
        else:
            colors[bid] = COLOR_MOTOR_NEURON
            is_highlight[bid] = False

    color_list = [colors[n.id] for n in skeletons]
    highlight_list = [is_highlight[n.id] for n in skeletons]
    return skeletons, color_list, highlight_list


LEGEND_ENTRIES = [
    ("DNg100 (command)", COLOR_DN),
    ("CPG, excitatory", COLOR_CPG_EXCITATORY),
    ("CPG, inhibitory", COLOR_CPG_INHIBITORY),
    ("T1 leg motor neurons", COLOR_MOTOR_NEURON),
]


def _vnc_crop_bounds(skeletons, highlight_list, margin_frac=0.35):
    """Bounding box (in plot x / z coords) of the non-highlighted (motor
    neuron) skeletons — i.e. the VNC leg neuropil, excluding the brain
    arbor that only DNg100 has. Expanded by a margin so the thin CPG
    neurons (which sit inside this volume) aren't clipped at the edge.

    Note: `view=("x", "-z")` in navis.plot2d inverts the *displayed* axis
    direction, it does not negate the underlying data — so bounds here
    must be computed from raw `z`, not `-z`, to match what's on screen.
    """
    vnc_skels = [s for s, hl in zip(skeletons, highlight_list) if not hl]
    xs, zs = [], []
    for s in vnc_skels:
        nodes = s.nodes
        xs.append(nodes["x"])
        zs.append(nodes["z"])
    import pandas as pd  # local import to avoid polluting module namespace
    xs = pd.concat(xs)
    zs = pd.concat(zs)
    x_margin = (xs.max() - xs.min()) * margin_frac
    z_margin = (zs.max() - zs.min()) * margin_frac
    return (xs.min() - x_margin, xs.max() + x_margin,
            zs.min() - z_margin, zs.max() + z_margin)


def render(skeletons, color_list, highlight_list, out_dir: Path, theme: str,
           zoom_vnc: bool):
    bg = "#1a1a19" if theme == "dark" else "#fcfcfb"
    fg = "#ffffff" if theme == "dark" else "#0b0b0b"

    # Base pass: all neurons, thin lines.
    fig, ax = navis.plot2d(
        skeletons,
        color=color_list,
        linewidth=0.6,
        method="2d",
        view=("x", "-z"),
        figsize=(10, 8),
    )

    # Overlay pass: redraw just the CPG/DN neurons thicker so the tiny
    # interneurons are actually visible against the much larger motor
    # neuron dendritic fields.
    highlight_skels = [s for s, hl in zip(skeletons, highlight_list) if hl]
    highlight_colors = [c for c, hl in zip(color_list, highlight_list) if hl]
    navis.plot2d(
        highlight_skels,
        color=highlight_colors,
        linewidth=2.2,
        method="2d",
        view=("x", "-z"),
        ax=ax,
    )

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    suffix = theme
    if zoom_vnc:
        x0, x1, z0, z1 = _vnc_crop_bounds(skeletons, highlight_list)
        ax.set_xlim(x0, x1)
        # y-axis here is displaying z inverted (view=("x","-z")); the
        # existing full-range ylim is (large, small), so match that
        # orientation rather than passing (z0, z1) = (small, large).
        ax.set_ylim(z1, z0)
        suffix = f"vnc_zoom_{theme}"

    handles = [
        plt.Line2D([0], [0], color=c, lw=2.5, label=label)
        for label, c in LEGEND_ENTRIES
    ]
    legend = ax.legend(
        handles=handles, loc="lower center", frameon=False,
        fontsize=9, labelcolor=fg, ncol=2,
        bbox_to_anchor=(0.5, -0.05),
    )

    out_path = out_dir / f"t1_walking_circuit_{suffix}.png"
    fig.savefig(out_path, dpi=200, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuit-csv", default="data/circuit_map/t1_front_leg_circuit.csv"
    )
    parser.add_argument("--out-dir", default="media")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    circuit_df = load_circuit(args.circuit_csv)
    print(f"Rendering {len(circuit_df)} neurons "
          f"({(circuit_df['role'] == 'CPG_or_command').sum()} CPG/command, "
          f"{(circuit_df['role'] == 'leg_motor_neuron').sum()} motor neurons)")

    skels, colors, highlights = fetch_and_color(circuit_df)

    for theme in ("dark", "light"):
        path = render(skels, colors, highlights, out_dir, theme, zoom_vnc=False)
        print(f"Wrote {path}")
        path = render(skels, colors, highlights, out_dir, theme, zoom_vnc=True)
        print(f"Wrote {path}")
