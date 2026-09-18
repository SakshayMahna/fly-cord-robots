"""Render the T1/T2/T3 CPG circuits + all leg motor neurons (MaleCNS) as a
3D skeleton plot — final Phase 1 circuit map.

Status as of 2026-09-19 (see CHANGELOG): T1's circuit is Pugliese's
published, validated result. T2/T3's circuits were identified by us from
static connectivity alone (`identify_all_legs.py`), then confirmed
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
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
import pandas as pd

from fly_robot.connectome.client import get_client

COLOR_MOTOR_NEURON = "#8a8a86"

# T1 (published, Pugliese's own validated circuit): saturated warm hues.
COLOR_T1_DN = "#e34948"
COLOR_T1_EXCIT = "#2a78d6"
COLOR_T1_INHIB = "#eb6834"

# T2/T3 (identified by us, confirmed via their real published simulation
# output — see CHANGELOG 2026-09-19): distinct cool hues, purely to keep
# the three segments visually separable in the render, NOT to imply lower
# confidence — these are confirmed too, not "candidates" anymore.
COLOR_CAND_DN = "#9085e9"      # violet
COLOR_CAND_EXCIT = "#1baf7a"   # aqua/green
COLOR_CAND_INHIB = "#eda100"   # yellow

ROLE_COLOR_BY_LEG = {
    ("T1", "command_DN"): COLOR_T1_DN,
    ("T1", "CPG_excit_hub"): COLOR_T1_EXCIT,
    ("T1", "CPG_excit2"): COLOR_T1_EXCIT,
    ("T1", "CPG_inhib"): COLOR_T1_INHIB,
}
CANDIDATE_ROLE_COLOR = {
    "command_DN": COLOR_CAND_DN,
    "CPG_excit_hub": COLOR_CAND_EXCIT,
    "CPG_excit2": COLOR_CAND_EXCIT,
    "CPG_inhib": COLOR_CAND_INHIB,
}

LEGEND_ENTRIES = [
    ("T1 DNg100 (Pugliese, published)", COLOR_T1_DN),
    ("T1 CPG (Pugliese, published)", COLOR_T1_EXCIT),
    ("T1 CPG, inhibitory (Pugliese, published)", COLOR_T1_INHIB),
    ("T2/T3 DNg100 copy (confirmed, see CHANGELOG)", COLOR_CAND_DN),
    ("T2/T3 CPG (confirmed, see CHANGELOG)", COLOR_CAND_EXCIT),
    ("T2/T3 CPG, inhibitory (confirmed, see CHANGELOG)", COLOR_CAND_INHIB),
    ("leg motor neurons (all legs)", COLOR_MOTOR_NEURON),
]


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
            key = (row["leg"], row["role"])
            colors[bid] = ROLE_COLOR_BY_LEG.get(key) or CANDIDATE_ROLE_COLOR[row["role"]]
            is_highlight[bid] = True

    color_list = [colors[n.id] for n in skeletons]
    highlight_list = [is_highlight[n.id] for n in skeletons]
    return skeletons, color_list, highlight_list


def _vnc_crop_bounds(skeletons, highlight_list, margin_frac=0.15):
    """See visualize_circuit.py — same caveat: view=('x','-z') inverts the
    displayed axis but doesn't negate the data, so bounds must use raw z."""
    vnc_skels = [s for s, hl in zip(skeletons, highlight_list) if not hl]
    xs = pd.concat([s.nodes["x"] for s in vnc_skels])
    zs = pd.concat([s.nodes["z"] for s in vnc_skels])
    x_margin = (xs.max() - xs.min()) * margin_frac
    z_margin = (zs.max() - zs.min()) * margin_frac
    return (xs.min() - x_margin, xs.max() + x_margin,
            zs.min() - z_margin, zs.max() + z_margin)


def render(skeletons, color_list, highlight_list, out_dir: Path, theme: str):
    bg = "#1a1a19" if theme == "dark" else "#fcfcfb"
    fg = "#ffffff" if theme == "dark" else "#0b0b0b"

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

    x0, x1, z0, z1 = _vnc_crop_bounds(skeletons, highlight_list)
    ax.set_xlim(x0, x1)
    ax.set_ylim(z1, z0)  # inverted, see _vnc_crop_bounds docstring

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
