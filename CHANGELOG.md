# Findings changelog

Chronological log of what we learned/decided each session. This is the
source material for the video script — write it so a future read explains
*why*, not just *what*.

## 2026-09-18 — Dataset choice: MaleCNS over MANC, and locating the DNg100 circuit in it

**Decision:** use MaleCNS (`male-cns:v1.0`, neuprint server
`neuprint.janelia.org`) as our connectome going forward, not MANC.

**Why MaleCNS, and what changes:**
- MANC (2023) is nerve-cord-only, ~23k neurons. Pugliese et al.'s CPG model
  (github.com/smpuglie/Pugliese_cpg_2025) was built on MANC and only ever
  covers the T1 (front-leg) DN→motor-neuron circuit — it does not solve
  our Phase 1 need for all six legs either way.
- MaleCNS (released June 2026, Cell paper Sept 2026) is a different,
  newer specimen: brain + VNC imaged as one continuous volume (including
  the neck connective MANC lacks), ~166,700 neurons. This matters for us
  because it's a superset — same VNC circuitry, plus the brain, which
  keeps later phases (real descending drive instead of synthetic tonic
  stimulation) open without a second dataset migration.
- Since we have to build our own multi-leg circuit extraction regardless
  of dataset, MaleCNS costs us nothing extra structurally and gets us the
  more complete dataset now.

**Caveat (still open):** Phase 0's original "done when" criterion was to
qualitatively match Pugliese's *published rhythm figures*. Those figures
were generated from MANC-derived weights. Running our model on
MaleCNS-derived weights is not the same numerical circuit, even though
the neuron identities match — connectivity/synapse counts between MANC
and MaleCNS for the same neuron pair can differ (different specimen, own
proofreading pass). We have not yet decided whether Phase 0's validation
step should also include a literal MANC run for direct comparison.

**How we identified the circuit in MaleCNS:**
MaleCNS neuron records carry a `mancBodyid` field — a per-neuron
cross-reference to the corresponding MANC specimen body, computed by
Janelia's own annotation team, not by us. We used this instead of
matching on the `type` string, because it's a hard per-neuron reference
rather than a name that could coincidentally collide.

Pugliese's own MANC body IDs for the core circuit are only recoverable
by reading their experiment configs, not published names:
`configs/experiment/DNg100_Stim_CoreCPG.yaml` sets
`keepOnly: [31, 277, 617, 1167]` — these are *row indices* into
`data/manc t1 connectome data/wTable_20250813_DNtoMN_unsorted_withModules.csv`,
which resolve to:

| MANC bodyId | type      | role         | predicted NT   |
|-------------|-----------|--------------|----------------|
| 10093       | DNg100    | command (DN) | acetylcholine  |
| 10707       | IN17A001  | CPG, excit.  | acetylcholine  |
| 11751       | INXXX466  | CPG, excit.  | acetylcholine  |
| 13905       | IN16B036  | CPG, inhib.  | glutamate      |

This matches the paper's description of a minimal 3-neuron CPG (2
excitatory + 1 inhibitory) driven by DNg100.

Querying MaleCNS by `mancBodyid` for these 4 plus all 148 rows tagged
`class == "motor neuron"` in the same table (`fly_robot/connectome/identify_circuit.py`):

- **4/4 CPG+command neurons resolved 1:1**, same `type` string on both
  sides, both `status == Traced`.
- **138/148 T1 leg motor neurons resolved** (see script output; unresolved
  ones span several joint modules — coxa stance, tarsus control, femur
  reductor — not concentrated in one type, so likely a proofreading/ID
  churn gap rather than a systematic mismatch, but not yet investigated).

Output: `data/circuit_map/t1_front_leg_circuit.csv` (gitignored,
regenerate via `python -m fly_robot.connectome.identify_circuit`).

**Not yet done:** T2/T3 (mid/hind leg) circuits — Pugliese's table is
T1-only, so those motor neurons and any analogous CPG will need to be
identified independently (Phase 1 work), not carried over from this table.

## 2026-09-18 — First 3D render of the T1 walking circuit (MaleCNS)

Built `fly_robot/analysis/visualize_circuit.py`: fetches real skeletons
(from neuprint, via `navis`) for the 142 resolved neurons in
`t1_front_leg_circuit.csv` and renders them in 3D, colored by role
(DNg100 = red, CPG excitatory = blue, CPG inhibitory = orange, T1 leg
motor neurons = gray). Dark/light variants, plus a version cropped to
just the VNC leg neuropil with the CPG neurons drawn thicker (otherwise
invisible next to the much larger motor neuron dendritic fields) — see
`media/t1_walking_circuit_*.png`.

This was a real check, not just a picture: if our neuron identification
via `mancBodyid` were wrong, we'd expect to see anatomically nonsensical
results (e.g. neurons scattered outside neuropil, or motor neurons in the
brain). Instead the render shows exactly what's biologically expected:

- DNg100's dendrites arborize in the brain; a single axon descends
  through the neck connective into the VNC (visible as the long thin red
  line spanning the whole image) — correct for a descending neuron.
- The 138 T1 leg motor neurons cluster bilaterally in the two T1 leg
  neuropils, with thin processes exiting toward the leg nerve.
- The 3 CPG interneurons cluster tightly inside one T1 neuropil, right
  where DNg100's axon terminates — consistent with DNg100 driving this
  circuit directly, as the paper describes.
- DNg100 also has a second, denser local arbor inside that same T1
  neuropil beyond just the CPG contact point — plausible for a command
  neuron with direct as well as CPG-mediated motor drive, but this is our
  visual read, not something we've independently confirmed against the
  paper's text.

One implementation note for future reruns: `navis.plot2d`'s
`view=("x", "-z")` inverts the *displayed* axis, it does not negate the
underlying coordinate data — cropping/zooming code must compute bounds
from raw `z`, not `-z`, or the crop silently points at empty space.
