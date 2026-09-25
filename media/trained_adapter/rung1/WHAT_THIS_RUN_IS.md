# `rung1` is a BALL-RIG run, not a walking run

**Do not cite any number in this directory as evidence about walking.**

This run optimised **ball rotation on a tethered fly**:

- rig: tethered ball (`run_trial(..., on_ball=True)`) — thorax fixed, legs
  on a freely rotating sphere
- reward: `fly_robot/adapter/reward.py`, config hash `88f681a4ab03d6d8…`
  (the **ball** reward, *not* `reward_ground`'s `0d6485421541f466…`)
- 71 parameters, all trained at once
- stopped at generation 70 of 250; best score **+0.2928 at generation 2**,
  with no improvement across the following 68 generations and mean progress
  negative throughout

It was launched and resumed under the mistaken belief that it was the
free-ground walking experiment. The full account, including why it was
possible and what now prevents it, is in
`docs/trained_adapter/GROUND_BRIDGE.md` §1.

**Kept deliberately** as the documented record of a rejected objective —
it is real data about the ball rig and an honest story beat, not a mistake
to be hidden.

The walking experiment is `r1a_ground` (stage `output`) and then
`r1b_sensory`. This checkpoint **cannot** be resumed under that objective:
the reward-hash guard and the training-contract guard both refuse it.
