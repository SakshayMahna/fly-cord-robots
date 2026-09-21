"""Phase 5 compute gate — component-level timing of one closed-loop trial.

Splits the measured ~25 s / 4 s-trial into its parts so we know what a
GPU/batched version would actually have to accelerate. Nothing here
changes the model; it calls the same pieces run_trial calls.
"""
import time
import numpy as np
import scipy.sparse as sp

from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

N_TIME = 200  # neural steps to time over


def timeit(fn, n):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


t0 = time.perf_counter()
model, motor_groups, sensory_groups, wtable, info = build_trial_components(
    replicate=1, param_seed=PILOT_PARAM_SEED, duration_s=4.0)
setup_s = time.perf_counter() - t0
n = model.neurons.n_neurons
W = model.neurons.w_eff
print(f"SETUP build_trial_components: {setup_s:.1f} s")
print(f"network: n={n}  nnz={W.nnz}  density={W.nnz/n/n*100:.4f}%  "
      f"dtype={W.dtype}  mem={W.data.nbytes/1e6:.1f} MB")

# --- 1. neural step, in isolation (no body at all) -----------------------
model.reset()
model.rates = np.random.default_rng(0).uniform(0, 20, n)  # realistic active state
extra = np.zeros(n)
neural_step_s = timeit(lambda: model.step(0.001, extra_input=extra), N_TIME)
print(f"\n[neural] one RK4 step (4 matvecs): {neural_step_s*1e3:.3f} ms"
      f"  -> {neural_step_s*4000:.2f} s per 4 s trial")

# --- 2. raw sparse matvec vs batched matmul ------------------------------
print("\n[matvec scaling] W @ R for batch B (this is what a GPU would batch)")
for B in (1, 8, 32, 64, 128, 256):
    R = np.asfortranarray(np.random.default_rng(1).uniform(0, 20, (n, B)))
    if B == 1:
        R = R[:, 0]
    dt = timeit(lambda: W @ R, 20 if B <= 64 else 5)
    print(f"   B={B:4d}  {dt*1e3:8.2f} ms   per-sample {dt/B*1e3:7.3f} ms"
          f"   speedup/sample vs B=1: {(base/ (dt/B)) if B>1 else 1:.1f}x"
          if B > 1 else f"   B={B:4d}  {dt*1e3:8.2f} ms   per-sample {dt*1e3:7.3f} ms")
    if B == 1:
        base = dt

# --- 3. physics, in isolation --------------------------------------------
from flygym import Simulation
from flygym.compose.fly.base_fly import ActuatorType
from fly_robot.bodies.neuromechfly import build_ball_fly

fly, world, _m, _d, *_cams = build_ball_fly()
physics = Simulation(world)
physics.reset()
physics.warmup()
actuated = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
targets = np.zeros(len(actuated))
phys_step_s = timeit(lambda: physics.step(), 2000)
print(f"\n[physics] one substep (dt={physics.timestep}): {phys_step_s*1e3:.4f} ms")
print(f"          10 substeps per neural step: {phys_step_s*10*1e3:.3f} ms"
      f"  -> {phys_step_s*10*4000:.2f} s per 4 s trial")

get_ang_s = timeit(lambda: physics.get_joint_angles(fly.name), 2000)
set_act_s = timeit(lambda: physics.set_actuator_inputs(
    fly.name, ActuatorType.POSITION, targets), 2000)
print(f"[physics] get_joint_angles {get_ang_s*1e3:.4f} ms, "
      f"set_actuator_inputs {set_act_s*1e3:.4f} ms")

# --- 4. interface costs ---------------------------------------------------
from flygym.compose import KinematicPosePreset
from flygym.anatomy import AxisOrder
from fly_robot.interface.joint_to_sensory_neuron import SensoryEncoder
from fly_robot.sim.closed_loop import _joint_targets_from_rates

neutral_lookup = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
    AxisOrder.ROLL_PITCH_YAW).joint_angles_lookup_rad
all_jd = fly.get_jointdofs_order()
enc = SensoryEncoder(sensory_groups, all_jd,
                     {d.name: neutral_lookup.get(d.name, 0.0) for d in all_jd},
                     gain=3.5, mode="deviation")
angles = physics.get_joint_angles(fly.name)
vels = physics.get_joint_velocities(fly.name)
enc_s = timeit(lambda: enc.input_current(angles, vels), 500)

dof_index_by_name = {d.name: i for i, d in enumerate(actuated)}
neutral_angles = np.array([neutral_lookup.get(d.name, 0.0) for d in actuated])
rates = model.rates
dec_s = timeit(lambda: _joint_targets_from_rates(
    rates, motor_groups, dof_index_by_name, neutral_angles, 0.5, 20.0), 500)
print(f"\n[interface] sensory encode {enc_s*1e3:.3f} ms"
      f"  -> {enc_s*4000:.2f} s per trial")
print(f"[interface] motor decode  {dec_s*1e3:.3f} ms"
      f"  -> {dec_s*4000:.2f} s per trial")

total = (neural_step_s + phys_step_s*10 + enc_s + dec_s + get_ang_s + set_act_s) * 4000
print(f"\n[TOTAL estimated] {total:.1f} s per 4 s trial "
      f"(recorded pilot median: 25.2 s)")
print(f"  neural   {neural_step_s*4000:6.1f} s  {neural_step_s/(total/4000)*100:5.1f}%")
print(f"  physics  {phys_step_s*10*4000:6.1f} s  {phys_step_s*10/(total/4000)*100:5.1f}%")
print(f"  encode   {enc_s*4000:6.1f} s  {enc_s/(total/4000)*100:5.1f}%")
print(f"  decode   {dec_s*4000:6.1f} s  {dec_s/(total/4000)*100:5.1f}%")
print(f"  io       {(get_ang_s+set_act_s)*4000:6.1f} s  "
      f"{(get_ang_s+set_act_s)/(total/4000)*100:5.1f}%")
