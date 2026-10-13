"""Generate the initial-pose cache used by the stand-up environment.

Output: frasa_env/env/standup_initial_configurations.pkl, a shuffled list of
[qpos, qvel, q] entries. The environment's reset() only uses qpos (velocities
are zeroed), so each entry is a robot that is already at rest.

Two kinds of poses are produced:
  * supine (default 70%): the robot lies on its back. It is dropped from a
    random height, left with zero torque for --settle-time seconds, and only
    the final settled pose is stored. Training therefore starts from a robot
    fully at rest on the floor, not from a fall in progress.
  * standing (default 30%): near the target pose. FRASA uses such starts
    (10% in the paper) so the agent sees high-reward states; the exponential
    state reward is almost zero far from the target.

Example: python standup_generate_initial.py --count 300
"""
import argparse
import pickle

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import frasa_env

gym.register_envs(frasa_env)

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=1000)
parser.add_argument("--stand-fraction", type=float, default=0.3,
                    help="fraction of poses that start close to the target (standing)")
parser.add_argument("--z-min", type=float, default=0.17)
parser.add_argument("--z-max", type=float, default=0.40)
parser.add_argument("--settle-time", type=float, default=10.0,
                    help="zero-torque settle time of the supine poses [s]")
parser.add_argument("--stand-settle-time", type=float, default=0.8,
                    help="PD-held settle time of the standing poses [s]")
parser.add_argument("--near-offset", type=float, default=0.1,
                    help="FRASA-style joint offset [rad] for most standing poses")
parser.add_argument("--far-offset", type=float, default=0.25,
                    help="joint offset [rad] for the partway standing poses")
parser.add_argument("--far-fraction", type=float, default=0.3,
                    help="fraction of standing poses that use --far-offset")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = gym.make("frasa-standup-v0")
u = env.unwrapped
env.reset(seed=args.seed, options={"use_cache": False})
rng = np.random.default_rng(args.seed)

# Target state = [6 joint angles, tilt]; its first entries are the target joints.
desired = np.array(u.options["desired_state"])
target_q = desired[: len(u.dofs)]

# Supine orientation: -90 deg about y. MuJoCo quaternion order is [w, x, y, z].
qx, qy, qz, qw = Rotation.from_euler("y", -90, degrees=True).as_quat()
SUPINE_QUAT = [qw, qx, qy, qz]
UPRIGHT_QUAT = [1.0, 0.0, 0.0, 0.0]


def q_of_pose() -> np.ndarray:
    """Current angles of the 6 controlled (left-side) joints."""
    return np.array([u.sim.get_q(n) for n in u.left_joint_names])


def place(qpos: np.ndarray) -> None:
    """Reset the simulator and put the robot at qpos with zero velocity and torque."""
    u.sim.reset()
    u.sim.data.qpos[:] = qpos
    u.sim.data.qvel[:] = 0
    u.sim.data.ctrl[:] = 0
    mujoco.mj_forward(u.sim.model, u.sim.data)


def supine_pose() -> np.ndarray:
    """Drop the robot supine from a random height and return the settled qpos."""
    qpos = np.zeros_like(u.sim.data.qpos)
    qpos[2] = rng.uniform(args.z_min, args.z_max)
    qpos[3:7] = SUPINE_QUAT
    place(qpos)
    # Zero torque: the robot simply falls and comes to rest under gravity.
    for _ in range(round(args.settle_time / u.sim.dt)):
        u.sim.data.ctrl[:] = 0
        u.sim.step()
    return u.sim.data.qpos.copy()


def standing_pose():
    """Near-target pose held by the env's own PD controller. Returns None if unstable."""
    scale = args.far_offset if rng.random() < args.far_fraction else args.near_offset
    offset = rng.uniform(-scale, scale)
    # Start from the target joints and perturb them so the starts are near, not exactly at, the target.
    q = target_q.copy()
    # Same balanced offset FRASA uses: hip_pitch, knee, ankle_pitch.
    q[3] -= offset
    q[4] += 2 * offset
    q[5] -= offset
    # Clip to the joint ranges.
    q = np.clip(q, u.range_low, u.range_high)
    # Right-side targets mirror the left; only shoulder roll flips sign.
    right = q.copy()
    right[2] *= -1.0

    qpos = np.zeros_like(u.sim.data.qpos)
    qpos[2] = 0.7  # pelvis height [m] of the upright robot at the start
    qpos[3:7] = UPRIGHT_QUAT
    place(qpos)
    for n, v in zip(u.left_joint_names, q):
        u.sim.set_q(n, v)
    for n, v in zip(u.right_joint_names, right):
        u.sim.set_q(n, v)
    mujoco.mj_forward(u.sim.model, u.sim.data)

    # Hold the target joints with the env's PD controller (same kp/kd as training).
    for _ in range(round(args.stand_settle_time / u.sim.dt)):
        u.sim.apply_pd_control(u.left_joint_names, u.left_actuators_indexes,
                               q, u.options["kp"], u.options["kd"])
        u.sim.apply_pd_control(u.right_joint_names, u.right_actuators_indexes,
                               right, u.options["kp"], u.options["kd"])
        u.sim.step()

    # Keep only poses that are really standing: pelvis high, torso upright, almost still.
    d = u.sim.data
    state = np.array([*q_of_pose(), u.get_tilt()])
    stable = (
        d.body("pelvis").xpos[2] > 0.55
        and abs(u.get_tilt()) < np.deg2rad(15)
        and np.abs(d.qvel).max() < 2.0
    )
    if not stable:
        return None, None
    return d.qpos.copy(), np.linalg.norm(state - desired)


n_stand = round(args.count * args.stand_fraction)
n_supine = args.count - n_stand

# Supine poses are always accepted; standing poses are retried until enough are stable.
configs, dists, stand_tries = [], [], 0
for i in range(n_supine):
    qpos = supine_pose()
    configs.append([qpos, np.zeros_like(u.sim.data.qvel), q_of_pose()])
    print(f"\rsupine {i + 1}/{n_supine}", end="", flush=True)
print()

while len(dists) < n_stand:
    stand_tries += 1
    qpos, dist = standing_pose()
    if qpos is None:
        continue
    configs.append([qpos, np.zeros_like(u.sim.data.qvel), q_of_pose()])
    dists.append(dist)
    print(f"\rstanding {len(dists)}/{n_stand} ({stand_tries} tries)", end="", flush=True)
print()

# Shuffle so supine and standing poses are mixed in the file.
order = rng.permutation(len(configs))
configs = [configs[i] for i in order]

filename = u.get_initial_config_filename()
pickle.dump(configs, open(filename, "wb"))
print(f"Saved {len(configs)} configs ({n_supine} supine, {n_stand} standing) to {filename}")
if dists:
    print(f"standing distance to target: mean {np.mean(dists):.2f}, max {np.max(dists):.2f}")
