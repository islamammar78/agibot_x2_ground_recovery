#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import frasa_env
from sbx import CrossQ


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "/home/islam/frasa_runs/E/logs/"
    "crossq/frasa-standup-v0_1/frasa-standup-v0.zip"
)

ENV_ID = "frasa-standup-v0"

START_HEIGHT = 0.16       # Initial base height in metres
SETTLE_SECONDS = 1.0      # Time to settle before policy control
CONTROL_DT = 0.05         # Policy control interval: 20 Hz


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

gym.register_envs(frasa_env)

print("Loading trained CrossQ model...")
model = CrossQ.load(MODEL_PATH, device="cpu")


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

print("Creating MuJoCo environment...")

env = gym.make(
    ENV_ID,
    render_mode="human",
    options={"render_realtime": True},
)

u = env.unwrapped

# Initialise the environment.
env.reset(options={"use_cache": False})


# ============================================================
# SET INITIAL ROBOT POSE
# ============================================================

print("Setting initial robot pose...")

u.sim.reset()

# Clear joint positions, velocities, and control inputs.
u.sim.data.qpos[:] = 0.0
u.sim.data.qvel[:] = 0.0
u.sim.data.ctrl[:] = 0.0

# Set the initial floating-base orientation.
# Quaternion below is assigned in MuJoCo's w, x, y, z order.
qx, qy, qz, qw = Rotation.from_euler(
    "y", -90, degrees=True
).as_quat()

u.sim.data.qpos[0:3] = [0.0, 0.0, START_HEIGHT]
u.sim.data.qpos[3:7] = [qw, qx, qy, qz]

# Update derived MuJoCo quantities after setting the pose.
mujoco.mj_forward(u.sim.model, u.sim.data)


# ============================================================
# SETTLE BEFORE POLICY CONTROL
# ============================================================

print(
    f"Letting the robot settle for "
    f"{SETTLE_SECONDS:.1f} seconds..."
)

# Keep the actuators' control inputs at zero during settling.
u.sim.data.ctrl[:] = 0.0

settle_steps = round(SETTLE_SECONDS / u.sim.dt)

for _ in range(settle_steps):
    u.sim.data.ctrl[:] = 0.0

    # Advance the physics simulation without policy actions.
    u.sim.step()

    # Keep the viewer updated, if supported by this environment.
    if u.render_mode == "human":
        u.sim.render(u.options["render_realtime"])

    # Pace the settling simulation in real time.
    time.sleep(u.sim.dt)

print("Settling complete.")


# ============================================================
# SYNCHRONISE ENVIRONMENT STATE AFTER SETTLING
# ============================================================

# The settling phase may have changed the robot's joint positions.
# Synchronise the environment's internal state with the actual pose.
q = np.array(
    [u.sim.get_q(name) for name in u.left_joint_names],
    dtype=np.float64,
)

u.q_desired = q.copy()

u.q_history = [
    q.copy()
    for _ in range(u.q_history_size)
]

initial_tilt = u.get_tilt()

u.tilt_history = [
    initial_tilt
    for _ in range(u.tilt_history_size)
]

u.dtilt_history = [
    0.0
    for _ in range(u.dtilt_history_size)
]

u.previous_actions = [
    np.zeros(len(u.dofs), dtype=np.float64)
    for _ in range(u.options["previous_actions"])
]

# Start the recovery episode's clock from zero after settling.
u.sim.frame = 0
u.sim.t = 0.0

# Get the initial observation from the settled state.
obs = u.get_observation()

print(
    f"Initial tilt after settling: "
    f"{np.rad2deg(u.get_tilt()):.1f} degrees"
)


# ============================================================
# RUN ONE RECOVERY EPISODE
# ============================================================

terminated = False
truncated = False
total_reward = 0.0
steps = 0

print("\nStarting one recovery episode from the settled pose.")
print("Close the MuJoCo viewer or press Ctrl+C to stop.")

try:
    while not (terminated or truncated):

        start_time = time.perf_counter()

        # Get the trained policy's action.
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        # Apply the action and advance the environment.
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        # Keep policy execution near 20 Hz in real time.
        elapsed = time.perf_counter() - start_time
        remaining = CONTROL_DT - elapsed

        if remaining > 0:
            time.sleep(remaining)

        # Periodic diagnostic output.
        if steps % 20 == 0:
            print(
                f"Step: {steps} | "
                f"Reward: {total_reward:.3f} | "
                f"Tilt: {np.rad2deg(u.get_tilt()):.1f} deg"
            )

except KeyboardInterrupt:
    print("\nEpisode interrupted by user.")

finally:
    print("\nEpisode finished")
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    env.close()