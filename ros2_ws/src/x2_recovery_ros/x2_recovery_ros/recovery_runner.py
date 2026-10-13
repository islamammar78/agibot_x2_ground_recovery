
#!/usr/bin/env python3

"""Runs one supine-to-standing recovery episode with the trained policy.

Steps: load the CrossQ checkpoint, create the MuJoCo frasa-standup-v0
environment, place the robot on its back, let it settle with zero torque,
then run the policy at 20 Hz while publishing joint states and checking for
success (upright and still for STABLE_HOLD_SECONDS) or a timeout.
"""
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from sbx import CrossQ
from sensor_msgs.msg import JointState


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

# Expected repository layout:
#
# agibot_x2_ground_recovery/
# ├── frasa/
# │   └── models/
# │       └── crossq/
# │           └── frasa-standup-v0.zip
# └── ros2_ws/
#     └── src/
#         └── x2_recovery_ros/
#             └── x2_recovery_ros/
#                 └── recovery_runner.py
#
# Always use the FRASA project inside this repository.
# Do not use /home/islam/Desktop/frasa.

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FRASA_DIR = (PROJECT_ROOT / "frasa").resolve()
FRASA_PATH = str(FRASA_DIR)

if not FRASA_DIR.is_dir():
    raise FileNotFoundError(
        f"FRASA project directory not found: {FRASA_DIR}"
    )

# Put the correct FRASA directory first in Python's import path.
while FRASA_PATH in sys.path:
    sys.path.remove(FRASA_PATH)

sys.path.insert(0, FRASA_PATH)

# Use the checkpoint included in the repository by default.
# An environment variable can override this path.
DEFAULT_MODEL_PATH = (
    FRASA_DIR
    / "models"
    / "crossq"
    / "frasa-standup-v0.zip"
)

MODEL_PATH = Path(
    os.environ.get(
        "X2_RECOVERY_MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )
).expanduser().resolve()

ENV_ID = "frasa-standup-v0"

START_HEIGHT = 0.16      # initial pelvis height [m]
SETTLE_SECONDS = 1.0     # zero-torque settle before the policy starts [s]
CONTROL_DT = 0.05        # policy period [s] = 20 Hz

# Provisional upright stability criteria.
# Validate these thresholds against the actual X2 environment.
UPRIGHT_TILT_DEG = 15.0      # max |tilt| to count as upright
MAX_ANGULAR_SPEED = 0.3     # max base angular speed [rad/s] to count as still
STABLE_HOLD_SECONDS = 1.0   # time both conditions must hold for success [s]


# ============================================================
# JOINT-STATE PUBLISHING
# ============================================================

def publish_joint_states(sim, publish_joint_state):
    """Publish positions for all scalar hinge joints in the model."""

    if publish_joint_state is None:
        return

    model = sim.model
    data = sim.data

    msg = JointState()

    # Use the ROS clock for the message timestamp.
    from rclpy.clock import Clock

    msg.header.stamp = Clock().now().to_msg()

    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]

        # JointState positions are scalar, so omit free/ball joints.
        if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if joint_name is None:
            continue

        qpos_index = model.jnt_qposadr[joint_id]

        msg.name.append(joint_name)
        msg.position.append(float(data.qpos[qpos_index]))

    publish_joint_state(msg)


# ============================================================
# RECOVERY HELPERS
# ============================================================

def get_angular_speed(sim):
    """Return the floating-base angular speed, if available."""

    qvel = np.asarray(sim.data.qvel)

    # A free joint normally has three translational and three
    # rotational velocity components at the start of qvel.
    if qvel.size >= 6:
        return float(np.linalg.norm(qvel[3:6]))

    return float("inf")


def is_stably_upright(env):
    """
    Provisional stability check.

    Checks tilt and angular speed only. It does not verify contact
    with both feet or exclude support from other body parts.
    """

    u = env.unwrapped

    tilt_deg = abs(float(np.rad2deg(u.get_tilt())))
    angular_speed = get_angular_speed(u.sim)

    return (
        tilt_deg <= UPRIGHT_TILT_DEG
        and angular_speed <= MAX_ANGULAR_SPEED
    )


def make_result(
    success,
    timed_out,
    steps,
    total_reward,
    message,
):
    """Create a consistent result dictionary."""

    return {
        "success": bool(success),
        "timed_out": bool(timed_out),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "message": str(message),
    }


# ============================================================
# RUN ONE RECOVERY EPISODE
# ============================================================

def run_recovery_episode(
    publish_joint_state=None,
    timeout_seconds=30.0,
):
    """
    Execute one recovery episode using the trained CrossQ policy.

    Args:
        publish_joint_state:
            Optional callback accepting a sensor_msgs/JointState.

        timeout_seconds:
            Maximum wall-clock duration for the complete episode,
            including environment setup and settling.

    Returns:
        A dictionary containing:
            success
            timed_out
            steps
            total_reward
            message
    """

    env = None

    start_time = time.monotonic()

    steps = 0
    total_reward = 0.0
    stable_since = None

    timed_out = False
    success = False

    message = "Recovery did not complete successfully."

    try:
        # ----------------------------------------------------
        # 1. Validate paths and load the trained policy
        # ----------------------------------------------------

        if not FRASA_DIR.is_dir():
            raise FileNotFoundError(
                f"FRASA project directory not found: {FRASA_DIR}"
            )

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"CrossQ checkpoint not found: {MODEL_PATH}\n"
                "Place the checkpoint at "
                f"{DEFAULT_MODEL_PATH} or set "
                "X2_RECOVERY_MODEL_PATH to its location."
            )

        import frasa_env

        gym.register_envs(frasa_env)

        print(f"FRASA directory: {FRASA_DIR}")
        print(f"Loading CrossQ checkpoint: {MODEL_PATH}")

        model = CrossQ.load(
            str(MODEL_PATH),
            device="cpu",
        )

        # ----------------------------------------------------
        # 2. Create the MuJoCo environment
        # ----------------------------------------------------

        print("Creating MuJoCo environment...")

        env = gym.make(
            ENV_ID,
            render_mode="human",
            options={"render_realtime": True},
        )

        u = env.unwrapped

        # ----------------------------------------------------
        # 3. Reset the environment and set the initial pose
        # ----------------------------------------------------

        env.reset(options={"use_cache": False})

        print("Setting initial robot pose...")

        u.sim.reset()

        u.sim.data.qpos[:] = 0.0
        u.sim.data.qvel[:] = 0.0
        u.sim.data.ctrl[:] = 0.0

        # Rotation.from_euler returns quaternion [x, y, z, w].
        qx, qy, qz, qw = Rotation.from_euler(
            "y",
            -90,
            degrees=True,
        ).as_quat()

        # MuJoCo free-joint qpos uses [x, y, z, qw, qx, qy, qz].
        u.sim.data.qpos[0:3] = [0.0, 0.0, START_HEIGHT]
        u.sim.data.qpos[3:7] = [qw, qx, qy, qz]

        mujoco.mj_forward(u.sim.model, u.sim.data)

        # ----------------------------------------------------
        # 4. Allow the robot to settle
        # ----------------------------------------------------

        print(f"Settling for {SETTLE_SECONDS:.1f} seconds...")

        settle_steps = round(SETTLE_SECONDS / u.sim.dt)

        for _ in range(settle_steps):
            if time.monotonic() - start_time >= timeout_seconds:
                timed_out = True
                message = "Timed out while settling."

                return make_result(
                    False,
                    True,
                    steps,
                    total_reward,
                    message,
                )

            u.sim.data.ctrl[:] = 0.0
            u.sim.step()

            # Render the viewer while the robot settles.
            if hasattr(env, "render"):
                env.render()

            time.sleep(u.sim.dt)

        # ----------------------------------------------------
        # 5. Initialise environment-specific observation history
        #    (reset() was bypassed above, so the env state that the
        #    observation uses is rebuilt here: q_desired = q, histories, zero previous actions)
        # ----------------------------------------------------

        q = np.array(
            [
                u.sim.get_q(name)
                for name in u.left_joint_names
            ],
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
            np.zeros(
                len(u.dofs),
                dtype=np.float64,
            )
            for _ in range(u.options["previous_actions"])
        ]

        u.sim.frame = 0
        u.sim.t = 0.0

        obs = u.get_observation()

        print(
            "Initial tilt after settling: "
            f"{np.rad2deg(u.get_tilt()):.1f} degrees"
        )

        print("Starting recovery episode.")

        # ----------------------------------------------------
        # 6. Run the trained policy
        # ----------------------------------------------------

        terminated = False
        truncated = False

        while not (terminated or truncated):
            elapsed_total = time.monotonic() - start_time

            if elapsed_total >= timeout_seconds:
                timed_out = True
                message = "Recovery timed out."
                break

            loop_start = time.monotonic()

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += float(reward)
            steps += 1

            # Publish the current simulated joint positions.
            publish_joint_states(
                u.sim,
                publish_joint_state,
            )

            # Prefer an explicit success signal if the environment
            # provides one.
            explicit_success = None

            if isinstance(info, dict):
                if "is_success" in info:
                    explicit_success = bool(info["is_success"])
                elif "success" in info:
                    explicit_success = bool(info["success"])

            if explicit_success is True:
                success = True
                message = "Environment reported recovery success."
                break

            if explicit_success is False:
                stable_since = None

            else:
                # Provisional fallback: sustained low tilt and
                # low angular speed. This is not a foot-contact test.
                if is_stably_upright(env):
                    if stable_since is None:
                        stable_since = time.monotonic()

                    elif (
                        time.monotonic() - stable_since
                        >= STABLE_HOLD_SECONDS
                    ):
                        success = True
                        message = (
                            "Upright stability heuristic satisfied."
                        )
                        break

                else:
                    stable_since = None

            if steps % 20 == 0:
                print(
                    f"Step: {steps} | "
                    f"Reward: {total_reward:.3f} | "
                    f"Tilt: {np.rad2deg(u.get_tilt()):.1f} deg"
                )

            # Keep the loop close to the configured control period.
            elapsed_loop = time.monotonic() - loop_start
            remaining = CONTROL_DT - elapsed_loop

            if remaining > 0:
                time.sleep(remaining)

        # ----------------------------------------------------
        # 7. Determine the final result
        # ----------------------------------------------------

        if timed_out:
            success = False
            message = "Recovery timed out."

        elif success:
            pass

        elif terminated or truncated:
            message = (
                "Episode ended before recovery success was detected."
            )

        print(
            f"Recovery finished | Steps: {steps} | "
            f"Reward: {total_reward:.3f} | "
            f"Success: {success} | Timeout: {timed_out}"
        )

        return make_result(
            success,
            timed_out,
            steps,
            total_reward,
            message,
        )

    except Exception as exc:
        message = f"Recovery runner error: {exc}"
        print(message)

        return make_result(
            False,
            timed_out,
            steps,
            total_reward,
            message,
        )

    finally:
        if env is not None:
            env.close()
