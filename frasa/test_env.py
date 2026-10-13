
import argparse

import gymnasium as gym
import numpy as np

from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

import frasa_env


gym.register_envs(frasa_env)


# ============================================================
# Arguments
# ============================================================

argparser = argparse.ArgumentParser(
    description="Test the X2 FRASA stand-up environment"
)

argparser.add_argument(
    "--env",
    type=str,
    default="frasa-standup-v0",
    help="Environment to test",
)

argparser.add_argument(
    "--random",
    action="store_true",
    help="Use random actions instead of zero actions",
)

argparser.add_argument(
    "--normal",
    action="store_true",
    help="Use normal action noise",
)

argparser.add_argument(
    "--orn",
    action="store_true",
    help="Use Ornstein-Uhlenbeck action noise",
)

argparser.add_argument(
    "--std",
    type=float,
    default=0.1,
    help="Standard deviation for action noise",
)

argparser.add_argument(
    "--theta",
    type=float,
    default=0.15,
    help="Theta for Ornstein-Uhlenbeck noise",
)

args = argparser.parse_args()


# ============================================================
# Create environment
# ============================================================

env = gym.make(args.env)

obs, info = env.reset()

print("Environment:", args.env)
print("Observation shape:", env.observation_space.shape)
print("Action shape:", env.action_space.shape)
print("Action low:", env.action_space.low)
print("Action high:", env.action_space.high)

# ============================================================
# Action noise
# ============================================================

noise = None

if args.normal:

    noise = NormalActionNoise(
        mean=np.zeros(
            env.action_space.shape[0]
        ),
        sigma=args.std * np.ones(
            env.action_space.shape[0]
        ),
    )

elif args.orn:

    noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(
            env.action_space.shape[0]
        ),
        sigma=args.std * np.ones(
            env.action_space.shape[0]
        ),
        theta=args.theta,
    )


# ============================================================
# Episode variables
# ============================================================

returns = 0.0
step = 0


# ============================================================
# Test loop
# ============================================================

while True:

    step += 1

    # --------------------------------------------------------
    # Generate action
    # --------------------------------------------------------

    if args.random:
        action = env.action_space.sample()

    else:
        action = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Add optional noise
    # --------------------------------------------------------

    if noise is not None:
        action += noise()

    # Keep action inside the environment's limits.
    action = np.clip(
        action,
        env.action_space.low,
        env.action_space.high,
    )

    # --------------------------------------------------------
    # Step environment
    # --------------------------------------------------------

    obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    returns += reward

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    env.render()

    # --------------------------------------------------------
    # Episode termination
    # --------------------------------------------------------

    if (
        terminated
        or truncated
        or step > 30
    ):

        if terminated:
            status = "terminated"

        elif truncated:
            status = "truncated"

        else:
            status = "30-step limit"

        print(
            f"Episode finished ({status}) "
            f"after {step} steps, "
            f"returns: {returns:.4f}"
        )

        step = 0
        returns = 0.0

        obs, info = env.reset()

