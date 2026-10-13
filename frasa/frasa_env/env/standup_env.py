import os
import pickle
import random
import warnings
from typing import Optional

import mujoco
import gymnasium
import numpy as np
from gymnasium import spaces

from frasa_env.mujoco_simulator.simulator import Simulator, tf


class StandupEnv(gymnasium.Env):
    metadata = {
        "render_modes": ["human", "none"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode="none",
        options: Optional[dict] = None,
        evaluation: bool = False,
    ):
        self.options = {
            # ---------------------------------------------------------
            # Timing
            # ---------------------------------------------------------

            # Duration of the stabilization pre-simulation [s]
            "stabilization_time": 2.0,

            # Episode duration before truncation [s]
            "truncate_duration": 5.0,

            # RL control period [s]
            # 0.05 s = 20 Hz
            "dt": 0.05,

            # Maximum desired joint velocity [rad/s]
            "vmax": 2 * np.pi,

            # Render in realtime
            "render_realtime": True,

            # ---------------------------------------------------------
            # Target state
            # ---------------------------------------------------------
            #
            # [elbow,
            #  shoulder_pitch,
            #  shoulder_roll,
            #  hip_pitch,
            #  knee,
            #  ankle_pitch,
            #  IMU_pitch]
            #
            # Shoulder roll is represented in the LEFT-arm coordinate
            # convention. The physical right shoulder roll is mirrored.
            #
            "desired_state": np.array([
                -0.3,      # elbow
                 0.0,      # shoulder_pitch
                 0.2,      # shoulder_roll
                -0.312,    # hip_pitch
                 0.669,    # knee
                -0.363,    # ankle_pitch
                 0.0,      # IMU_pitch
            ]),

            # Probability of starting near the final/target configuration
            "reset_final_p": 0.1,

            # ---------------------------------------------------------
            # Termination conditions
            # ---------------------------------------------------------

            "terminate_upside_down": True,
            "terminate_gyro": True,
            "terminate_shock": False,

            # ---------------------------------------------------------
            # Domain randomization
            # ---------------------------------------------------------

            # Joint/body orientation error [degrees]
            "random_angles": 1.5,

            # RL timestep randomization
            "random_time_ratio": [0.98, 1.02],

            # Floor friction
            "random_friction": [0.25, 1.5],

            # Trunk mass multiplier
            "random_body_mass": [0.8, 1.2],

            # Trunk COM position randomization [m]
            "random_body_com_pos": 5e-3,

            # Joint damping multiplier
            "random_damping": [0.8, 1.2],

            # Joint friction-loss multiplier
            "random_frictionloss": [0.8, 1.2],

            # X2 does not use the FRASA voltage randomization
            "random_v": None,
            "nominal_v": None,

            # ---------------------------------------------------------
            # Control
            # ---------------------------------------------------------

            # X2 training uses:
            #
            # action = desired joint velocity
            # q_desired += action * dt
            # torque = PD(q_desired, q, qdot)
            #
            "control": "velocity",

            # Kept for compatibility with old configuration files.
            # It is no longer used because PD is applied continuously
            # at every MuJoCo timestep.
            "interpolate": True,

            # ---------------------------------------------------------
            # Sensor delays
            # ---------------------------------------------------------

            "qdot_delay": 0.030,
            "tilt_delay": 0.050,
            "dtilt_delay": 0.050,

            # Number of previous actions included in observation
            "previous_actions": 1,

            # ---------------------------------------------------------
            # PD controller
            # ---------------------------------------------------------

            "kp": 120.0,
            "kd": 12.0,
        }

        self.options.update(options or {})

        # Only velocity control is supported by this X2 environment.
        if self.options["control"] != "velocity":
            raise ValueError(
                "This X2 environment only supports control='velocity'. "
                "The action is desired joint velocity."
            )

        self.render_mode = render_mode
        self.evaluation = evaluation

        # -------------------------------------------------------------
        # Simulator
        # -------------------------------------------------------------

        self.sim = Simulator()

        # -------------------------------------------------------------
        # Controlled DoFs
        # -------------------------------------------------------------
        #
        # These are the 6 symmetric RL DoFs.
        #
        self.dofs = [
            "elbow_joint",
            "shoulder_pitch_joint",
            "shoulder_roll_joint",
            "hip_pitch_joint",
            "knee_joint",
            "ankle_pitch_joint",
        ]

        self.left_joint_names = [
            f"left_{dof}"
            for dof in self.dofs
        ]

        self.right_joint_names = [
            f"right_{dof}"
            for dof in self.dofs
        ]

        # -------------------------------------------------------------
        # Joint position ranges
        # -------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # q_desired is a JOINT POSITION.
        #
        # Therefore these must come from the joint limits, NOT the
        # actuator torque ctrlrange.
        #
        self.ranges = [
            self.sim.get_range(joint_name)
            for joint_name in self.left_joint_names
        ]

        self.range_low = np.array(
            [r[0] for r in self.ranges],
            dtype=np.float64,
        )

        self.range_high = np.array(
            [r[1] for r in self.ranges],
            dtype=np.float64,
        )

        # -------------------------------------------------------------
        # Actuator indices
        # -------------------------------------------------------------

        self.left_actuators_indexes = [
            self.sim.get_actuator_index(name)
            for name in self.left_joint_names
        ]

        self.right_actuators_indexes = [
            self.sim.get_actuator_index(name)
            for name in self.right_joint_names
        ]

        # -------------------------------------------------------------
        # Body orientation sensor
        # -------------------------------------------------------------

        self.body_orientation_sensor = (
            self.sim.model.sensor("body-orientation")
        )

        self.body_orientation_sensor_id = (
            self.body_orientation_sensor.id
        )

        self.body_orientation_sensor_adr = int(
            self.body_orientation_sensor.adr[0]
        )

        self.body_orientation_sensor_dim = int(
            self.body_orientation_sensor.dim[0]
        )

        # -------------------------------------------------------------
        # Action space
        # -------------------------------------------------------------
        #
        # Action = desired joint velocity [rad/s]
        #
        # 6 dimensions:
        #
        # [-2*pi, +2*pi]
        #
        # for each joint.
        #

        self.action_space = spaces.Box(
            low=np.array(
                [-self.options["vmax"]] * len(self.dofs),
                dtype=np.float32,
            ),
            high=np.array(
                [self.options["vmax"]] * len(self.dofs),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # -------------------------------------------------------------
        # Observation space
        # -------------------------------------------------------------
        #
        # Observation:
        #
        # q                  6
        # qdot               6
        # q_desired          6
        # tilt               1
        # dtilt              1
        # previous action    6
        #
        # Total = 26
        #

        observation_low = np.array(
            [
                # q
                *self.range_low,

                # qdot
                *(-10 * np.ones(len(self.dofs))),

                # q_desired
                *self.range_low,

                # tilt
                -np.pi,

                # dtilt
                -10,

                # previous action
                *(
                    list(self.action_space.low)
                    * self.options["previous_actions"]
                ),
            ],
            dtype=np.float32,
        )

        observation_high = np.array(
            [
                # q
                *self.range_high,

                # qdot
                *(10 * np.ones(len(self.dofs))),

                # q_desired
                *self.range_high,

                # tilt
                np.pi,

                # dtilt
                10,

                # previous action
                *(
                    list(self.action_space.high)
                    * self.options["previous_actions"]
                ),
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )

        # -------------------------------------------------------------
        # qdot delay history
        # -------------------------------------------------------------

        self.q_history = []

        self.q_history_size = max(
            1,
            round(
                self.options["qdot_delay"] / self.sim.dt
            ),
        )

        # -------------------------------------------------------------
        # Tilt delay history
        # -------------------------------------------------------------

        self.tilt_history = []

        self.tilt_history_size = max(
            1,
            round(
                self.options["tilt_delay"] / self.sim.dt
            ),
        )

        # -------------------------------------------------------------
        # Tilt velocity delay history
        # -------------------------------------------------------------

        self.dtilt_history = []

        self.dtilt_history_size = max(
            1,
            round(
                self.options["dtilt_delay"] / self.sim.dt
            ),
        )

        # -------------------------------------------------------------
        # Desired joint position
        # -------------------------------------------------------------
        #
        # This is the internal reference generated by the RL action.
        #
        # action = desired joint velocity
        #
        # q_desired += action * dt
        #
        self.q_desired = np.zeros(
            len(self.dofs),
            dtype=np.float64,
        )

        # -------------------------------------------------------------
        # Previous actions
        # -------------------------------------------------------------

        self.previous_actions = [
            np.zeros(len(self.dofs), dtype=np.float64)
            for _ in range(self.options["previous_actions"])
        ]

        # -------------------------------------------------------------
        # Randomization values
        # -------------------------------------------------------------

        self.trunk_body = "torso_link"

        self.trunk_mass = (
            self.sim.model.body(self.trunk_body).mass.copy()
        )

        self.trunk_ipos = (
            self.sim.model.body(self.trunk_body).ipos.copy()
        )

        self.gainprm_original = (
            self.sim.model.actuator_gainprm.copy()
        )

        self.biasprm_original = (
            self.sim.model.actuator_biasprm.copy()
        )

        self.damping_original = (
            self.sim.model.dof_damping.copy()
        )

        self.frictionloss_original = (
            self.sim.model.dof_frictionloss.copy()
        )

        self.forcerange_original = (
            self.sim.model.actuator_forcerange.copy()
        )

        self.body_quat_original = (
            self.sim.model.body_quat.copy()
        )

        # -------------------------------------------------------------
        # Initial configuration cache
        # -------------------------------------------------------------

        initial_config_path = (
            self.get_initial_config_filename()
        )

        self.initial_config = None

        if os.path.exists(initial_config_path):
            with open(initial_config_path, "rb") as f:
                self.initial_config = pickle.load(f)

    # =================================================================
    # INITIAL CONFIGURATION
    # =================================================================

    def get_initial_config_filename(self) -> str:
        return os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)
            ),
            "standup_initial_configurations.pkl",
        )

    # =================================================================
    # TILT
    # =================================================================

    def get_tilt(self) -> float:
        """
        Get trunk pitch from the X2 body-orientation sensor.
        """

        quat = self.sim.data.sensordata[
            self.body_orientation_sensor_adr:
            self.body_orientation_sensor_adr
            + self.body_orientation_sensor_dim
        ]

        R = np.zeros((3, 3))

        mujoco.mju_quat2Mat(
            R.reshape(-1),
            quat,
        )

        return -np.arctan2(
            R[2, 0],
            R[2, 2],
        )

    # =================================================================
    # OBSERVATION
    # =================================================================

    def get_observation(self) -> np.ndarray:
        """
        Observation:

            q
            qdot
            q_desired
            tilt
            dtilt
            previous_action

        Total dimension = 26.
        """

        # -------------------------------------------------------------
        # Joint positions
        # -------------------------------------------------------------

        q = np.array(
            self.q_history[-1],
            dtype=np.float64,
        )

        # -------------------------------------------------------------
        # Delayed joint velocity
        # -------------------------------------------------------------

        q_first = np.array(
            self.q_history[0],
            dtype=np.float64,
        )

        q_last = np.array(
            self.q_history[-1],
            dtype=np.float64,
        )

        q_dot = (
            q_last - q_first
        ) / (
            self.q_history_size * self.sim.dt
        )

        # -------------------------------------------------------------
        # Delayed tilt
        # -------------------------------------------------------------

        tilt = self.tilt_history[0]

        # -------------------------------------------------------------
        # Delayed tilt velocity
        # -------------------------------------------------------------

        dtilt = self.dtilt_history[0]

        # -------------------------------------------------------------
        # Observation
        # -------------------------------------------------------------

        # Joints can overshoot their limits slightly; keep the observation
        # inside the declared space.
        q = np.clip(q, self.range_low, self.range_high)
        q_dot = np.clip(q_dot, -10.0, 10.0)
        q_desired = np.clip(
            self.q_desired, self.range_low, self.range_high
        )

        return np.array(
            [
                *q,
                *q_dot,
                *q_desired,
                tilt,
                dtilt,
                *np.array(
                    self.previous_actions
                ).flatten(),
            ],
            dtype=np.float32,
        )

    # =================================================================
    # STEP
    # =================================================================

    def step(self, action):
        """
        High-level control loop.

        RL action:
            desired joint velocity

        Integration:
            q_desired += action * dt

        Low-level control:
            torque = kp * (q_desired - q) - kd * qdot

        MuJoCo:
            data.ctrl = torque
        """

        # -------------------------------------------------------------
        # Convert and clip action
        # -------------------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        # -------------------------------------------------------------
        # Actual randomized RL timestep
        # -------------------------------------------------------------

        step_dt = (
            self.time_ratio
            * self.options["dt"]
        )

        # -------------------------------------------------------------
        # Integrate desired velocity into desired position
        # -------------------------------------------------------------

        self.q_desired = (
            self.q_desired
            + action * step_dt
        )

        # Keep desired positions inside physical joint limits.
        self.q_desired = np.clip(
            self.q_desired,
            self.range_low,
            self.range_high,
        )

        # -------------------------------------------------------------
        # Number of MuJoCo steps for this RL action
        # -------------------------------------------------------------

        timesteps = max(
            1,
            round(
                step_dt / self.sim.dt
            ),
        )

        terminated = False
        shock = False

        # -------------------------------------------------------------
        # Low-level PD control
        # -------------------------------------------------------------

        for _ in range(timesteps):

            # ---------------------------------------------------------
            # LEFT ARM / LEG
            # ---------------------------------------------------------

            self.sim.apply_pd_control(
                self.left_joint_names,
                self.left_actuators_indexes,
                self.q_desired,
                self.options["kp"],
                self.options["kd"],
            )

            # ---------------------------------------------------------
            # RIGHT ARM / LEG
            # ---------------------------------------------------------
            #
            # RL coordinates are symmetric.
            #
            # Only shoulder roll needs physical sign conversion:
            #
            # left  +0.2
            # right -0.2
            #

            q_desired_right = (
                self.q_desired.copy()
            )

            q_desired_right[2] *= -1.0

            self.sim.apply_pd_control(
                self.right_joint_names,
                self.right_actuators_indexes,
                q_desired_right,
                self.options["kp"],
                self.options["kd"],
            )

            # ---------------------------------------------------------
            # Step MuJoCo
            # ---------------------------------------------------------

            self.sim.step()

            # ---------------------------------------------------------
            # Shock termination
            # ---------------------------------------------------------

            if self.options["terminate_shock"]:

                centroidal_force = (
                    self.sim.centroidal_force()
                )

                if (
                    centroidal_force
                    - self.centroidal_force
                    > 200.0
                ):
                    shock = True

                self.centroidal_force = (
                    centroidal_force
                )

            # ---------------------------------------------------------
            # Update joint history
            # ---------------------------------------------------------

            q = [
                self.sim.get_q(name)
                for name in self.left_joint_names
            ]

            self.q_history.append(q)

            # ---------------------------------------------------------
            # Update tilt histories
            # ---------------------------------------------------------

            self.tilt_history.append(
                self.get_tilt()
            )

            self.dtilt_history.append(
                self.sim.get_gyro()[1]
            )

            # ---------------------------------------------------------
            # Render
            # ---------------------------------------------------------

            if self.render_mode == "human":
                self.sim.render(
                    self.options["render_realtime"]
                )

        # -------------------------------------------------------------
        # Upside-down termination
        # -------------------------------------------------------------

        if self.options["terminate_upside_down"]:

            tilt = self.get_tilt()

            if np.rad2deg(
                np.abs(tilt)
            ) > 135:

                terminated = True

        # -------------------------------------------------------------
        # High gyro termination
        # -------------------------------------------------------------

        if self.options["terminate_gyro"]:

            gyro = self.sim.get_gyro()

            if abs(gyro[1]) > 5:

                terminated = True

        # -------------------------------------------------------------
        # Shock termination
        # -------------------------------------------------------------

        if (
            self.options["terminate_shock"]
            and shock
        ):
            terminated = True

        # -------------------------------------------------------------
        # Trim histories
        # -------------------------------------------------------------

        self.q_history = (
            self.q_history[
                -self.q_history_size:
            ]
        )

        self.tilt_history = (
            self.tilt_history[
                -self.tilt_history_size:
            ]
        )

        self.dtilt_history = (
            self.dtilt_history[
                -self.dtilt_history_size:
            ]
        )

        # -------------------------------------------------------------
        # Observation
        # -------------------------------------------------------------

        obs = self.get_observation()

        # -------------------------------------------------------------
        # State reward
        # -------------------------------------------------------------
        #
        # Actual physical state:
        #
        # [q(6), tilt]
        #
        # compared against:
        #
        # [desired q(6), desired tilt]
        #

        # R = R_state + R_variation + R_collision
        state_current = np.array(
            [*self.q_history[-1], self.tilt_history[-1]]
        )
        state_error = state_current - np.array(
            self.options["desired_state"]
        )

        # R_state = exp(-w1 * ||psi_t - psi_target||^2)
        # Close to 1 near the standing target and almost 0 far from it
        # (e.g. ~0.45 at distance 0.2, ~0 beyond 0.5), so it is a sparse reward.
        w1 = 20.0
        reward = np.exp(-w1 * np.linalg.norm(state_error) ** 2)

        # R_variation = w2 * exp(-||a_t - a_{t-1}||)
        # Small bonus for smooth actions (no jerky commands).
        action_variation = np.abs(
            action - self.previous_actions[-1]
        )
        w2 = 5e-2
        reward += w2 * np.exp(-np.linalg.norm(action_variation))

        self.previous_actions.append(action.copy())
        self.previous_actions = self.previous_actions[
            -self.options["previous_actions"]:
        ]

        # R_collision = w3 * exp(-self_collisions)
        # Small bonus that shrinks as self-collisions increase.
        w3 = 1e-1
        reward += w3 * np.exp(-self.sim.self_collisions())

        # -------------------------------------------------------------
        # Episode truncation
        # -------------------------------------------------------------
        #
        # self.sim.t is reset to zero after the stabilization phase.
        #

        truncated = (
            self.sim.t
            > self.options["truncate_duration"]
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            {},
        )

    # =================================================================
    # ANGULAR RANDOMIZATION
    # =================================================================

    def apply_angular_offset(
        self,
        joint: str,
        offset: float,
    ):
        """
        Apply a small orientation error to the body associated
        with a joint.

        This follows the original FRASA-style angular
        randomization.
        """

        body_id = int(
            self.sim.model.joint(joint).bodyid[0]
        )

        quat = (
            self.sim.model.body_quat[body_id]
        )

        quat = tf.quaternion_multiply(
            quat,
            tf.quaternion_about_axis(
                offset,
                [0, 0, 1],
            ),
        )

        self.sim.model.body_quat[body_id] = quat

    # =================================================================
    # DOMAIN RANDOMIZATION
    # =================================================================

    def apply_randomization(self):
        """
        Apply domain randomization for the current episode.

        Randomized:
            - RL timestep
            - floor friction
            - trunk mass
            - trunk COM
            - joint damping
            - joint friction loss
            - angular/body orientation offsets

        X2 does not use the original FRASA voltage
        randomization.
        """

        # -------------------------------------------------------------
        # Randomized RL timestep
        # -------------------------------------------------------------

        self.time_ratio = self.np_random.uniform(
            *self.options["random_time_ratio"]
        )

        # -------------------------------------------------------------
        # Floor friction
        # -------------------------------------------------------------

        self.sim.set_floor_friction(
            self.np_random.uniform(
                *self.options["random_friction"]
            )
        )

        # -------------------------------------------------------------
        # Trunk mass
        # -------------------------------------------------------------

        self.sim.model.body(
            self.trunk_body
        ).mass = (
            self.trunk_mass
            * self.np_random.uniform(
                *self.options["random_body_mass"]
            )
        )

        # -------------------------------------------------------------
        # Trunk centre of mass
        # -------------------------------------------------------------

        self.sim.model.body(
            self.trunk_body
        ).ipos = (
            self.trunk_ipos
            + self.np_random.uniform(
                -self.options["random_body_com_pos"],
                self.options["random_body_com_pos"],
                size=3,
            )
        )

        # -------------------------------------------------------------
        # Voltage randomization
        # -------------------------------------------------------------
        #
        # Not used for X2.
        #
        # X2 is being controlled through torque actuators and
        # the original FRASA voltage model is not applicable here.
        #

        # -------------------------------------------------------------
        # Damping
        # -------------------------------------------------------------

        damping_ratio = self.np_random.uniform(
            *self.options["random_damping"]
        )

        self.sim.model.dof_damping[:] = (
            self.damping_original
            * damping_ratio
        )

        # -------------------------------------------------------------
        # Friction loss
        # -------------------------------------------------------------

        frictionloss_ratio = (
            self.np_random.uniform(
                *self.options["random_frictionloss"]
            )
        )

        self.sim.model.dof_frictionloss[:] = (
            self.frictionloss_original
            * frictionloss_ratio
        )

        # -------------------------------------------------------------
        # Restore original body orientations
        # -------------------------------------------------------------

        self.sim.model.body_quat[:] = (
            self.body_quat_original.copy()
        )

        # -------------------------------------------------------------
        # Random angular errors
        # -------------------------------------------------------------
        #
        # Apply to the actual X2 joints rather than relying on
        # actuator names.
        #

        err_rad = np.deg2rad(
            self.options["random_angles"]
        )

        for joint in (
            self.left_joint_names
            + self.right_joint_names
        ):
            self.apply_angular_offset(
                joint,
                self.np_random.uniform(
                    -err_rad,
                    err_rad,
                ),
            )

    # =================================================================
    # RANDOM FALL CONFIGURATION
    # =================================================================

    def randomize_fall(
        self,
        target: bool = False,
    ):
        """
        Place X2 in a randomized fallen configuration.

        IMPORTANT:

        This function directly changes qpos.

        It must NOT use apply_control(), because data.ctrl is
        torque for the new X2 environment.
        """

        # -------------------------------------------------------------
        # Decide whether to start close to target
        # -------------------------------------------------------------

        my_target = np.copy(
            self.options["desired_state"]
        )

        if target is False:
            target = (
                self.np_random.random()
                < self.options["reset_final_p"]
            )

        # -------------------------------------------------------------
        # Random left-side configuration
        # -------------------------------------------------------------

        initial_q = self.np_random.uniform(
            low=-np.pi,
            high=np.pi,
            size=len(self.dofs),
        )

        # -------------------------------------------------------------
        # Target configuration
        # -------------------------------------------------------------

        if target:

            initial_q = np.copy(
                my_target[
                    :len(self.dofs)
                ]
            )

            offset = self.np_random.uniform(
                -0.1,
                0.1,
            )

            initial_q[2] -= offset
            initial_q[3] += 2 * offset
            initial_q[4] -= offset

        # -------------------------------------------------------------
        # Limit left configuration
        # -------------------------------------------------------------

        initial_q = np.clip(
            initial_q,
            self.range_low,
            self.range_high,
        )

        # -------------------------------------------------------------
        # Physical right-side configuration
        # -------------------------------------------------------------
        #
        # RL coordinate convention is symmetric.
        #
        # Shoulder roll is physically mirrored.
        #

        right_q = initial_q.copy()

        right_q[2] *= -1.0

        # -------------------------------------------------------------
        # Set actual joint positions
        # -------------------------------------------------------------

        for k, joint_name in enumerate(
            self.left_joint_names
        ):
            self.sim.set_q(
                joint_name,
                initial_q[k],
            )

        for k, joint_name in enumerate(
            self.right_joint_names
        ):
            self.sim.set_q(
                joint_name,
                right_q[k],
            )

        # -------------------------------------------------------------
        # No torque during initial fall/stabilization
        # -------------------------------------------------------------

        self.sim.data.ctrl[:] = 0

        mujoco.mj_forward(
            self.sim.model,
            self.sim.data,
        )

        # -------------------------------------------------------------
        # Initial trunk orientation
        # -------------------------------------------------------------

        initial_tilt = self.np_random.uniform(
            -np.pi / 2,
            np.pi / 2,
        )

        if target:
            initial_tilt = my_target[-1]

        T_world_trunk = (
            tf.rotation_matrix(
                initial_tilt,
                [0, 1, 0],
            )
        )

        T_world_trunk[:3, 3] = [
            0,
            0,
            0.4,
        ]

        self.sim.set_T_world_site(
            "trunk",
            T_world_trunk,
        )

        # -------------------------------------------------------------
        # Let robot settle under gravity
        # -------------------------------------------------------------

        for _ in range(
            round(
                self.options["stabilization_time"]
                / self.sim.dt
            )
        ):
            self.sim.data.ctrl[:] = 0
            self.sim.step()

    # =================================================================
    # RESET
    # =================================================================

    def reset(
        self,
        seed: int = None,
        options: Optional[dict] = None,
    ):
        """
        Reset the environment.

        After reset:

            actual q
                ↓
            q_desired = actual q
                ↓
            PD position error = 0

        This prevents an artificial initial torque spike.
        """

        super().reset(seed=seed)

        self.sim.reset()

        options = options or {}

        target = options.get(
            "target",
            False,
        )

        use_cache = options.get(
            "use_cache",
            True,
        )

        # -------------------------------------------------------------
        # Initial configuration cache
        # -------------------------------------------------------------

        if (
            use_cache
            and self.initial_config is None
        ):
            warnings.warn(
                "use_cache=True but no initial config "
                "file could be loaded. Did you run "
                "standup_generate_initial.py?"
            )

        # -------------------------------------------------------------
        # Restore cached configuration
        # -------------------------------------------------------------

        if (
            use_cache
            and self.initial_config is not None
        ):
            qpos, *_ = self.initial_config[
                self.np_random.integers(
                    len(self.initial_config)
                )
            ]

            # qpos is useful; old cached ctrl is NOT.
            #
            # data.ctrl now represents torque.
            #

            self.sim.data.qpos[:] = qpos

            self.sim.data.qvel[:] = 0

            self.sim.data.ctrl[:] = 0

            mujoco.mj_forward(
                self.sim.model,
                self.sim.data,
            )

        # -------------------------------------------------------------
        # Generate new fallen configuration
        # -------------------------------------------------------------

        else:
            self.randomize_fall(
                target=target
            )

        # -------------------------------------------------------------
        # Domain randomization
        # -------------------------------------------------------------

        self.apply_randomization()

        # Recompute MuJoCo derived quantities after modifying
        # body parameters/orientations.
        mujoco.mj_forward(
            self.sim.model,
            self.sim.data,
        )

        # -------------------------------------------------------------
        # Put arms slightly inward
        # -------------------------------------------------------------
        #
        # Physical convention:
        #
        # left shoulder roll  = +5 deg
        # right shoulder roll = -5 deg
        #

        self.sim.set_q(
            "left_shoulder_roll_joint",
            np.deg2rad(5),
        )

        self.sim.set_q(
            "right_shoulder_roll_joint",
            -np.deg2rad(5),
        )

        mujoco.mj_forward(
            self.sim.model,
            self.sim.data,
        )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # Initialize q_desired from the actual robot configuration.
        #
        # This means:
        #
        # q_desired == q
        #
        # at the beginning of the episode.
        # -------------------------------------------------------------

        q = np.array(
            [
                self.sim.get_q(name)
                for name in self.left_joint_names
            ],
            dtype=np.float64,
        )

        self.q_desired = q.copy()

        # -------------------------------------------------------------
        # Initialize q history
        # -------------------------------------------------------------

        self.q_history = [
            q.copy()
            for _ in range(
                self.q_history_size
            )
        ]

        # -------------------------------------------------------------
        # Initialize tilt history
        # -------------------------------------------------------------

        initial_tilt = self.get_tilt()

        self.tilt_history = [
            initial_tilt
            for _ in range(
                self.tilt_history_size
            )
        ]

        # -------------------------------------------------------------
        # Initialize tilt velocity history
        # -------------------------------------------------------------

        self.dtilt_history = [
            0.0
            for _ in range(
                self.dtilt_history_size
            )
        ]

        # -------------------------------------------------------------
        # Initialize previous actions
        # -------------------------------------------------------------

        self.previous_actions = [
            np.zeros(
                len(self.dofs),
                dtype=np.float64,
            )
            for _ in range(
                self.options["previous_actions"]
            )
        ]

        # -------------------------------------------------------------
        # Initial centroidal force
        # -------------------------------------------------------------

        self.centroidal_force = (
            self.sim.centroidal_force()
        )

        # -------------------------------------------------------------
        # Reset episode clock
        # -------------------------------------------------------------
        #
        # randomize_fall() includes the stabilization period.
        # That period should NOT count toward the RL episode.
        #

        self.sim.frame = 0
        self.sim.t = 0.0

        # -------------------------------------------------------------
        # Reset render state
        # -------------------------------------------------------------

        self.sim.reset_render()

        # -------------------------------------------------------------
        # Return initial observation
        # -------------------------------------------------------------

        return (
            self.get_observation(),
            {},
        )

    # =================================================================
    # RENDER
    # =================================================================

    def render(self):
        self.render_mode = "human"