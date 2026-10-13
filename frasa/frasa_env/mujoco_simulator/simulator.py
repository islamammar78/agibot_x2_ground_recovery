"""Thin wrapper around MuJoCo used by the stand-up environment.

It loads scene.xml (X2 Ultra + floor), gives name-based access to joints,
actuators, sensors and body/site poses, and provides the PD controller that
turns desired joint positions into torques for the torque motors.
"""
import os
import time
from typing import Optional

import meshcat.transformations as tf
import mujoco
import mujoco.viewer
import numpy as np


class Simulator:
    """MuJoCo model/data pair plus helpers used by FRASAEnv."""

    def __init__(self, model_dir: Optional[str] = None):
        # If model_dir is not provided, use the current directory
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__) + "/model/")
        self.model_dir = model_dir

        # Load the model and data
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(f"{model_dir}/scene.xml")
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        # (joint id, joint name) pairs. Joint 0 is the floating base, so it is skipped.
        joints = len(self.model.jnt_pos)
        self.dofs = [[k, self.model.jnt(k).name] for k in range(1, joints)]
        self.dofs_to_index = {dof: k for k, dof in self.dofs}

        self.viewer = None  # created lazily by render()
        self.t: float = 0.0
        # Physics step (0.002 s); the env runs several steps per control step.
        self.dt: float = self.model.opt.timestep
        self.frame: int = 0
        self.data.ctrl[:] = 0

    def set_floor_friction(self, friction: float) -> None:
        """Set the floor sliding friction (used by domain randomization)."""
        self.model.geom("floor").friction[0] = friction
        self.model.geom("floor").priority = 1

    def self_collisions(self) -> float:
        """Sum of contact force norms between robot geoms (neither geom is the floor, geom 0).

        Used by the collision term of the reward.
        """
        forcetorque = np.zeros(6)
        contacts = self.data.contact
        selector = (contacts.geom[:, 0] != 0) * (contacts.geom[:, 1] != 0)
        forces = 0.0
        for id in np.argwhere(selector):
            mujoco.mj_contactForce(self.model, self.data, id, forcetorque)
            forces += np.linalg.norm(forcetorque[:3])

        return forces

    def centroidal_force(self) -> float:
        """Norm of the constraint (contact) force on the floating base."""
        return np.linalg.norm(self.data.qfrc_constraint[3:])

    def dof_names(self) -> list:
        """Names of all actuators, in actuator order."""
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

    def reset(self) -> None:
        """Reset the MuJoCo state to the model defaults."""
        mujoco.mj_resetData(self.model, self.data)

    def reset_render(self) -> None:
        """Restart the clock used to render in real time."""
        self.t = 0
        self.frame = 0
        self.viewer_start = time.time()

    def get_range(self, name: str) -> np.ndarray:
        """Joint limits [low, high] in rad."""
        return self.model.joint(name).range
    
    def get_q(self, name: str) -> float:
        """
        Gets the position of a given joint.

        Args:
            name (str): joint name

        Returns:
            float: joint position
        """
        addr = self.model.jnt_qposadr[self.dofs_to_index[name]]
        return self.data.qpos[addr]

    def get_qdot(self, name: str) -> float:
        """
        Gets the velocity of a given joint.

        Args:
            name (str): joint name

        Returns:
            float: joint velocity
        """
        addr = self.model.jnt_dofadr[self.dofs_to_index[name]]
        return self.data.qvel[addr]

    def set_q(self, name: str, value: float) -> None:
        """
        Sets a value of a given joint.

        Args:
            name (str): joint name
            value (float): target value
        """
        addr = self.model.jnt_qposadr[self.dofs_to_index[name]]
        self.data.qpos[addr] = value

    def get_control(self, name: str) -> None:
        """
        Gets the control for a given actuator

        Args:
            name (str): actuator name
        """
        actuator_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return self.data.ctrl[actuator_idx]

    def get_actuator_index(self, name: str) -> int:
        """Index of an actuator in data.ctrl."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def set_control(self, name: str, value: float, reset: bool = False) -> None:
        """
        Sets the control for a given actuator.
        If the actuator is a position actuator, the value is the desired position.
        If the actuator is a motor actuator, the value is the desired torque.

        Args:
            name (str): actuator name
            value (float): target value
        """
        actuator_idx = self.get_actuator_index(name)
        self.data.ctrl[actuator_idx] = value

        if reset:
            self.set_q(name, value)


    # X2: the original gyro sensor was replaced with the body-angular-velocity
    # sensor; the name get_gyro() is kept for compatibility with FRASA.
    def get_gyro(self) -> np.ndarray:
        """
        Gets the gyroscope data.

        Returns:
            np.ndarray: gyroscope data
        """
        return self.data.sensor("body-angular-velocity").data

    def get_T_world_body(self, body_name: str) -> np.ndarray:
        """
        Gets the transformation from world to body frame.

        Args:
            body_name (str): body name
        """
        T = np.eye(4)
        body = self.data.body(body_name)
        T[:3, :3] = body.xmat.reshape(3, 3)
        T[:3, 3] = body.xpos
        return T

    def get_T_world_site(self, site_name: str) -> np.ndarray:
        """
        Gets the transformation from world to site frame.

        Args:
            site_name (str): site name
        """
        T = np.eye(4)
        site = self.data.site(site_name)
        T[:3, :3] = site.xmat.reshape(3, 3)
        T[:3, 3] = site.xpos
        return T

    # Old Sigmaban version (free joint called "root"), kept for reference:
    """def get_T_world_fbase(self) -> np.ndarray:
        
        #Gets the transformation from world to floating base frame.
        
        data = self.data.joint("root").qpos
        quat = data[3:]
        pos = data[:3]

        T = tf.quaternion_matrix(quat)
        T[:3, 3] = pos
        return T"""
    
    # X2 version: the free joint is called "floating_base_joint".
    def get_T_world_fbase(self) -> np.ndarray:
            """
            Gets the transformation from world to floating base frame.
            """
            data = self.data.joint("floating_base_joint").qpos
            quat = data[3:]
            pos = data[:3]
    
            T = tf.quaternion_matrix(quat)
            T[:3, 3] = pos
            return T

    # Old Sigmaban version, kept for reference:
    """def set_T_world_fbase(self, T: np.ndarray) -> None:
        
        #Updates the floating base so that a body transformation match the target one

        Args:
            #T (np.ndarray): target transformation
        
        joint = self.data.joint("root")

        quat = tf.quaternion_from_matrix(T)
        pos = T[:3, 3]

        joint.qpos[:] = [*pos, *quat]
        self.reset_velocity()"""

    def set_T_world_fbase(self, T: np.ndarray) -> None:
            """
            Updates the floating base so that a body transformation match the target one
    
            Args:
                T (np.ndarray): target transformation
            """
            joint = self.data.joint("floating_base_joint")
    
            quat = tf.quaternion_from_matrix(T)
            pos = T[:3, 3]
    
            joint.qpos[:] = [*pos, *quat]
            self.reset_velocity()

    def reset_velocity(self) -> None:
        """
        Resets the velocity of all the joints
        """
        self.data.qvel[:] = 0

    def set_T_world_body(self, body_name: str, T_world_bodyTarget: np.ndarray) -> None:
        """
        Updates the floating base so that a body transformation match the target one

        Args:
            body_name (str): body name
        """
        T_world_fbase = self.get_T_world_fbase()
        T_world_body = self.get_T_world_body(body_name)
        T_body_fbase = np.linalg.inv(T_world_body) @ T_world_fbase

        self.set_T_world_fbase(T_world_bodyTarget @ T_body_fbase)

    def set_T_world_site(self, site_name: str, T_world_siteTarget: np.ndarray) -> None:
        """
        Updates the floating base so that a site transformation match the target one

        Args:
            site_name (str): site name
        """
        T_world_fbase = self.get_T_world_fbase()
        T_world_site = self.get_T_world_site(site_name)
        T_site_fbase = np.linalg.inv(T_world_site) @ T_world_fbase

        self.set_T_world_fbase(T_world_siteTarget @ T_site_fbase)
    # Foot pressure sensors (Sigmaban cleat sensors). Not used anywhere in the codebase.
    def get_pressure_sensors(self) -> dict:
        left_pressures = [
            -self.data.sensor("left_foot_cleat_front_right").data[2],
            -self.data.sensor("left_foot_cleat_front_left").data[2],
            -self.data.sensor("left_foot_cleat_back_right").data[2],
            -self.data.sensor("left_foot_cleat_back_left").data[2],
        ]
        right_pressures = [
            -self.data.sensor("right_foot_cleat_front_right").data[2],
            -self.data.sensor("right_foot_cleat_front_left").data[2],
            -self.data.sensor("right_foot_cleat_back_right").data[2],
            -self.data.sensor("right_foot_cleat_back_left").data[2],
        ]

        return {"left": left_pressures, "right": right_pressures}

    def step(self) -> None:
        """Advance the physics by one timestep (self.dt)."""
        self.t = self.frame * self.dt
        mujoco.mj_step(self.model, self.data)
        self.frame += 1

    def set_gravity(self, gravity: np.ndarray) -> None:
        """
        Sets the gravity vector.

        Args:
            gravity (np.ndarray): gravity vector
        """
        self.model.opt.gravity[:] = gravity

    def render(self, realtime: bool = True):
        """
        Renders the visualization of the simulation.

        Args:
            realtime (bool, optional): if True, render will sleep to ensure real time viewing. Defaults to True.
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.reset_render()

        if realtime:
            current_ts = self.viewer_start + self.frame * self.dt
            to_sleep = current_ts - time.time()
            if to_sleep > 0:
                time.sleep(to_sleep)

        self.viewer.sync()

    # Added for the X2: the actuators are torque motors, so a PD controller converts
    # desired joint positions into torques: tau = kp*(q_des - q) - kd*q_dot.
    def apply_pd_control(
        self,
        joint_names,
        actuator_indices,
        q_desired,
        kp,
        kd,
    ):
        q = np.array([
            self.get_q(joint)
            for joint in joint_names
        ])

        q_dot = np.array([
            self.get_qdot(joint)
            for joint in joint_names
        ])

        torque = kp * (q_desired - q) - kd * q_dot

        # Saturate to each motor's torque limit (ctrlrange).
        torque = np.clip(
            torque,
            self.model.actuator_ctrlrange[actuator_indices, 0],
            self.model.actuator_ctrlrange[actuator_indices, 1],
        )

        # The torques only take effect at the next step().
        self.data.ctrl[actuator_indices] = torque


# Manual demo: hold the default pose with PD plus an extra ankle pitch feedback
# and show the viewer. Not used by training.
if __name__ == "__main__":
    sim = Simulator()
    mujoco.mj_forward(sim.model, sim.data)
    left_foot_pos = sim.get_T_world_site("left_foot")[:3, 3]
    freejoint_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    freejoint_qpos = sim.model.jnt_qposadr[freejoint_id]
    sim.data.qpos[freejoint_qpos : freejoint_qpos + 3] -= left_foot_pos
    mujoco.mj_forward(sim.model, sim.data)

    actuated_joint_ids = sim.model.actuator_trnid[:, 0]
    joint_qpos = sim.model.jnt_qposadr[actuated_joint_ids]
    joint_dof = sim.model.jnt_dofadr[actuated_joint_ids]
    standing_pose = sim.data.qpos[joint_qpos].copy()
    kp, kd = 120.0, 12.0
    ankle_actuators = [
        sim.get_actuator_index(f"{side}_ankle_pitch_joint")
        for side in ("left", "right")
    ]
    pelvis_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    gyro_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_SENSOR, "body-angular-velocity"
    )
    gyro_adr = sim.model.sensor_adr[gyro_id]
    start = time.time()
    while True:
        sim.data.ctrl[:] = np.clip(
            kp * (standing_pose - sim.data.qpos[joint_qpos])
            - kd * sim.data.qvel[joint_dof],
            sim.model.actuator_ctrlrange[:, 0],
            sim.model.actuator_ctrlrange[:, 1],
        )
        pelvis_rotation = sim.data.xmat[pelvis_id].reshape(3, 3)
        pitch = np.arctan2(pelvis_rotation[0, 2], pelvis_rotation[0, 0])
        pitch_rate = sim.data.sensordata[gyro_adr + 1]
        sim.data.ctrl[ankle_actuators] += 120.0 * pitch + 18.0 * pitch_rate
        sim.data.ctrl[:] = np.clip(
            sim.data.ctrl,
            sim.model.actuator_ctrlrange[:, 0],
            sim.model.actuator_ctrlrange[:, 1],
        )
        sim.render(True)
        #sim.set_control("head_yaw", np.sin(sim.t))
        #sim.set_control("head_pitch", np.sin(sim.t))
        sim.step()

        elapsed = time.time() - start
        frames = sim.frame
        print(f"Elapsed: {elapsed:.2f}, Frames: {frames}, FPS: {frames / elapsed:.2f}")
        print("timestep:", sim.model.opt.timestep)
        print("iterations:", sim.model.opt.iterations)
        print("integrator:", sim.model.opt.integrator)
