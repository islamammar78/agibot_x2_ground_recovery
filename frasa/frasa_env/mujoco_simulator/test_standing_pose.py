#!/usr/bin/env python3

import time

import mujoco
import mujoco.viewer
import numpy as np

from simulator import Simulator


# ============================================================
# Target standing pose
# ============================================================

TARGET_POS = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,

    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,

    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,

    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,

    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}


# ============================================================
# Create simulator
# ============================================================

sim = Simulator()

mujoco.mj_forward(sim.model, sim.data)

print()
print("Loaded MuJoCo model")
print(f"Number of joints    : {sim.model.njnt}")
print(f"Number of actuators : {sim.model.nu}")
print()


# ============================================================
# Find floating-base freejoint
# ============================================================

freejoint_id = None

for i in range(sim.model.njnt):

    if sim.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        freejoint_id = i
        break

if freejoint_id is None:
    raise RuntimeError("No freejoint found in the model.")

freejoint_qpos = sim.model.jnt_qposadr[freejoint_id]


# ============================================================
# Find body-orientation sensor
# ============================================================

body_orientation_sensor = sim.model.sensor("body-orientation")

body_orientation_sensor_adr = int(
    body_orientation_sensor.adr[0]
)

body_orientation_sensor_dim = int(
    body_orientation_sensor.dim[0]
)

print("Body orientation sensor")
print("-----------------------")
print(f"Address   : {body_orientation_sensor_adr}")
print(f"Dimension : {body_orientation_sensor_dim}")
print()


# ============================================================
# PD controller parameters
# ============================================================

kp = 120.0
kd = 12.0


# ============================================================
# Launch MuJoCo viewer
# ============================================================

with mujoco.viewer.launch_passive(
    sim.model,
    sim.data
) as viewer:

    last_print_time = time.time()

    while viewer.is_running():

        # ====================================================
        # PD control
        # ====================================================

        for joint_name, target_position in TARGET_POS.items():

            joint_id = sim.model.joint(joint_name).id

            qpos_address = sim.model.jnt_qposadr[joint_id]
            qvel_address = sim.model.jnt_dofadr[joint_id]

            current_position = sim.data.qpos[qpos_address]
            current_velocity = sim.data.qvel[qvel_address]

            torque = (
                kp * (target_position - current_position)
                - kd * current_velocity
            )

            # Find actuator controlling this joint
            for actuator_index in range(sim.model.nu):

                actuator_joint_id = sim.model.actuator_trnid[
                    actuator_index, 0
                ]

                if actuator_joint_id == joint_id:

                    sim.data.ctrl[actuator_index] = torque
                    break


        # ====================================================
        # Step simulation
        # ====================================================

        mujoco.mj_step(sim.model, sim.data)


        # ====================================================
        # QPOS-based pitch
        # ====================================================

        quat = sim.data.qpos[
            freejoint_qpos + 3:
            freejoint_qpos + 7
        ]

        rotation = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation,
            quat,
        )

        rotation = rotation.reshape(3, 3)

        imu_pitch = np.arctan2(
            -rotation[2, 0],
            np.sqrt(
                rotation[0, 0] ** 2
                + rotation[1, 0] ** 2
            )
        )


        # ====================================================
        # Body-orientation sensor
        # ====================================================

        sensor_quat = sim.data.sensordata[
            body_orientation_sensor_adr:
            body_orientation_sensor_adr
            + body_orientation_sensor_dim
        ]

        sensor_rotation = np.zeros(9)

        mujoco.mju_quat2Mat(
            sensor_rotation,
            sensor_quat,
        )

        sensor_rotation = sensor_rotation.reshape(3, 3)


        # Exact FRASA get_tilt() formula
        sensor_pitch = -np.arctan2(
            sensor_rotation[2, 0],
            sensor_rotation[2, 2]
        )


        # ====================================================
        # Print pitch values
        # ====================================================

        current_time = time.time()

        if current_time - last_print_time >= 0.5:

            print("----------------------------------------")

            print(
                f"FRASA sensor tilt : "
                f"{np.rad2deg(sensor_pitch):8.2f} deg"
            )

            print(
                f"QPOS pitch        : "
                f"{np.rad2deg(imu_pitch):8.2f} deg"
            )

            print(
                f"Difference        : "
                f"{np.rad2deg(sensor_pitch - imu_pitch):8.2f} deg"
            )

            last_print_time = current_time


        # ====================================================
        # Update viewer
        # ====================================================

        viewer.sync()

        time.sleep(0.001)