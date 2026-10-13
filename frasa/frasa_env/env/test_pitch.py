#!/usr/bin/env python3

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from mujoco_simulator.simulator import Simulator


sim = Simulator()

print("\n========================================")
print("BODY-ORIENTATION SENSOR TEST")
print("========================================")


# ============================================================
# 1. Find body-orientation sensor
# ============================================================

sensor_name = "body-orientation"

sensor = sim.model.sensor(sensor_name)

sensor_id = sensor.id
sensor_adr = int(sensor.adr[0])
sensor_dim = int(sensor.dim[0])

print("\nSensor:")
print("  name      =", sensor_name)
print("  id        =", sensor_id)
print("  address   =", sensor_adr)
print("  dimension =", sensor_dim)
print("  type      =", sensor.type)


# ============================================================
# 2. Read raw sensor
# ============================================================

raw = sim.data.sensordata[
    sensor_adr : sensor_adr + sensor_dim
].copy()

print("\n========================================")
print("RAW SENSOR OUTPUT")
print("========================================")

print("Quaternion:")
print(raw)

print("\nIndividual values:")

for i, value in enumerate(raw):
    print(f"  [{i}] = {value}")


# ============================================================
# 3. Convert quaternion -> rotation matrix
# ============================================================

# MuJoCo framequat is:
#
# [w, x, y, z]
#
# Convert it to a 3x3 rotation matrix.

quat = raw

rotation_matrix = np.zeros((3, 3))

mujoco.mju_quat2Mat(
    rotation_matrix.reshape(-1),
    quat
)

print("\n========================================")
print("ROTATION MATRIX")
print("========================================")

print(rotation_matrix)


# ============================================================
# 4. Calculate pitch using FRASA's original method
# ============================================================

R = rotation_matrix

pitch_frasa = -np.arctan2(R[2, 0], R[2, 2])

print("\n========================================")
print("FRASA PITCH CALCULATION")
print("========================================")

print("R[2,0] =", R[2, 0])
print("R[2,2] =", R[2, 2])

print("\nPitch:")
print("  radians =", pitch_frasa)
print("  degrees =", np.rad2deg(pitch_frasa))


# ============================================================
# 5. Compare against trunk_site.xmat
# ============================================================

print("\n========================================")
print("COMPARISON WITH OLD FRASA METHOD")
print("========================================")

# Try to find the trunk site used by the original code.

try:

    trunk_site = sim.model.site("trunk")

    R_trunk = sim.data.site_xmat[trunk_site.id].reshape(3, 3)

    pitch_old = -np.arctan2(
        R_trunk[2, 0],
        R_trunk[2, 2]
    )

    print("trunk site rotation matrix:")
    print(R_trunk)

    print("\nOld get_tilt():")
    print("  radians =", pitch_old)
    print("  degrees =", np.rad2deg(pitch_old))

    print("\nSensor pitch:")
    print("  radians =", pitch_frasa)
    print("  degrees =", np.rad2deg(pitch_frasa))

    print("\nDifference:")
    print(
        "  degrees =",
        np.rad2deg(pitch_frasa - pitch_old)
    )

except Exception as e:

    print("Could not compare with trunk site.")
    print("Reason:", e)


# ============================================================
# 6. Step simulation and test again
# ============================================================

print("\n========================================")
print("AFTER ONE SIMULATION STEP")
print("========================================")

sim.step()

raw = sim.data.sensordata[
    sensor_adr : sensor_adr + sensor_dim
].copy()

print("Quaternion:")
print(raw)

rotation_matrix = np.zeros((3, 3))

mujoco.mju_quat2Mat(
    rotation_matrix.reshape(-1),
    raw
)

pitch = -np.arctan2(
    rotation_matrix[2, 0],
    rotation_matrix[2, 2]
)

print("\nPitch:")
print("  radians =", pitch)
print("  degrees =", np.rad2deg(pitch))


print("\n========================================")
print("TEST COMPLETE")
print("========================================")