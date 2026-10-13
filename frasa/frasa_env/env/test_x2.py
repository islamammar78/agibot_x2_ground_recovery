#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco

from mujoco_simulator.simulator import Simulator

sim = Simulator()
print("\n==============================")
print("FRASA SIMULATOR INITIALISATION")
print("==============================\n")

# --------------------------------------------------
# 1. Model directory
# --------------------------------------------------
print("1. model_dir")
print("   ", sim.model_dir)

# --------------------------------------------------
# 2. MuJoCo model
# --------------------------------------------------
print("\n2. MuJoCo model")
print("   model:", sim.model)
print("   model type:", type(sim.model))

# --------------------------------------------------
# 3. MuJoCo data
# --------------------------------------------------
print("\n3. MuJoCo data")
print("   data:", sim.data)
print("   data type:", type(sim.data))

# --------------------------------------------------
# 4. Number of joints
# --------------------------------------------------
joints = len(sim.model.jnt_pos)

print("\n4. Number of joints")
print("   joints =", joints)

# --------------------------------------------------
# 5. DOFs
# --------------------------------------------------
print("\n5. DOFs")
print("   self.dofs =")
for dof in sim.dofs:
    print("   ", dof)

# --------------------------------------------------
# 6. DOF → MuJoCo joint index mapping
# --------------------------------------------------
print("\n6. dofs_to_index")
for name, index in sim.dofs_to_index.items():
    print(f"   {name} -> {index}")

# --------------------------------------------------
# 7. Viewer
# --------------------------------------------------
print("\n7. viewer")
print("   ", sim.viewer)

# --------------------------------------------------
# 8. Simulation time
# --------------------------------------------------
print("\n8. t")
print("   ", sim.t)

# --------------------------------------------------
# 9. MuJoCo timestep
# --------------------------------------------------
print("\n9. dt")
print("   ", sim.dt)

# --------------------------------------------------
# 10. Frame counter
# --------------------------------------------------
print("\n10. frame")
print("   ", sim.frame)

# --------------------------------------------------
# 11. Control vector
# --------------------------------------------------
print("\n11. data.ctrl")
print("   shape:", sim.data.ctrl.shape)
print("   values:", sim.data.ctrl)

# --------------------------------------------------
# 12. Extra useful MuJoCo information
# --------------------------------------------------
print("\n==============================")
print("ADDITIONAL MUJOCO INFORMATION")
print("==============================")

print("\nNumber of bodies:")
print("   model.nbody =", sim.model.nbody)

print("\nNumber of joints:")
print("   model.njnt =", sim.model.njnt)

print("\nNumber of actuators:")
print("   model.nu =", sim.model.nu)

print("\nNumber of sensors:")
print("   model.nsensor =", sim.model.nsensor)

print("\nPhysics timestep:")
print("   model.opt.timestep =", sim.model.opt.timestep)

print("\nGravity:")
print("   model.opt.gravity =", sim.model.opt.gravity)

print("\nIntegrator:")
print("   model.opt.integrator =", sim.model.opt.integrator)

print("\nSolver iterations:")
print("   model.opt.iterations =", sim.model.opt.iterations)

print("\n==============================")
print("ACTUATORS")
print("==============================")

for i in range(sim.model.nu):
    name = sim.model.actuator(i).name
    print(f"   actuator {i}: {name}")

print("\n==============================")
print("SENSORS")
print("==============================")

for i in range(sim.model.nsensor):
    name = sim.model.sensor(i).name
    print(f"   sensor {i}: {name}")

print("\n==============================")
print("JOINTS")
print("==============================")

for i in range(sim.model.njnt):
    name = sim.model.jnt(i).name
    joint_type = sim.model.jnt_type[i]

    print(f"   joint {i}: {name} | type = {joint_type}")

print("\n==============================")
print("INITIAL STATE")
print("==============================")

print("\nqpos:")
print(sim.data.qpos)

print("\nqvel:")
print(sim.data.qvel)

print("\nctrl:")
print(sim.data.ctrl)

print("\n==============================")
print("TESTING SIMULATOR FUNCTIONS")
print("==============================")

# ==================================================
# 1. Test set_floor_friction()
# ==================================================

print("\n--- 1. set_floor_friction() ---")

# Read the original floor friction
original_friction = sim.model.geom("floor").friction[0]

print("Original floor friction:")
print("   ", original_friction)

# Choose a test value
test_friction = 0.5

# Change the floor friction
sim.set_floor_friction(test_friction)

# Read it back
new_friction = sim.model.geom("floor").friction[0]

print("Requested friction:")
print("   ", test_friction)

print("Actual floor friction after function:")
print("   ", new_friction)

print("Floor priority:")
print("   ", sim.model.geom("floor").priority)


# ==================================================
# 2. Test self_collisions()
# ==================================================

print("\n--- 2. self_collisions() ---")

collision_force = sim.self_collisions()

print("Self-collision force:")
print("   ", collision_force)


# ==================================================
# 3. Test centroidal_force()
# ==================================================

print("\n--- 3. centroidal_force() ---")

centroidal_force = sim.centroidal_force()

print("Centroidal constraint force:")
print("   ", centroidal_force)


# ==================================================
# 4. Step the simulation and test again
# ==================================================

print("\n==============================")
print("AFTER ONE MUJOCO STEP")
print("==============================")

sim.step()

print("\nSelf-collision force:")
print("   ", sim.self_collisions())

print("\nCentroidal constraint force:")
print("   ", sim.centroidal_force())


print("\n==============================")
print("GEOMETRY IDS")
print("==============================")

for i in range(sim.model.ngeom):
    geom_name = sim.model.geom(i).name
    body_id = sim.model.geom_bodyid[i]
    body_name = sim.model.body(body_id).name

    print(
        f"geom ID {i:3d}: "
        f"name='{geom_name}', "
        f"body='{body_name}'"
    )

floor = sim.model.geom("floor")

print("Floor ID:", floor.id)
print("Initial friction:", floor.friction[0])
print("Initial priority:", floor.priority)

sim.set_floor_friction(0.5)

print("New friction:", sim.model.geom("floor").friction[0])
print("New priority:", sim.model.geom("floor").priority)


print("\n==============================")
print("1. dof_names()")
print("==============================")

dofs = sim.dof_names()

print("Number of actuators:", len(dofs))
print("Actuator names:")

for i, name in enumerate(dofs):
    print(f"  {i}: {name}")


print("\n==============================")
print("2. get_range()")
print("==============================")

print("\n==============================")
print("JOINT RANGES")
print("==============================")

dofs = sim.dof_names()

for name in dofs:
    joint_range = sim.get_range(name)

    print(f"{name}")
    print(f"  Range: {joint_range}")
    print(f"  Lower: {joint_range[0]}")
    print(f"  Upper: {joint_range[1]}")
    print()


print("\n==============================")
print("3. reset_render()")
print("==============================")

# Change the render bookkeeping values first
sim.t = 123.456
sim.frame = 999

print("Before reset_render():")
print("  t     =", sim.t)
print("  frame =", sim.frame)

sim.reset_render()

print("After reset_render():")
print("  t     =", sim.t)
print("  frame =", sim.frame)
print("  viewer_start =", sim.viewer_start)


print("\n==============================")
print("4. reset()")
print("==============================")

# First advance the simulation so that there is some state to reset
sim.step()

print("After one simulation step:")
print("  frame =", sim.frame)
print("  t     =", sim.t)

print("Calling sim.reset()...")

sim.reset()

print("After sim.reset():")
print("  qpos =", sim.data.qpos)
print("  qvel =", sim.data.qvel)
print("  ctrl =", sim.data.ctrl)


# ============================================================
# SELECT JOINT / ACTUATOR
# ============================================================

joint_index = 5

joint_name = sim.model.joint(joint_index).name
actuator_name = sim.model.actuator(joint_index).name

print("\n========================================")
print("SELECTED JOINT / ACTUATOR")
print("========================================")

print("Joint index    :", joint_index)
print("Joint name     :", joint_name)

print("Actuator index :", joint_index)
print("Actuator name  :", actuator_name)


# ============================================================
# 1. get_q()
# ============================================================

print("\n========================================")
print("1. get_q()")
print("========================================")

q = sim.get_q(joint_name)

print("Joint:", joint_name)
print("q =", q, "rad")


# ============================================================
# 2. get_qdot()
# ============================================================

print("\n========================================")
print("2. get_qdot()")
print("========================================")

qdot = sim.get_qdot(joint_name)

print("Joint:", joint_name)
print("qdot =", qdot, "rad/s")


# ============================================================
# 3. set_q()
# ============================================================

print("\n========================================")
print("3. set_q()")
print("========================================")

new_q = q + 0.2

print("Original q:", sim.get_q(joint_name))
print("Setting q to:", new_q)

sim.set_q(joint_name, new_q)

print("q after set_q():", sim.get_q(joint_name))


# ============================================================
# 4. get_control()
# ============================================================

print("\n========================================")
print("4. get_control()")
print("========================================")

ctrl = sim.get_control(actuator_name)

print("Actuator:", actuator_name)
print("ctrl =", ctrl)


# ============================================================
# 5. get_actuator_index()
# ============================================================

print("\n========================================")
print("5. get_actuator_index()")
print("========================================")

actuator_index = sim.get_actuator_index(actuator_name)

print("Actuator:", actuator_name)
print("Actuator index:", actuator_index)


# ============================================================
# 6. COMPARE q AND ctrl
# ============================================================

print("\n========================================")
print("6. ACTUAL POSITION VS COMMAND")
print("========================================")

print("Actual q :", sim.get_q(joint_name))
print("ctrl     :", sim.get_control(actuator_name))

print("Difference:",
      sim.get_control(actuator_name) - sim.get_q(joint_name))


# ============================================================
# 7. STEP SIMULATION
# ============================================================

print("\n========================================")
print("7. STEP SIMULATION")
print("========================================")

print("Before step:")
print("q    =", sim.get_q(joint_name))
print("qdot =", sim.get_qdot(joint_name))
print("ctrl =", sim.get_control(actuator_name))

sim.step()

print("\nAfter step:")
print("q    =", sim.get_q(joint_name))
print("qdot =", sim.get_qdot(joint_name))
print("ctrl =", sim.get_control(actuator_name))


# ============================================================
# 1. Choose actuator index
# ============================================================

actuator_index = 7

print("\n========================================")
print("TESTING set_control()")
print("========================================")


# ============================================================
# 2. Get actuator name from index
# ============================================================

actuator_name = mujoco.mj_id2name(
    sim.model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    actuator_index
)

print("\nActuator index:", actuator_index)
print("Actuator name :", actuator_name)


# ============================================================
# 3. Get initial values
# ============================================================

print("\n--- INITIAL STATE ---")

ctrl = sim.data.ctrl[actuator_index]

print("ctrl =", ctrl)


# ============================================================
# 4. Find the corresponding joint
# ============================================================

joint_name = actuator_name


print("Joint name   :", joint_name)

q = sim.get_q(joint_name)
qdot = sim.get_qdot(joint_name)

print("q    =", q)
print("qdot =", qdot)


# ============================================================
# 5. set_control(reset=False)
# ============================================================

print("\n--- set_control(reset=False) ---")

new_value = q + 0.5

print("New value:", new_value)

sim.set_control(
    actuator_name,
    new_value,
    reset=False
)

print("\nAfter set_control(reset=False):")
print("q    =", sim.get_q(joint_name))
print("qdot =", sim.get_qdot(joint_name))
print("ctrl =", sim.data.ctrl[actuator_index])


# ============================================================
# 6. Step simulation
# ============================================================

print("\n--- sim.step() ---")

sim.step()

print("After sim.step():")
print("q    =", sim.get_q(joint_name))
print("qdot =", sim.get_qdot(joint_name))
print("ctrl =", sim.data.ctrl[actuator_index])


# ============================================================
# 7. set_control(reset=True)
# ============================================================

print("\n--- set_control(reset=True) ---")

new_value = sim.get_q(joint_name) + 0.5

print("New value:", new_value)

sim.set_control(
    actuator_name,
    new_value,
    reset=True
)

print("\nAfter set_control(reset=True):")
print("q    =", sim.get_q(joint_name))
print("qdot =", sim.get_qdot(joint_name))
print("ctrl =", sim.data.ctrl[actuator_index])

print("\n==============================")
print("TESTING get_gyro()")
print("==============================")

gyro = sim.get_gyro()

print("Gyro / body angular velocity:")
print(gyro)
print("Shape:", gyro.shape)
print("Units: rad/s")


print("Bodies:")
for i in range(sim.model.nbody):
    print(i, sim.model.body(i).name)

print("\nSites:")
for i in range(sim.model.nsite):
    print(i, sim.model.site(i).name)

print("\n==============================")
print("TESTING TRANSFORM FUNCTIONS")
print("==============================")

# Use names that actually exist in your X2 XML
body_name = "pelvis"       # change if needed
site_name = "imu_0"    # change if needed

print("\nWorld -> Body:")
T_body = sim.get_T_world_body(body_name)
print(T_body)

print("\nWorld -> Site:")
T_site = sim.get_T_world_site(site_name)
print(T_site)

print("\nWorld -> Floating Base:")
T_fbase = sim.get_T_world_fbase()
print(T_fbase)

print("\nTesting set_T_world_fbase():")
sim.set_T_world_fbase(T_fbase)
print("Successfully set floating base to its current transform.")

print("\n==============================")
print("TESTING step(), set_gravity(), render()")
print("==============================")

# ------------------------------
# Test set_gravity()
# ------------------------------
print("\n1. TESTING set_gravity()")

print("Gravity before:", sim.model.opt.gravity.copy())

new_gravity = np.array([0.0, 0.0, -10.1])
sim.set_gravity(new_gravity)

print("Gravity after :", sim.model.opt.gravity.copy())


# ------------------------------
# Test step()
# ------------------------------
print("\n2. TESTING step()")

print("Before step:")
print("  frame =", sim.frame)
print("  t     =", sim.t)

sim.step()

print("After step:")
print("  frame =", sim.frame)
print("  t     =", sim.t)


# ------------------------------
# Test render()
# ------------------------------
print("\n3. TESTING render()")

print("Launching/syncing viewer...")
sim.render(realtime=False)

print("render() completed successfully.")

# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("TEST COMPLETE")
print("========================================")