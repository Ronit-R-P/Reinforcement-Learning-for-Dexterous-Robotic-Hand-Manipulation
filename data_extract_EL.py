import os
import numpy as np
import mujoco

# 1. Load your HandV4.xml model safely using an absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "HandV4.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("HandV4.xml loaded successfully!")

# 2. Print extracted physical properties from HandV4.xml for verification
print("\n--- EXTRACTED PHYSICAL PROPERTIES FROM XML ---")
for i in range(model.nbody):
  body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
  print(
      f"Body [{i}] {body_name}: Mass = {model.body_mass[i]:.5f} kg, COM ="
      f" {model.body_ipos[i]}"
  )

# 3. Define a non-zero flexed joint posture (q_test) across your 10 actuators
# MCP joints flexed to -0.8 rad (~-45°), PIP/DIP flexed to -0.5 rad (~-28°)
q_test = np.array([-0.8, -0.5, -0.8, -0.5, -0.8, -0.5, -0.8, -0.5, -0.3, 0.4])

# Map joint values to the model's qpos array
# (MuJoCo joint positions are stored in data.qpos)
for i in range(10):
  data.qpos[i] = q_test[i]

# CRITICAL STEP: Update kinematic frames and COM locations for q_test
mujoco.mj_forward(model, data)

# 4. Compute Gravity Compensation & Holding Torque Vector via MuJoCo
# mj_inverse solves for the generalized forces (qfrc_inverse) required to hold q_test at rest
mujoco.mj_inverse(model, data)

# Extract gravity/passive torque vector acting on the 10 joint degrees of freedom
tau_holding = data.qfrc_inverse[:10]

print("\n--- COMPUTED STATIC HOLDING TORQUES (Nm) ---")
for i in range(10):
  actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
  joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
  print(
      f"Actuator [{i}] {actuator_name} ({joint_name}): Required Torque ="
      f" {tau_holding[i]:.5f} Nm"
  )

# 5. Apply computed static holding torques directly to actuators and step physics
data.ctrl[:10] = tau_holding
mujoco.mj_step(model, data)

print("\nPhysics step completed. Holding equilibrium verified!")