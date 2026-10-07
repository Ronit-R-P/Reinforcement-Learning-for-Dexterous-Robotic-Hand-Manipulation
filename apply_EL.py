import mujoco
import numpy as np

# 1. Load model
model = mujoco.MjModel.from_xml_path('HandV4.xml')
data = mujoco.MjData(model)

# 2. Set any valid flexed pose (MUST be negative for MCP/PIP joints!)
# Index_MCP, Index_PIP, Middle_MCP, Middle_PIP, Ring_MCP, Ring_PIP, Pinky_MCP, Pinky_PIP, Thumb_K, Thumb_MCP
q_test = np.array([-0.5, -0.4, -0.5, -0.4, -0.5, -0.4, -0.5, -0.4, -0.2, 0.2])
data.qpos[:10] = q_test
mujoco.mj_forward(model, data)

# 3. Calculate holding torque via inverse dynamics (Euler-Lagrange)
mujoco.mj_inverse(model, data)
tau_holding = data.qfrc_inverse[:10].copy()

# 4. Apply torques directly to joint axes (bypassing XML position actuators)
data.qfrc_applied[:10] = tau_holding
data.qvel[:10] = 0 # Zero velocity

# 5. Run forward dynamics step to see resulting joint acceleration
mujoco.mj_forward(model, data)
joint_accelerations = data.qacc[:10]

print("Joint Accelerations under Euler-Lagrange Holding Torque:")
print(joint_accelerations)