import numpy as np
import mujoco
import mujoco.viewer
import time
 
XML_PATH = "F:\MuJoCo\HandV4\mujoco hand\HandV5.xml"   # change this if your file is named/located differently
 
# ---------------------------------------------------------------------------
# Step 1: object geometry -> principal curvature
# ---------------------------------------------------------------------------
DIAMETER = 0.06          # 60 mm
RADIUS = DIAMETER / 2     # 30 mm
HEIGHT = 2 * DIAMETER      # 120 mm
 
def cylinder_principal_curvatures(radius):
    return 1.0 / radius, 0.0
 
kappa1, kappa2 = cylinder_principal_curvatures(RADIUS)
print(f"Cylinder: radius={RADIUS*1000:.0f}mm, height={HEIGHT*1000:.0f}mm")
print(f"Principal curvatures: kappa1={kappa1:.4f} (1/m), kappa2={kappa2:.4f} (1/m)\n")
 
# ---------------------------------------------------------------------------
# Step 2: curvature -> joint angles (real link lengths, from handV5.xml body offsets)
# L2 (PIP&DIP-to-fingertip) is NOT given explicitly in the XML -- approximated as
# 0.78 * L1, matching the proximal:distal proportion used in earlier testing.
# Replace with your real CAD value here if you have it, for a more accurate result.
# ---------------------------------------------------------------------------
FINGERS = {
    "Index":  {"L1": 0.0450, "mcp": "Index_MCP_act",  "pip": "Index_PIP_DIP_act"},
    "Middle": {"L1": 0.0485, "mcp": "Middle_MCP_act", "pip": "Middle_PIP_DIP_act"},
    "Ring":   {"L1": 0.0450, "mcp": "Ring_MCP_act",   "pip": "Ring_PIP_DIP_act"},
    "Pinky":  {"L1": 0.0435, "mcp": "Pinky_MCP_act",  "pip": "Pinky_PIP_DIP_act"},
}
L2_RATIO = 0.78
 
# ---------------------------------------------------------------------------
# Step 3: load the real model, exactly as-is
# ---------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
 
def act_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
 
# Build target ctrl array, using the model's OWN ctrlrange for clipping (not hardcoded)
target_ctrl = np.zeros(model.nu)
print(f"{'Finger':8} {'theta_MCP(deg)':>14} {'theta_PIP(deg)':>14}")
for name, f in FINGERS.items():
    L1 = f["L1"]
    L2 = L1 * L2_RATIO
    theta_mcp = -L1 * kappa1
    theta_pip = -L2 * kappa1
    print(f"{name:8} {np.degrees(theta_mcp):14.2f} {np.degrees(theta_pip):14.2f}")
 
    i_mcp = act_id(f["mcp"])
    i_pip = act_id(f["pip"])
    lo, hi = model.actuator_ctrlrange[i_mcp]
    target_ctrl[i_mcp] = np.clip(theta_mcp, lo, hi)
    lo, hi = model.actuator_ctrlrange[i_pip]
    target_ctrl[i_pip] = np.clip(theta_pip, lo, hi)
 
# Thumb: axis/mounting geometry differs from the fingers (not a simple same-axis
# planar chain) -- the cylinder-wrap formula above does not apply without a
# separate derivation. Left at 0 (neutral), not curvature-derived.
target_ctrl[act_id("Thumb_Knuckle_act")] = 0.0
target_ctrl[act_id("Thumb_MCP_act")] = 0.0
 
print("\nFull target_ctrl (model's own actuator order):")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {name:20s}: {target_ctrl[i]:+.4f} rad")
 
# ---------------------------------------------------------------------------
# Step 4: open the interactive viewer and command the pose
# ---------------------------------------------------------------------------
data.ctrl[:] = target_ctrl
 
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\nViewer open. Driving hand to curvature-derived pose... (close window to exit)")
    start = time.time()
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)