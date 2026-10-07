import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('debug_hand.xml')
data = mujoco.MjData(model)

# Simulate a few steps to let things settle
for _ in range(1000):
    mujoco.mj_step(model, data)

# Print initial sensor states
print("Initial touch sensor values (should be [0,0,0]):")
print(f"  Proximal: {data.sensor('index_prox_touch_s').data}")
print(f"  Distal:   {data.sensor('index_dist_touch_s').data}")

# Now you could send commands to the finger actuators to touch something
# and re-check the sensor values