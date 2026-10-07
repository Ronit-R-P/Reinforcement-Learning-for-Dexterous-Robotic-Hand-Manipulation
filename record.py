import mujoco
import mujoco.viewer
import numpy as np
import time, os
from datetime import datetime

model = mujoco.MjModel.from_xml_path("debug_hand.xml")
data = mujoco.MjData(model)

os.makedirs("records", exist_ok=True)
save_path = os.path.join("records", f"hand_log_{datetime.now():%Y%m%d_%H%M%S}.npy")

log = []

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            log.append({
                "qpos": data.qpos.copy(),
                "ctrl": data.ctrl.copy(),
                "sensordata": data.sensordata.copy(),
            })

            viewer.sync()
            time.sleep(model.opt.timestep)
finally:
    np.save(save_path, log, allow_pickle=True)
    print(f"Saved {len(log)} steps to {os.path.abspath(save_path)}")

