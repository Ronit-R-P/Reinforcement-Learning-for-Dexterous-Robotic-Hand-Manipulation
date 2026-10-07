import numpy as np

log = np.load("records/hand_log_20260728_233651.npy", allow_pickle=True)  # replace filename
sensor_arr = np.array([e["sensordata"] for e in log])
qpos_arr = np.array([e["qpos"] for e in log])
ctrl_arr = np.array([e["ctrl"] for e in log])

sensor_names = [i for i in range(19)]  # placeholder, real names below

import mujoco
model = mujoco.MjModel.from_xml_path("debug_hand.xml")
names = [model.sensor(i).name for i in range(model.nsensor)]
adrs = model.sensor_adr
dims = model.sensor_dim

def sensor_dict(row):
    return {names[i]: row[adrs[i]:adrs[i]+dims[i]] for i in range(model.nsensor)}

n_steps = len(log)
print(f"Total steps recorded: {n_steps}\n")

blocks = [(1000, 1010), (10000, 10010), (20000, 20010), (29990, 30000)]

for start, end in blocks:
    print(f"--- Rows {start} to {end-1} ---")
    for i in range(start, min(end, n_steps)):
        print(f"Row {i} | qpos[:3]: {qpos_arr[i][:3]} | ctrl[:3]: {ctrl_arr[i][:3]}")
        print(f"   sensors: {sensor_dict(sensor_arr[i])}")
    print()