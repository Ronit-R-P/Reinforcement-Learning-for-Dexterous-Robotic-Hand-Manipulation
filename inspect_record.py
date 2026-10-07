import numpy as np

log = np.load("records/hand_log_20260727_232926.npy", allow_pickle=True)
print("Steps:", len(log))

sensordata = np.array([entry["sensordata"] for entry in log])
touch = sensordata[:, 3:8]  # cols 3-7 = your 5 touch sensors (after 3 cylinder sensors: pos3+quat4+vel3=10... adjust index if different)

qpos_series = np.array([e["qpos"] for e in log])
ctrl_series = np.array([e["ctrl"] for e in log])

print("First row qpos:", qpos_series[0])
print("First row ctrl:", ctrl_series[0])

rand_idx = np.random.randint(len(log))
print(f"\nRandom row ({rand_idx}) qpos:", qpos_series[rand_idx])
print(f"Random row ({rand_idx}) ctrl:", ctrl_series[rand_idx])
# print(qpos_series.shape, ctrl_series.shape)
# print("Max touch values per sensor:", touch.max(axis=0))
# print("Any nonzero contact:", (touch > 0).any())