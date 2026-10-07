import mujoco
import mujoco.viewer
import numpy as np
import os
from datetime import datetime

model = mujoco.MjModel.from_xml_path("debug_hand.xml")
data = mujoco.MjData(model)

os.makedirs("records", exist_ok=True)
save_path = os.path.join("records", f"hand_log_{datetime.now():%Y%m%d_%H%M%S}.npy")

# ---- Build name-to-address maps ----
# Sensors use adr (address) and dim (dimension), NOT id
sensor_map = {}
for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    if name:
        adr = model.sensor_adr[i]   # Starting position in sensordata array
        dim = model.sensor_dim[i]   # Number of elements this sensor uses
        sensor_map[name] = (adr, dim)

# Joints and actuators are 1:1 with their arrays — id works fine
joint_map = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
        joint_map[name] = i

actuator_map = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name:
        actuator_map[name] = i

# Print for verification
print("Sensor map (name → adr, dim):")
for name, (adr, dim) in sensor_map.items():
    print(f"  {name}: adr={adr}, dim={dim}")

log = []

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # ---- Build named row using correct addresses ----
            row = {
                'step': len(log),
                
                # Joint positions and velocities (scalar per joint, id works fine)
                'joint_positions': {name: float(data.qpos[idx]) for name, idx in joint_map.items()},
                'joint_velocities': {name: float(data.qvel[idx]) for name, idx in joint_map.items()},
                
                # Actuator forces and controls (scalar per actuator, id works fine)
                'actuator_forces': {name: float(data.actuator_force[idx]) for name, idx in actuator_map.items()},
                'controls': {name: float(data.ctrl[idx]) for name, idx in actuator_map.items()},
                
                # Touch sensors — scalar (dim=1), read single value at correct address
                'touch_forces': {
                    name: float(data.sensordata[adr])
                    for name, (adr, dim) in sensor_map.items() if 'touch' in name
                },
                
                # Cylinder state — use address + dimension for multi-value sensors
                'cylinder_pos':  data.sensordata[sensor_map['cylinder_pos'][0] : sensor_map['cylinder_pos'][0] + sensor_map['cylinder_pos'][1]].tolist(),
                'cylinder_quat': data.sensordata[sensor_map['cylinder_quat'][0] : sensor_map['cylinder_quat'][0] + sensor_map['cylinder_quat'][1]].tolist(),
                'cylinder_vel':  data.sensordata[sensor_map['cylinder_vel'][0] : sensor_map['cylinder_vel'][0] + sensor_map['cylinder_vel'][1]].tolist(),
                
                # Raw arrays (always correct, no address issues)
                'qpos_raw': data.qpos.copy(),
                'qvel_raw': data.qvel.copy(),
                'ctrl_raw': data.ctrl.copy(),
                'sensordata_raw': data.sensordata.copy(),
            }
            log.append(row)
            viewer.sync()

finally:
    np.save(save_path, np.array(log, dtype=object), allow_pickle=True)
    print(f"\nSaved {len(log)} steps to {os.path.abspath(save_path)}")
    
    # Quick verification on saved data
    if len(log) > 0:
        sample = log[len(log)//2]  # Middle step
        print(f"\nVerification (step {sample['step']}):")
        print(f"  cylinder_pos:  {sample['cylinder_pos']}")
        print(f"  cylinder_quat: {sample['cylinder_quat'][:2]}...")
        print(f"  cylinder_vel:  {sample['cylinder_vel']}")
        print(f"  touch_forces:  {dict(list(sample['touch_forces'].items())[:3])}")