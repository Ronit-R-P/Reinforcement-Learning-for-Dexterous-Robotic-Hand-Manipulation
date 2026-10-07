import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("test_mount copy.xml")
data = mujoco.MjData(model)

# HARDCODED VALUES based on your XML's ctrlrange
# Format: (open_value, closed_value)
# MCP joints: min=-1.58 (closed), max=-0.135 (open)
# PIP/DIP joints: min varies (closed), max varies (open)
fingers = {
    "Pinky_MCP_act":       (-0.135, -1.58),   # (open, closed)
    "Pinky_PIP_DIP_act":   (-0.02, -1.79),    # (open, closed)
    "Ring_MCP_act":        (-0.135, -1.58),   # (open, closed)
    "Ring_PIP_DIP_act":    (0.26, -1.462),    # (open, closed)
    "Middle_MCP_act":      (-0.135, -1.7),    # (open, closed)
    "Middle_PIP_DIP_act":  (0.0369, -1.77),   # (open, closed)
    "Index_MCP_act":       (-0.135, -1.58),   # (open, closed)
    "Index_PIP_DIP_act":   (0.385, -1.375),   # (open, closed)
}

# Thumb: only move the MCP, not the knuckle
# Thumb MCP range in XML: -0.65 to 0.83
thumb_open = 0.0      # neutral position
thumb_closed = 0.6    # curl inward (positive because of joint orientation)

close_order = ["Pinky", "Ring", "Middle", "Index"]
stagger = 0.25
close_duration = 0.4
hold_time = 0.4
open_stagger = 0.25
open_duration = 0.5

def actuator_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    # Set initial positions - fingers open, thumb neutral
    for act_name in fingers.keys():
        aid = actuator_id(act_name)
        open_val, _ = fingers[act_name]
        data.ctrl[aid] = open_val
    
    # Set thumb MCP to neutral, thumb knuckle stays at 0
    thumb_mcp_id = actuator_id("Thumb_MCP_act")
    data.ctrl[thumb_mcp_id] = thumb_open
    
    thumb_knuckle_id = actuator_id("Thumb_Knuckle_act")
    data.ctrl[thumb_knuckle_id] = 0  # keep knuckle fixed

    while viewer.is_running():
        t = time.time() - start_time

        # CLOSING PHASE
        for i, finger in enumerate(close_order):
            finger_start = i * stagger
            local_t = (t - finger_start) / close_duration
            
            if 0 <= local_t <= 1:
                mcp = f"{finger}_MCP_act"
                pip = f"{finger}_PIP_DIP_act"
                
                open_val, closed_val = fingers[mcp]
                data.ctrl[actuator_id(mcp)] = lerp(open_val, closed_val, local_t)
                
                open_val2, closed_val2 = fingers[pip]
                data.ctrl[actuator_id(pip)] = lerp(open_val2, closed_val2, local_t)

        total_close_time = (len(close_order) - 1) * stagger + close_duration
        
        # THUMB closes after all fingers
        thumb_start = total_close_time + 0.1
        thumb_local_t = (t - thumb_start) / 0.3
        if 0 <= thumb_local_t <= 1:
            data.ctrl[thumb_mcp_id] = lerp(thumb_open, thumb_closed, thumb_local_t)
        
        reopen_start = total_close_time + hold_time + 0.2

        # OPENING PHASE
        open_order = list(reversed(close_order))
        for i, finger in enumerate(open_order):
            finger_start = reopen_start + i * open_stagger
            local_t = (t - finger_start) / open_duration
            
            if 0 <= local_t <= 1:
                mcp = f"{finger}_MCP_act"
                pip = f"{finger}_PIP_DIP_act"
                
                open_val, closed_val = fingers[mcp]
                data.ctrl[actuator_id(mcp)] = lerp(closed_val, open_val, local_t)
                
                open_val2, closed_val2 = fingers[pip]
                data.ctrl[actuator_id(pip)] = lerp(closed_val2, open_val2, local_t)

        # THUMB opens
        thumb_open_start = reopen_start + (len(open_order) - 1) * open_stagger + open_duration + 0.1
        thumb_open_t = (t - thumb_open_start) / 0.3
        if 0 <= thumb_open_t <= 1:
            data.ctrl[thumb_mcp_id] = lerp(thumb_closed, thumb_open, thumb_open_t)

        total_cycle = thumb_open_start + 0.3 + 0.3
        if t > total_cycle:
            start_time = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)