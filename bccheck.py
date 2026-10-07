import numpy as np
import torch
import torch.nn as nn
from env import DumbbellLiftEnv

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_layers):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

norm = np.load("bc_normalize.npy", allow_pickle=True).item()

model = BCPolicy(52, 11, [256, 256])
model.load_state_dict(torch.load("bc_policy.pt", map_location="cpu"))
model.eval()

env = DumbbellLiftEnv(xml_path="debug_hand.xml", max_steps=2000)
obs = env.reset()

act_low = env.ctrl_range_low
act_high = env.ctrl_range_high

actuator_names = ['IdxMCP', 'IdxPIP', 'MidMCP', 'MidPIP', 
                  'RngMCP', 'RngPIP', 'PnkMCP', 'PnkPIP',
                  'ThbKnk', 'ThbMCP', 'MntZ']

total_reward = 0
grip_viable = False

for step in range(2000):
    obs_norm = (obs - norm['obs_mean']) / norm['obs_std']
    
    with torch.no_grad():
        action_norm = model(torch.FloatTensor(obs_norm)).numpy()
    
    action = action_norm * norm['act_std'] + norm['act_mean']
    
    # Force mount_z to stay down until grip is established
    if not grip_viable:
        action[10] = act_low[10]
    
    action = np.clip(action, act_low, act_high)
    action_scaled = 2.0 * (action - act_low) / (act_high - act_low) - 1.0
    
    obs, reward, done, info = env.step(action_scaled)
    total_reward += reward
    grip_viable = info['grip_viable']
    
    if step % 200 == 0:
        act_dict = dict(zip(actuator_names, [f'{a:.3f}' for a in action]))
        print(f"Step {step}: reward={reward:.3f}, lift={info['lift_amount']*100:.1f}cm, "
              f"curled={info['all_curled']}, grip={info['grip_viable']}")
        print(f"  Actions: {act_dict}")
        jp = env.data.qpos
        print(f"  Joints: mntZ={jp[0]:.3f}, IdxMCP={jp[1]:.3f}, ThbKnk={jp[9]:.3f}")
        print(f"  Touch: idx={env.data.sensordata[10]:.1f}, mid={env.data.sensordata[11]:.1f}, thb={env.data.sensordata[14]:.1f}")
    
    if done:
        print(f"Done at step {step}, total reward={total_reward:.1f}")
        break