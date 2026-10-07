import numpy as np
import torch
import torch.nn as nn
import os
import glob
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# CONFIG
# ==========================================
DEMO_DIR = "records"
MODEL_SAVE_PATH = "bc_policy.pt"
NORMALIZE_PATH = "bc_normalize.npy"

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 3e-4
HIDDEN_LAYERS = [256, 256]
VAL_DEMO_FRACTION = 0.15   # CHANGED: fraction of *demo files* (not rows) held out for val
DECIMATE_EVERY = 5          # CHANGED: keep 1 in N frames overall, active or idle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ==========================================
# TRAINING CODE — only runs when executed directly
# ==========================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    demo_files = sorted(glob.glob(os.path.join(DEMO_DIR, "*_named.npy")))
    print(f"Found {len(demo_files)} demo files")

    # CHANGED: hold out whole demo files for validation, not a row-index slice
    n_val_demos = max(1, int(len(demo_files) * VAL_DEMO_FRACTION))
    val_demo_files = set(demo_files[-n_val_demos:])   # last N files by name/timestamp
    train_demo_files = [f for f in demo_files if f not in val_demo_files]
    print(f"Train demos: {len(train_demo_files)} | Val demos: {len(val_demo_files)}")
    for f in val_demo_files:
        print(f"  -> held out for val: {os.path.basename(f)}")

    actuator_order = ['Index_MCP_act', 'Index_PIP_DIP_act',
                      'Middle_MCP_act', 'Middle_PIP_DIP_act',
                      'Ring_MCP_act', 'Ring_PIP_DIP_act',
                      'Pinky_MCP_act', 'Pinky_PIP_DIP_act',
                      'Thumb_Knuckle_act', 'Thumb_MCP_act', 'mount_z_act']

    def build_obs_action(demo_file):
        data = np.load(demo_file, allow_pickle=True)
        obs_list, act_list = [], []
        kept = 0
        for i, row in enumerate(data):
            # CHANGED: uniform decimation instead of idle-only subsampling.
            # Keeps 1 in DECIMATE_EVERY frames regardless of active/idle state,
            # which addresses redundancy across the WHOLE trajectory, not just
            # the idle portion.
            if i % DECIMATE_EVERY != 0:
                continue

            kept += 1
            tf = row['touch_forces']
            ctrl = row['controls']
            jp = row['joint_positions']
            jv = row['joint_velocities']
            # NOTE: 'actuator_forces' intentionally NOT read into obs anymore.
            # It's a near-direct algebraic function of qpos/qvel/ctrl for
            # position actuators, so including it let the network shortcut
            # around learning a real contact-driven policy.

            obs = []
            joint_order = ['mount_z_joint', 'Index_MCP ', 'Index_PIP&DIP_Joint',
                           'MIddle MCP Joint', 'Middle PIP&DIP Joint',
                           'Ring MCP Joint', 'Ring PIP&DIP Joint',
                           'Pinky MCP', 'Pinky PIP&DIP Joint',
                           'Thumb Knuckle Joint', 'Thumb MCP Joint']

            for name in joint_order:
                obs.append(jp.get(name, 0.0))
            for name in joint_order:
                obs.append(jv.get(name, 0.0))
            # CHANGED: actuator_forces block removed entirely (was here, 11 dims)
            for name in ['index_dist_touch_s', 'middle_dist_touch_s', 'ring_dist_touch_s',
                         'pinky_dist_touch_s', 'thumb_dist_touch_s']:
                obs.append(tf.get(name, 0.0))
            for name in ['index_prox_touch_s', 'middle_prox_touch_s', 'ring_prox_touch_s', 'pinky_prox_touch_s']:
                obs.append(tf.get(name, 0.0))
            obs.extend(row['cylinder_pos'])
            obs.extend(row['cylinder_quat'])
            obs.extend(row['cylinder_vel'])

            obs_list.append(obs)
            act_list.append([ctrl.get(name, 0.0) for name in actuator_order])

        print(f"  {os.path.basename(demo_file)}: {len(data)} -> {kept} kept ({kept*100/len(data):.0f}%)")
        return obs_list, act_list

    train_obs_raw, train_act_raw = [], []
    for f in train_demo_files:
        o, a = build_obs_action(f)
        train_obs_raw.extend(o)
        train_act_raw.extend(a)

    val_obs_raw, val_act_raw = [], []
    for f in val_demo_files:
        o, a = build_obs_action(f)
        val_obs_raw.extend(o)
        val_act_raw.extend(a)

    train_obs_np = np.array(train_obs_raw, dtype=np.float32)
    train_act_np = np.array(train_act_raw, dtype=np.float32)
    val_obs_np = np.array(val_obs_raw, dtype=np.float32)
    val_act_np = np.array(val_act_raw, dtype=np.float32)

    print(f"\nObs dim: {train_obs_np.shape[1]}  (should be 41, was 52 before)")
    print(f"Train samples: {len(train_obs_np)}, Val samples: {len(val_obs_np)}")

    # CHANGED: normalization stats computed from TRAIN split only (was: all data
    # combined before splitting, which let val statistics leak into normalization)
    obs_mean = train_obs_np.mean(axis=0)
    obs_std = train_obs_np.std(axis=0)
    obs_std[obs_std < 1e-6] = 1.0

    act_mean = train_act_np.mean(axis=0)
    act_std = train_act_np.std(axis=0)
    act_std[act_std < 1e-6] = 1.0

    np.save(NORMALIZE_PATH, {
        'obs_mean': obs_mean, 'obs_std': obs_std,
        'act_mean': act_mean, 'act_std': act_std
    })

    train_obs_norm = (train_obs_np - obs_mean) / obs_std
    train_act_norm = (train_act_np - act_mean) / act_std
    val_obs_norm = (val_obs_np - obs_mean) / obs_std
    val_act_norm = (val_act_np - act_mean) / act_std

    train_obs = torch.FloatTensor(train_obs_norm)
    train_actions = torch.FloatTensor(train_act_norm)
    val_obs = torch.FloatTensor(val_obs_norm)
    val_actions = torch.FloatTensor(val_act_norm)

    train_loader = DataLoader(TensorDataset(train_obs, train_actions), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_obs, val_actions), batch_size=BATCH_SIZE)

    model = BCPolicy(train_obs_np.shape[1], train_act_np.shape[1], HIDDEN_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # CHANGED: added weight_decay
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for obs_batch, act_batch in train_loader:
            obs_batch, act_batch = obs_batch.to(DEVICE), act_batch.to(DEVICE)
            pred = model(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch, act_batch = obs_batch.to(DEVICE), act_batch.to(DEVICE)
                pred = model(obs_batch)
                val_loss += loss_fn(pred, act_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    print(f"\nDone. Best epoch: {best_epoch} | Val loss: {best_val_loss:.6f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")