import numpy as np
import torch
import torch.nn as nn

# Adjust these imports to match your actual filenames
from env import DumbbellLiftEnv          # your env wrapper file
from bc_train import BCPolicy            # reuse the same class definition

MODEL_PATH = "bc_policy.pt"
NORMALIZE_PATH = "bc_normalize.npy"
NUM_ROLLOUTS = 3
MAX_STEPS = 5000   # shorter than full 5000-step episodes is fine for a quick check
PRINT_EVERY = 250

def load_policy_and_norm(obs_dim, action_dim, hidden_layers=[256, 256]):
    norm = np.load(NORMALIZE_PATH, allow_pickle=True).item()
    model = BCPolicy(obs_dim, action_dim, hidden_layers)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, norm

def run_rollout(env, model, norm, rollout_idx):
    obs = env.reset()
    events_seen = set()

    print(f"\n{'='*70}\nROLLOUT {rollout_idx}\n{'='*70}")

    for step in range(MAX_STEPS):
        obs_norm = (obs - norm['obs_mean']) / norm['obs_std']
        with torch.no_grad():
            action_norm = model(torch.FloatTensor(obs_norm).unsqueeze(0)).squeeze(0).numpy()
        action = action_norm * norm['act_std'] + norm['act_mean']
        # BC output is in raw ctrl units already (since we denormalized against
        # act_mean/act_std, which were fit on raw ctrl values) -- env.step()
        # expects actions in [-1, 1] scaled space, so re-map back if needed.
        # If your env.step() instead expects raw ctrl directly, skip this line:
        action_clipped = np.clip(action, env.ctrl_range_low, env.ctrl_range_high)
        action_rescaled = 2.0 * (action_clipped - env.ctrl_range_low) / (env.ctrl_range_high - env.ctrl_range_low) - 1.0

        obs, reward, done, info = env.step(action_rescaled)

        # Track same milestones as your test_reward.py analytics
        if info['lift_amount'] > 0.01 and 'FIRST_LIFT' not in events_seen:
            events_seen.add('FIRST_LIFT')
            print(f"  step {step}: FIRST_LIFT (lift={info['lift_amount']*100:.2f}cm)")
        if info['all_curled'] and 'FULL_CURL' not in events_seen:
            events_seen.add('FULL_CURL')
            print(f"  step {step}: FULL_CURL")
        if info['grip_viable'] and 'FIRST_GRIP_VIABLE' not in events_seen:
            events_seen.add('FIRST_GRIP_VIABLE')
            print(f"  step {step}: FIRST_GRIP_VIABLE")

        if step % PRINT_EVERY == 0:
            print(f"  step {step:5d} | reward {reward:7.3f} | lift {info['lift_amount']*100:6.2f}cm "
                  f"| max_z {info['max_z']:.3f} | thumb_rank {info['thumb_rank']}")

        if done:
            print(f"  DONE at step {step} (info: {info})")
            break

    print(f"  Final: lift={info['lift_amount']*100:.2f}cm, max_z={info['max_z']:.3f}, "
          f"events_seen={sorted(events_seen)}")
    return events_seen, info

if __name__ == "__main__":
    env = DumbbellLiftEnv()
    obs_dim = env.observation_space[0]
    action_dim = env.action_space[0]
    print(f"env obs_dim={obs_dim}, action_dim={action_dim}")

    model, norm = load_policy_and_norm(obs_dim, action_dim)

    # Sanity check the normalization stats match this env's obs shape
    assert norm['obs_mean'].shape[0] == obs_dim, (
        f"obs_mean dim {norm['obs_mean'].shape[0]} != env obs_dim {obs_dim} "
        f"-- did you update _get_obs() to drop actuator_forces too?"
    )

    all_events = []
    for i in range(NUM_ROLLOUTS):
        events, info = run_rollout(env, model, norm, i + 1)
        all_events.append(events)

    print(f"\n{'='*70}\nSUMMARY across {NUM_ROLLOUTS} rollouts\n{'='*70}")
    for i, ev in enumerate(all_events):
        print(f"  rollout {i+1}: {sorted(ev)}")