import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import os
from env import DumbbellLiftEnv

# ==========================================
# CONFIG
# ==========================================
ENV_XML = "debug_hand.xml"
BC_MODEL = "bc_policy.pt"
NORMALIZE_PATH = "bc_normalize.npy"
SAVE_PATH = "ppo_policy.pt"

MAX_STEPS = 5000
TOTAL_TIMESTEPS = 500_000
STEPS_PER_ROLLOUT = 2000
BATCH_SIZE = 128
EPOCHS_PER_ROLLOUT = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LEARNING_RATE = 3e-4
HIDDEN_LAYERS = [256, 256]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# LOAD NORMALIZATION
# ==========================================
norm = np.load(NORMALIZE_PATH, allow_pickle=True).item()
obs_mean = torch.FloatTensor(norm['obs_mean']).to(DEVICE)
obs_std = torch.FloatTensor(norm['obs_std']).to(DEVICE)
act_mean = torch.FloatTensor(norm['act_mean']).to(DEVICE)
act_std = torch.FloatTensor(norm['act_std']).to(DEVICE)

# ==========================================
# ACTOR-CRITIC NETWORK
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_layers):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(in_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(in_dim, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        mean = self.actor_mean(shared)
        std = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(shared)
        return mean, std, value
    
    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        if deterministic:
            return mean, value
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value, mean
    
    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

# ==========================================
# ENVIRONMENT
# ==========================================
env = DumbbellLiftEnv(xml_path=ENV_XML, max_steps=MAX_STEPS, warmup_steps=500)

# ==========================================
# INITIALIZE MODEL FROM BC
# ==========================================
model = ActorCritic(env.obs_dim, env.action_dim, HIDDEN_LAYERS).to(DEVICE)

# Load BC weights into shared layers
bc_state = torch.load(BC_MODEL, map_location=DEVICE)
bc_keys = list(bc_state.keys())
model_keys = list(model.state_dict().keys())

# Map BC linear layers to shared layers
for i in range(len(HIDDEN_LAYERS)):
    model.state_dict()[f'shared.{i*2}.weight'].copy_(bc_state[f'net.{i*2}.weight'])
    model.state_dict()[f'shared.{i*2}.bias'].copy_(bc_state[f'net.{i*2}.bias'])

# Initialize actor_mean from BC output layer
model.state_dict()['actor_mean.weight'].copy_(bc_state['net.4.weight'])
model.state_dict()['actor_mean.bias'].copy_(bc_state['net.4.bias'])

print("Loaded BC weights into PPO model")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# PPO TRAINING LOOP
# ==========================================
total_steps = 0
episode = 0

while total_steps < TOTAL_TIMESTEPS:
    # Collect rollout
    obs_buffer = []
    action_buffer = []
    reward_buffer = []
    value_buffer = []
    log_prob_buffer = []
    done_buffer = []
    
    obs = env.reset()
    episode_steps = 0
    
    for _ in range(STEPS_PER_ROLLOUT):
        obs_tensor = torch.FloatTensor(obs).to(DEVICE)
        obs_norm = (obs_tensor - obs_mean) / obs_std
        
        with torch.no_grad():
            action_norm, log_prob, value, _ = model.get_action(obs_norm.unsqueeze(0))
        
        action_norm = action_norm.squeeze(0).cpu().numpy()
        log_prob = log_prob.item()
        value = value.item()
        
        # Denormalize action
        action = action_norm * norm['act_std'] + norm['act_mean']
        act_low = env.ctrl_range_low
        act_high = env.ctrl_range_high
        action = np.clip(action, act_low, act_high)
        action_scaled = 2.0 * (action - act_low) / (act_high - act_low) - 1.0
        
        next_obs, reward, done, info = env.step(action_scaled)
        
        obs_buffer.append(obs)
        action_buffer.append(action_norm)
        reward_buffer.append(reward)
        value_buffer.append(value)
        log_prob_buffer.append(log_prob)
        done_buffer.append(done)
        
        obs = next_obs
        episode_steps += 1
        total_steps += 1
        
        if done:
            obs = env.reset()
            episode += 1
            episode_steps = 0
        
        if total_steps >= TOTAL_TIMESTEPS:
            break
    
    # Compute advantages
    rewards = np.array(reward_buffer)
    values = np.array(value_buffer)
    dones = np.array(done_buffer, dtype=np.float32)
    
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(values) else 0.0
        next_done = dones[t+1] if t+1 < len(dones) else 1.0
        delta = rewards[t] + GAMMA * next_value * (1 - next_done) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - next_done) * gae
        advantages[t] = gae
    
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Convert to tensors
    obs_tensor = torch.FloatTensor(np.array(obs_buffer)).to(DEVICE)
    action_tensor = torch.FloatTensor(np.array(action_buffer)).to(DEVICE)
    return_tensor = torch.FloatTensor(returns).to(DEVICE)
    advantage_tensor = torch.FloatTensor(advantages).to(DEVICE)
    old_log_prob_tensor = torch.FloatTensor(np.array(log_prob_buffer)).to(DEVICE)
    
    obs_norm_tensor = (obs_tensor - obs_mean) / obs_std
    
    # PPO update
    for epoch in range(EPOCHS_PER_ROLLOUT):
        indices = np.random.permutation(len(obs_buffer))
        
        for start in range(0, len(obs_buffer), BATCH_SIZE):
            batch_idx = indices[start:start+BATCH_SIZE]
            
            batch_obs = obs_norm_tensor[batch_idx]
            batch_action = action_tensor[batch_idx]
            batch_return = return_tensor[batch_idx]
            batch_adv = advantage_tensor[batch_idx]
            batch_old_log_prob = old_log_prob_tensor[batch_idx]
            
            log_prob, entropy, value = model.evaluate(batch_obs, batch_action)
            
            ratio = (log_prob - batch_old_log_prob).exp()
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * batch_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (batch_return - value.squeeze()).pow(2).mean()
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    
    # Logging
    avg_reward = rewards.mean()
    print(f"Steps: {total_steps}/{TOTAL_TIMESTEPS} | Ep: {episode} | "
      f"AvgR: {avg_reward:.3f} | Lift: {info['max_z']*100:.0f}cm | "
      f"Grip: {info['grip_viable']} | Curled: {info['all_curled']}")
    
    # Save checkpoint
    if total_steps % 50000 < STEPS_PER_ROLLOUT:
        torch.save(model.state_dict(), f"ppo_checkpoint_{total_steps}.pt")

# Save final model
torch.save(model.state_dict(), SAVE_PATH)
print(f"\nTraining complete. Model saved to {SAVE_PATH}")