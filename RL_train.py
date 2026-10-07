"""
SAC training on DumbbellLiftEnv, warm-started from your BC policy.

Run this from your MAIN working folder (same place as env.py, reward.py,
bc_policy.pt, bc_normalize.npy) -- NOT the old RL_train/PPO folder, that one
is unrelated leftover from an earlier attempt against the old reward function.

Install if needed:
    pip install stable-baselines3 gymnasium torch
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from env import DumbbellLiftEnv  # your patched (41-dim) env wrapper

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

BC_MODEL_PATH = "bc_policy.pt"
XML_PATH = "debug_hand.xml"          # <-- confirm this matches your actual model file


class GymDumbbellLiftEnv(gym.Env):
    """Adapter: DumbbellLiftEnv (4-tuple) -> gymnasium.Env (5-tuple)."""

    def __init__(self, xml_path=XML_PATH):
        super().__init__()
        self._env = DumbbellLiftEnv(xml_path=xml_path)
        obs_dim = self._env.observation_space[0]
        act_dim = self._env.action_space[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        hit_step_limit = self._env.step_count >= self._env.max_steps
        terminated = bool(done and not hit_step_limit)
        truncated = bool(hit_step_limit)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


def make_env():
    return GymDumbbellLiftEnv(xml_path=XML_PATH)


class EpisodePrintCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                print(f"  [episode {self.episode_count}] reward={info['episode']['r']:.2f}  length={info['episode']['l']}")
        return True


def warm_start_actor_from_bc(model, bc_path):
    """Copies BC's hidden-layer + output-layer weights into SAC's actor.
    Only the actor mean is warm-started; log_std and both critics stay at
    their normal random init (SAC always needs the critics to learn from
    scratch regardless -- there's no BC equivalent for those).

    Caveat, stated plainly: BC's weights were calibrated against its own
    z-scored obs/action normalization, while SAC here uses VecNormalize's
    running stats. The scales are roughly comparable (both center around 0
    with similar spread) but not identical -- this is an approximation, not
    an exact transplant. SAC's gradient updates correct for the mismatch
    during training; this just saves it from starting at random.
    """
    bc_state = torch.load(bc_path, map_location=model.device)
    actor = model.policy.actor

    with torch.no_grad():
        actor.latent_pi[0].weight.copy_(bc_state['net.0.weight'])
        actor.latent_pi[0].bias.copy_(bc_state['net.0.bias'])
        actor.latent_pi[2].weight.copy_(bc_state['net.2.weight'])
        actor.latent_pi[2].bias.copy_(bc_state['net.2.bias'])
        actor.mu.weight.copy_(bc_state['net.4.weight'])
        actor.mu.bias.copy_(bc_state['net.4.bias'])

    print("BC weights loaded into SAC actor (latent_pi + mu). log_std and both critics remain randomly initialized.")


if __name__ == "__main__":
    TOTAL_TIMESTEPS = 200_000
    N_ENVS = 1
    CHECKPOINT_EVERY = 10_000

    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),  # ReLU to match BC's activation
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        device="auto",
        tensorboard_log="./sac_tensorboard/",
    )

    warm_start_actor_from_bc(model, BC_MODEL_PATH)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path="./sac_checkpoints/",
        name_prefix="sac_dumbbell_lift_bcinit",
    )
    episode_print_callback = EpisodePrintCallback()

    print(f"Starting SAC (BC-initialized) training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, episode_print_callback],
        progress_bar=True,
    )

    model.save("sac_dumbbell_lift_bcinit_final")
    vec_env.save("vecnormalize_stats_bcinit.pkl")
    print("Done. Saved sac_dumbbell_lift_bcinit_final.zip and vecnormalize_stats_bcinit.pkl")