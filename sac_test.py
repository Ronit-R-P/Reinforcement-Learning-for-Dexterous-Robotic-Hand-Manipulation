import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer

from env import DumbbellLiftEnv

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "sac_dumbbell_lift_bcinit_final.zip"
VECNORM_PATH = "vecnormalize_stats_bcinit.pkl"
XML_PATH = "debug_hand.xml"


# ============================================================
# SAME GYMNASIUM WRAPPER USED DURING TRAINING
# ============================================================

class GymDumbbellLiftEnv(gym.Env):
    """
    Adapter:
    DumbbellLiftEnv returns:
        obs, reward, done, info

    Gymnasium requires:
        obs, reward, terminated, truncated, info
    """

    def __init__(self, xml_path=XML_PATH):
        super().__init__()

        self._env = DumbbellLiftEnv(xml_path=xml_path)

        obs_dim = self._env.observation_space[0]
        act_dim = self._env.action_space[0]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = self._env.reset()

        return obs.astype(np.float32), {}

    def step(self, action):

        obs, reward, done, info = self._env.step(action)

        hit_step_limit = (
            self._env.step_count >= self._env.max_steps
        )

        terminated = bool(done and not hit_step_limit)
        truncated = bool(hit_step_limit)

        return (
            obs.astype(np.float32),
            float(reward),
            terminated,
            truncated,
            info
        )

    def render(self):
        pass

    def close(self):
        pass


# ============================================================
# ENVIRONMENT CREATION
# ============================================================

def make_env():
    return GymDumbbellLiftEnv(xml_path=XML_PATH)


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Loading trained SAC dumbbell lifting policy...")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Recreate the original vectorized environment
    # --------------------------------------------------------

    base_env = make_vec_env(
        make_env,
        n_envs=1
    )

    # --------------------------------------------------------
    # 2. Load EXACT observation normalization statistics
    # --------------------------------------------------------

    vec_env = VecNormalize.load(
        VECNORM_PATH,
        base_env
    )

    # VERY IMPORTANT:
    # We are TESTING, not training.
    vec_env.training = False

    # Do not normalize rewards during testing
    vec_env.norm_reward = False


    # --------------------------------------------------------
    # 3. Load trained SAC model
    # --------------------------------------------------------

    model = SAC.load(
        MODEL_PATH,
        env=vec_env,
        device="auto"
    )

    print("Model loaded successfully.")
    print("Using deterministic policy actions.")
    print()

    # --------------------------------------------------------
    # 4. Get the ACTUAL underlying MuJoCo environment
    # --------------------------------------------------------

    gym_env = vec_env.venv.envs[0].env
    raw_env = gym_env._env

    print("Opening MuJoCo viewer...")
    print("Close the viewer window to stop testing.")
    print()

    episode_number = 0

    # ========================================================
    # MUJOCO VIEWER
    # ========================================================

    with mujoco.viewer.launch_passive(
        raw_env.model,
        raw_env.data
    ) as viewer:

        while viewer.is_running():

            # ------------------------------------------------
            # Start a new episode
            # ------------------------------------------------

            episode_number += 1

            print("=" * 60)
            print(f"EPISODE {episode_number}")
            print("=" * 60)

            obs = vec_env.reset()

            episode_reward = 0.0
            done = False

            max_lift = 0.0
            max_height = raw_env.reward_fn.initial_z

            # ------------------------------------------------
            # Run one complete episode
            # ------------------------------------------------

            while not done and viewer.is_running():

                # Deterministic = use learned policy,
                # not random SAC exploration
                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                # Step the trained environment
                obs, rewards, dones, infos = vec_env.step(
                    action
                )

                # Extract values
                reward = float(rewards[0])
                info = infos[0]

                episode_reward += reward

                lift_amount = float(
                    info.get("lift_amount", 0.0)
                )

                max_z = float(
                    info.get("max_z", raw_env.reward_fn.initial_z)
                )

                if lift_amount > max_lift:
                    max_lift = lift_amount

                if max_z > max_height:
                    max_height = max_z

                done = bool(dones[0])

                # ------------------------------------------------
                # Print status every 100 steps
                # ------------------------------------------------

                if raw_env.step_count % 100 == 0:

                    print(
                        f"Step: {raw_env.step_count:4d} | "
                        f"Reward: {episode_reward:8.2f} | "
                        f"Lift: {lift_amount * 100:6.2f} cm | "
                        f"Max lift: {max_lift * 100:6.2f} cm | "
                        f"Curled: {info.get('all_curled', False)} | "
                        f"Grip: {info.get('grip_viable', False)}"
                    )

                # Update the MuJoCo viewer
                viewer.sync()


            # ====================================================
            # EPISODE FINISHED
            # ====================================================

            print()
            print("-" * 60)
            print(f"EPISODE {episode_number} FINISHED")
            print("-" * 60)

            print(
                f"Total reward: {episode_reward:.2f}"
            )

            print(
                f"Episode length: {raw_env.step_count}"
            )

            print(
                f"Maximum lift: {max_lift * 100:.2f} cm"
            )

            print(
                f"Maximum Z reached: {max_height:.4f}"
            )

            print("-" * 60)
            print()

            if viewer.is_running():

                print(
                    "Starting a new episode in 1 second..."
                )

                # Hold briefly so you can see the final state
                import time
                time.sleep(1.0)