#!/usr/bin/env python3
# %%
import numpy as np
from magneto_env import MagnetoEnv
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# %%
def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        # env = gym.make(env_id, render_mode="human")
        env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=10)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# %%
def train (env, path, rel_path, timesteps):
    # . Training    
    checkpoint_callback = CheckpointCallback(
        # save_freq=10,
        save_freq=200000,
        save_path=path + rel_path + 'weights/',
        name_prefix='magneto',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # - Start from scratch or load specified weights
    # model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    model = DQN("MultiInputPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # model = PPO.load(path + rel_path + 'breakpoint.zip', env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    print(model.policy)
    
    # # - Training
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, progress_bar=True)
        
        for i in range(10):
            obs, _ = env.reset()
            over = False
            counter = 0
            while not over:
                action, _states = model.predict(obs)
                obs, rewards, over, _, _ = env.step(action)
                env.render()
                counter += 1
        env.close()
        
    finally:
        # pass
        model.save(path + rel_path + 'breakpoint.zip')

def main ():
    path = '/home/steven/magneto_ws/outputs/'
    # env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=10)
    env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=10)
    # rel_path = 'dqn/leader_follower/multi_input/paraboloid_penalty/'
    # rel_path = 'dqn/independent/multi_input/paraboloid_penalty/'
    rel_path = 'dqn/independent/multi_input/no_magnetism/'
    
    # . Training
    train (env, path, rel_path, 1000000)
    

# %%
if __name__ == "__main__":
    main()

# %%
