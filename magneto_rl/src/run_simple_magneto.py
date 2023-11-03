#!/usr/bin/env python3
# %%
import sys
# from magneto_env import MagnetoEnv
from simple_magneto_env import SimpleMagnetoEnv
from magneto_utils import iterate
from stable_baselines3.common.env_checker import check_env
from magneto_policy_learner import CustomActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
import time

def main ():
    # . Trying to learn SOMETHING
    path = '/home/steven/magneto_ws/outputs/'
    
    env = SimpleMagnetoEnv(render_mode="human", sim_mode="grid")
    rel_path = 'simple_walking/'
    
    # # . Training
    # # $ tensorboard --logdir /home/steven/magneto_tensorboard/
    
    # # # - Callback to save weights during training
    # # checkpoint_callback = CheckpointCallback(
    # #     # save_freq=10,
    # #     save_freq=10000,
    # #     save_path=path + rel_path + 'weights/',
    # #     name_prefix='magneto',
    # #     save_replay_buffer=True,
    # #     save_vecnormalize=True,
    # # )
    
    # # - Loading specified weights
    # # & I can't actually load after declaring the model, have to instead load it alone
    # # model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # # model = PPO("MlpPolicy", env=env, verbose=1)
    # model = PPO.load(path + rel_path + 'breakpoint.zip', env=env)
    
    # # # - Training
    # try:
    #     # model.learn(total_timesteps=100000, callback=checkpoint_callback, progress_bar=True)
    #     model.learn(total_timesteps=100000, progress_bar=True)
        
    #     for i in range(10):
    #         obs, _ = env.reset()
    #         over = False
    #         counter = 0
    #         while not over:
    #             action, _states = model.predict(obs)
    #             obs, rewards, over, _, _ = env.step(action)
    #             env.render()
    #             counter += 1
    #             print(counter)
    #     env.close()
        
    # finally:
    #     model.save(path + rel_path + 'breakpoint.zip')
    #     # stamp = env.export_video()
    #     # model.save(path + rel_path + stamp + '.zip')
    
    # . Evaluation
    model = PPO.load(path + rel_path + 'breakpoint.zip')
    
    # env.reset()
    # env.render()
    
    # time.sleep(3.0)
    # env.close()
    
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

    # # reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=10)

if __name__ == "__main__":
    main()

# %%
