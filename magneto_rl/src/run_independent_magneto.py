#!/usr/bin/env python3
# %%
import sys
# from magneto_env import MagnetoEnv
from independent_magneto_env import SimpleMagnetoEnv
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
    rel_path = 'independent_walking/'
    
    # . Training
    # & I can't actually load after declaring the model, have to instead load it alone
    model = PPO("MlpPolicy", env=env, verbose=1)
    # model = PPO.load(path + rel_path + 'breakpoint.zip', env=env)
    
    # # - Training
    try:
        
        for ii in range(100):
            model.learn(total_timesteps=10000, progress_bar=True)
            
            if ii % 10 == 0:
                model.save(path + rel_path + 'breakpoint_' + str(ii) + '.zip')
        
        # for i in range(10):
        #     obs, _ = env.reset()
        #     over = False
        #     counter = 0
        #     while not over:
        #         action, _states = model.predict(obs)
        #         obs, rewards, over, _, _ = env.step(action)
        #         env.render()
        #         counter += 1
        #         print(counter)
        # env.close()
        
    finally:
        model.save(path + rel_path + 'breakpoint.zip')
    
    # # . Evaluation
    # model = PPO.load(path + rel_path + 'breakpoint.zip')
    
    # for i in range(1):
    #     obs, _ = env.reset()
    #     over = False
    #     counter = 0
        
    #     while not over:
    #         action, _states = model.predict(obs)
    #         obs, rewards, over, _, _ = env.step(action)
    #         env.render()
    #         counter += 1

if __name__ == "__main__":
    main()

# %%
