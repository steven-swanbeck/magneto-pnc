#!/usr/bin/env python3
import sys
from magneto_env import MagnetoEnv
from magneto_utils import iterate
from stable_baselines3.common.env_checker import check_env
from magneto_policy_learner import CustomActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

def main ():
    # . Trying to learn SOMETHING
    path = '/home/steven/magneto_ws/outputs/'
    
    # env = MagnetoEnv()
    # rel_path = 'full_walking/'
    
    env = MagnetoEnv()
    # env = SimpleMagnetoEnv(sim_mode="grid")
    rel_path = 'independent_walking/'
    
    # model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # $ tensorboard --logdir /home/steven/magneto_tensorboard/
    
    # - Callback to save weights during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10,
        save_path=path + rel_path + 'weights/',
        name_prefix='magneto',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # - Loading specified weights
    # model = PPO("MlpPolicy", env=env, verbose=1)
    model = PPO.load(path + rel_path + 'breakpoint.zip', env=env)
    # model = PPO.load(path + rel_path + 'good0.5.zip', env=env)
    
    # reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=10)
    
    # - Training
    try:
        model.learn(total_timesteps=10, callback=checkpoint_callback, progress_bar=True)
    finally:
        pass
    #     model.save(path + rel_path + 'breakpoint.zip')
    #     # stamp = env.export_video()
    #     # model.save(path + rel_path + stamp + '.zip')

if __name__ == "__main__":
    main()

'''
The graveyard

path = '/home/steven/magneto_ws/outputs/'

# env = MagnetoEnv()
# rel_path = 'full_walking/'

env = SimpleMagnetoEnv()
rel_path = 'single_walking/'

# model = PPO(CustomActorCriticPolicy, env=env, verbose=0)
# model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="home/steven/magneto_ws/tensorboard/")
model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
# $ tensorboard --logdir /home/steven/magneto_tensorboard/

checkpoint_callback = CheckpointCallback(
    save_freq=10,
    save_path=path + rel_path + 'weights/',
    name_prefix='magneto',
    save_replay_buffer=True,
    save_vecnormalize=True,
)

model.load(path + rel_path + 'breakpoint.zip')

try:
    model.learn(total_timesteps=10, callback=checkpoint_callback, progress_bar=True)
finally:
    model.save(path + rel_path + 'breakpoint.zip')
    stamp = env.export_video()
    model.save(path + rel_path + stamp + '.zip')
    sys.exit()
    

#  Checking to make sure env is properly set up
# check_env(env)
# env.close()
# print('Past environment check!')

#  Just iterating to test
# iterate(env, 5)

#  Saving and loading model state
# model.save('/home/steven/magneto_ws/src/magneto-pnc/magneto_rl/weights/pre.zip')
# model.load('/home/steven/magneto_ws/src/magneto-pnc/magneto_rl/weights/pre.zip')
# print("LOADED!")

#  Just trying it out for funsies
# reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=1)
# env.close()
# print("Pre-Training")
# print(f"Reward Mean: {reward_mean:.3f} Reward Std.: {reward_std:.3f}")
    
'''
