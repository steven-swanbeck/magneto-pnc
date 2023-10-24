#!/usr/bin/env python3
import sys
from magneto_env import MagnetoEnv
from magneto_utils import iterate
from stable_baselines3.common.env_checker import check_env
from magneto_policy_learner import CustomActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

if __name__ == "__main__":
    env = MagnetoEnv()
    # TODO need to add an observation space for this check to work (probably also an action space)
    # & Perhaps these are only used to specify the input and output shapes of the NN?
    # # & Maybe I can use a costmap of knowns and unknowns so that the input to the model is the same size always?
    
    # . Checking to make sure env is properly set up
    # check_env(env)
    # env.close()
    # print('Past environment check!')
    
    # . Just iterating to test
    # iterate(env, 5)
    
    # . Trying to learn SOMETHING with stable baselines and the simple network provided by Sentis
    model = PPO(CustomActorCriticPolicy, env=env, verbose=0)
    
    # - Saving and loading model state
    # model.save('/home/steven/magneto_ws/src/magneto-pnc/magneto_rl/weights/pre.zip')
    # model.load('/home/steven/magneto_ws/src/magneto-pnc/magneto_rl/weights/pre.zip')
    # print("LOADED!")
    
    # - Just trying it out for funsies
    # reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=1)
    # env.close()
    # print("Pre-Training")
    # print(f"Reward Mean: {reward_mean:.3f} Reward Std.: {reward_std:.3f}")
    
    # - Callback to save weights during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10,
        save_path='/home/steven/magneto_ws/src/magneto-pnc/magneto_rl/weights/',
        name_prefix='magneto',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # - Loading specified weights
    model.load('/home/steven/magneto_ws/src/magneto-pnc/magneto_rl/weights/snapshot.zip')
    
    # - Training
    try:
        model.learn(total_timesteps=10, callback=checkpoint_callback, progress_bar=True)
    except KeyboardInterrupt:
        model.save('/home/steven/magneto_ws/src/magneto-pnc/magneto_rl/weights/breakpoint.zip')
        env.close()
        sys.exit()
