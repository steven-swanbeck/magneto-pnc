#!/usr/bin/env python3
import sys
from magneto_env import MagnetoEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def main ():
    # . Trying to learn SOMETHING
    path = '/home/steven/magneto_ws/outputs/'
    
    env = MagnetoEnv(render_mode="human", sim_mode="full", magnetic_seeds=10)
    rel_path = 'dqn/independent/multi_input/paraboloid_penalty/'
    
    # - Callback to save weights during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10,
        save_path=path + rel_path + 'weights/',
        name_prefix='magneto',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # - Loading specified weights
    
    # model = DQN.load(path + 'dqn/independent/multi_input/paraboloid_penalty/breakpoint.zip', env=env)
    # model = DQN.load(path + 'dqn/independent/multi_input/simulated_annealing/breakpoint.zip', env=env)
    # model = DQN.load(path + 'dqn/independent/multi_input/cone/1mil_0.05_0.5.zip', env=env)
    # model = DQN.load(path + 'dqn/independent/multi_input/cone/breakpoint.zip', env=env)
    model = DQN.load(path + 'dqn/independent/multi_input/simulated_annealing/great_circles/magneto_1000000_steps.zip', env=env)
    
    # model = PPO.load(path + 'leader-follower/no_magnetic_seeds/added_magnetics_obs/breakpoint.zip', env=env)
    # model = DQN.load(path + 'dqn/independent/multi_input/no_magnetism/breakpoint.zip', env=env)
    
    # - Training
    try:
        # model.learn(total_timesteps=10, callback=checkpoint_callback, progress_bar=True)
        
        for _ in range(1):
            obs, _ = env.reset()
            over = False
            counter = 0
            while not over:
                env.render()
                action, _states = model.predict(obs)
                obs, rewards, over, _, _ = env.step(action)
                # env.plugin._render_frame()
                # input("Next step...")
                counter += 1
            env.close()
        
    finally:
        pass
    #     model.save(path + rel_path + 'breakpoint.zip')
    #     # stamp = env.export_video()
    #     # model.save(path + rel_path + stamp + '.zip')

if __name__ == "__main__":
    main()
