#!/usr/bin/env python3
# %%
from independent_magneto_env import SimpleMagnetoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def main ():
    # . Trying to learn SOMETHING
    path = '/home/steven/magneto_ws/outputs/'
    
    env = SimpleMagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=0)
    rel_path = 'independent_walking/'
    
    # # . Training    
    
    # checkpoint_callback = CheckpointCallback(
    #     # save_freq=10,
    #     save_freq=100000,
    #     save_path=path + rel_path + 'weights/',
    #     name_prefix='magneto',
    #     save_replay_buffer=True,
    #     save_vecnormalize=True,
    # )

    # # - Start from scratch or load specified weights
    # # model = PPO("MlpPolicy", env=env, verbose=1)
    # # model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # model = PPO.load(path + rel_path + 'breakpoint.zip', env=env)
    
    # # # - Training
    # try:
    #     model.learn(total_timesteps=100000000, callback=checkpoint_callback, progress_bar=True)
    #     # model.learn(total_timesteps=10000000, progress_bar=True)
        
    #     for i in range(10):
    #         obs, _ = env.reset()
    #         over = False
    #         counter = 0
    #         while not over:
    #             action, _states = model.predict(obs)
    #             obs, rewards, over, _, _ = env.step(action)
    #             env.render()
    #             counter += 1
    #             # print(counter)
    #     env.close()
        
    # finally:
    #     model.save(path + rel_path + 'breakpoint.zip')
    
    # . Evaluation
    model = PPO.load(path + rel_path + 'breakpoint.zip')
    
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

if __name__ == "__main__":
    main()

# %%
