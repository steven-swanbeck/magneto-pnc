#!/usr/bin/env python3
# %%
from magneto_env import MagnetoEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from magneto_policy_learner import CustomActorCriticPolicy

def eval (env, path, rel_path, iterations):
    # . Evaluation
    # model = DQN.load(path + rel_path + 'breakpoint.zip')
    # model = DQN.load(path + rel_path + '1mil_0.01_1.0.zip')
    # model = DQN.load(path + rel_path + 'better_0.03_1.0.zip')
    # model = DQN.load(path + rel_path + 'better_0.03_0.5.zip')
    # model = DQN.load(path + rel_path + 'weights/magneto_200000_steps.zip')
    # model = DQN.load(path + rel_path + 'weights/magneto_800000_steps.zip')
    model = DQN.load(path + rel_path + 'weights/magneto_1000000_steps.zip')

    # model = DQN.load(path + rel_path + 'weights/magneto_675000_steps.zip')
    
    for _ in range(iterations):
        obs, _ = env.reset()
        # print(obs)
        # input('...')
        over = False
        counter = 0
        while not over:
            action, _states = model.predict(obs)
            obs, rewards, over, _, _ = env.step(action)
            env.render()
            # input("Next step...")
            counter += 1
    env.close()

def main ():
    path = '/home/steven/magneto_ws/outputs/'
    # env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=10, anneal=True)
    env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=15, anneal=False)
    # rel_path = 'dqn/leader_follower/multi_input/paraboloid_penalty/'
    # rel_path = 'dqn/independent/multi_input/paraboloid_penalty/'
    # rel_path = 'dqn/independent/multi_input/no_magnetism/'
    rel_path = 'dqn/independent/multi_input/simulated_annealing/'
    # rel_path = 'dqn/independent/multi_input/cone/'
    
    # . Evaluation
    eval(env, path, rel_path, 5)

if __name__ == "__main__":
    main()

# %%
