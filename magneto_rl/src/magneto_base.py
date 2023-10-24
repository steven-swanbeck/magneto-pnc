#!/usr/bin/env python3
from magneto_env import MagnetoEnv
from magneto_utils import iterate
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    env = MagnetoEnv()
    # TODO need to add an observation space for this check to work (probably also an action space)
    # & Perhaps these are only used to specify the input and output shapes of the NN?
    # & Maybe I can use a costmap of knowns and unknowns so that the input to the model is the same size always?
    check_env(env)
    
    env.close()
    print('Past environment check!')
    
    # . Next steps
    # - 1. Figure out how to add state and action spaces to Gym environment
    # - 2. Test to make sure we can use the environment then with stable baselines
    # - 3. Then try to package it nicely into an installable library that can be launched via ROS
    # - 4. Try to learn something with the network supplied by Sentis and after figuring out the state and actions spaces
    
    # iterate(env, 5)
