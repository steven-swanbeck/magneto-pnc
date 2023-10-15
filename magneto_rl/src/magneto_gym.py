#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gym import Env, spaces
import time

# > https://www.gymlibrary.dev/content/environment_creation/

class MagnetoEnv (gym.env):
    metadata = {"render_modes":[], "render_fps":0}
    
    def __init__ (self, render_mode=None):
        # TODO 
        # . Verify that the ROS elements we need to add can be provided here without changing functionality
        # . Probably should add info about wall size and pose here? Or maybe need to make a serice call to the sim to get this?
        
        raise NotImplementedError
    
    def _get_obs (self):
        # TODO 
        # . Add service call to get state information about the robot from sim
        
        raise NotImplementedError
        return {"agent": self._agent_location, "target": self._target_location}
        
    def _get_info (self):
        # TODO
        # . Get auxillary information about robot/sim that may be helpful?
        
        raise NotImplementedError
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO
        # . Allow robot to step in place four times and measure the magnetic forces at each foot location
        # . Fill in state information of robot, goal, etc.
        
        raise NotImplementedError
        
    def step (self, action):
        # TODO
        # . Map action output of RL stuff into action for robot in sim to execute
        # & This function may also get the resulting state information due to the nature of the sim, which can be stored for reference by the other relevant functions
        
        raise NotImplementedError
    
    def discretize_geometry ():
        # TODO
        # . Break geometry we are standing on into a 2d or 3d grid of space to be seeded with variable magnetism
        
        raise NotImplementedError
    
    def seed_magnetism (): 
        # TODO
        # . Randomly seed magnetism (or more likely weak magnetism in a subtractive sense) to the wall in strong shapes and grow out from all of these accordingly
        # . Add a visualization method to see how this is working
        
        raise NotImplementedError
    
    def begin_episode ():
        # TODO
        # . Using roslaunch api and pyautogui, begin and manage a new simulation episode
        # . Let the robot perform its initial settling before turning control back to code for other functionality
        
        raise NotImplementedError
    
    def terminate_episode ():
        # TODO
        # . Using pyautogui, kill a currently running simulation episode
        
        raise NotImplementedError
    
    # TODO misc.
    # & Think about how to incorporate uncertainty about where the foot will actually land given an instruction since it often does not perfectly map to the goal
    # & Think about the algorithm or network that will learn and become the policy (refer to example code, probably want some RNN (LSTM?) that accepts all state info with uncertainty, goal, and perceived magentism and outputs a foot to move and location to which to move it)
    # & Think about the RL algorithm we should use to train and other dynamics (continuous vs. discrete action space, state space, etc.)
