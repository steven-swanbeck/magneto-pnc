#!/usr/bin/env python3
# * Goals
# . Launch node using roslaunch api
# . Click on screen in correct location to focus window then unpause simulation
# . Kill node and repeat

import rospy
import roslaunch
# from subprocess import Popen, PIPE
import pyautogui
import time

if __name__ == "__main__":
    
    rospy.init_node('roslaunch_test')
    
    node = roslaunch.core.Node('my_simulator', 
                            'magneto_ros',
                            args='config/Magneto/USERCONTROLWALK.yaml')
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()
    
    process = launch.launch(node)
    print(process.is_alive)
    
    # rospy.sleep(5)
    time.sleep(3)
    
    pyautogui.doubleClick(1440 + 500/2, 10)
    pyautogui.click(1440 + 500/2, 500/2)
    pyautogui.press('space')
    time.sleep(1)
    
    
    pyautogui.press('s')
    time.sleep(5)
    
    print("Forcing shutdown!")
    pyautogui.click(1899, 21)
    process.stop()
    time.sleep(1)
    pyautogui.click(1440 + 500/2, 500/2)
    with pyautogui.hold('ctrl'):
        pyautogui.press('c')
    
    while not rospy.is_shutdown():
        rospy.spin()

# %%
import pyautogui
screenWidth, screenHeight = pyautogui.size()
print(screenWidth, screenHeight)

currentMouseX, currentMouseY = pyautogui.position()
print(currentMouseX, currentMouseY)

# # %%
# # pyautogui.moveTo(1650, 150)
# pyautogui.click(1650, 150)
# pyautogui.press('space')

# # %%
# pyautogui.doubleClick(1650, 150)

# %%
import gymnasium as gym
import magneto_gym

env = gym.make('MagnetoWorld-v0')

# %%
import pyautogui
# pyautogui.moveTo(1440 + 500/2, 10)
pyautogui.moveTo(1899, 21 + 20)
# %%
print(pyautogui.position())

# %%
import numpy as np
import gymnasium as gym
from gym import Env, spaces
import time
from std_srvs.srv import Trigger
from magneto_plugin import MagnetoRLPlugin
from geometry_msgs.msg import Pose
import random
from scipy.spatial.transform import Rotation as R
import math

from magneto_env import MagnetoEnv

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from typing import Tuple, Callable

class ModelLearnerNetwork (torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network
        # Your policy_net must take in a vector of length feature_dim
        # and ouput a vector of length last_layer_dim_pi
        self.policy_net = torch.nn.Linear(feature_dim, last_layer_dim_pi)

        # Value network
        # Your value_net must take in a vector of length feature_dim
        # and ouput a vector of length last_layer_dim_vf
        self.value_net = torch.nn.Linear(feature_dim, last_layer_dim_vf)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
class CustomActroCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ModelLearnerNetwork(self.features_dim)

env = MagnetoEnv()

model = PPO(CustomActroCriticPolicy, env=env, verbose=0)
# model = PPO(CustomActroCriticPolicy, 'MagnetoSim-v0', verbose=0)
# env = gym.make('MagnetoSim-v0', render_mode=None)

reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=10)
print("Pre-Training")
print(f"Reward Mean: {reward_mean:.3f} Reward Std.: {reward_std:.3f}")

# %%
import numpy as np

np.int8(np.round(-0.5))

# %%
