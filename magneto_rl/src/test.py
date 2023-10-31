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
# import magneto_gym

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
import numpy as np
from magneto_utils import MagnetoState

def foot_at_goal (goal=np.array([1,1]), tol=0.01):
    feets = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
    ])
    
    for foot in feets:
        if np.linalg.norm(foot - goal) < tol:
            return True
    return False


feets = foot_at_goal()
print(feets)

# %%
import pyscreenshot as ImageGrab

im = ImageGrab.grab(bbox=(100, 200, 1800, 1050)).save('/home/steven/magneto_ws/images/test.png')

test = np.array(ImageGrab.grab(bbox=(100, 200, 1800, 1050)))
# im.show()

# %%
import cv2
import os

image_folder = '/home/steven/magneto_ws/images/'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width,height))
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

# %%
# !!! THIS IS THE ONE TO USE!!!
import os
import moviepy.video.io.ImageSequenceClip
image_folder='/home/steven/magneto_ws/images/'
fps=8

image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('my_video.mp4')

# %%
from datetime import datetime

print(str(datetime.now()))
print('/home/steven/magneto_ws/videos/' + str(datetime.now()) + '.mp4')

# %%
# import pyscreenshot as ImageGrab
import numpy as np
from PIL import ImageGrab, Image

test = []
im = np.array(ImageGrab.grab(bbox=(100, 200, 1800, 1050)))
test.append(im)
im = np.array(ImageGrab.grab(bbox=(100, 200, 1800, 1050)))
test.append(im)

print(test)

# %%
import numpy as np

r = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

# %%
import numpy as np

def g2b(body_location, body_yaw, goal_location):
    r = np.empty((3, 3))
    r[0,:] = [np.cos(body_yaw), -np.sin(body_yaw), 0]
    r[1,:] = [np.sin(body_yaw), np.cos(body_yaw), 0]
    r[2,:] = [0, 0, 1]
    t = np.expand_dims(np.array([body_location[0], body_location[1], 0]), 1)
    T = np.linalg.inv(np.concatenate([np.concatenate([r, t], axis=1), np.expand_dims(np.array([0, 0, 0, 1]), 1).T], axis=0))
    # T = np.concatenate([np.concatenate([r, t], axis=1), np.expand_dims(np.array([0, 0, 0, 1]), 1).T], axis=0)
    
    g = np.expand_dims(np.array([goal_location[0], goal_location[1], 0, 1]), 1)
    
    g_b = T @ g
    
    return np.array([g_b[0][0], g_b[1][0]])
    
body_pos = np.array([1, 1])
body_yaw = np.pi / 4
goal_pos = np.array([0, 0])

g = g2b(body_pos, body_yaw, goal_pos)

# %%
import numpy as np

p1 = np.array([0, 0, 0])
p2 = np.array([1, 1, 0])

x1 = np.linalg.norm(p2- p1)
x2 = np.linalg.norm(p2- p1, 1)
print(x1)
print(x2)

# %%
import numpy as np
goal = np.array([0, 0])
curr = np.array([0, 0.9])
prev = np.array([0, 1])

sgn = np.sign(np.linalg.norm(prev - goal) - np.linalg.norm(curr - goal))
delta = np.linalg.norm(prev - curr, 1)
dist = np.linalg.norm(curr - goal)

print(f'sgn: {sgn}\ndelta: {delta}\ndist: {dist}\ns * d: {sgn * delta}\ns * d / p: {sgn * delta / dist}')

# %%
import numpy as np
goal = np.array([0, 0])
curr = np.array([1, 0])
prev = np.array([0, 1])

opt = goal - prev
act = curr - prev

print(f'opt: {opt}\nact: {act}')

res = np.dot(opt, act)
print(res)

# %%
import numpy as np

class paraboloid (object):
    def __init__(self, origin:np.array) -> None:
        self.w = origin[0]
        self.h = origin[1]
        
    def eval(self, location:np.array) -> float:
        return (location[0] - self.w)**2 + (location[1] - self.h)**2

goal = np.array([0, 0])
curr = np.array([0, -10])
prev = np.array([0, 10])

P = paraboloid(goal)
c1 = P.eval(prev)
c2 = P.eval(curr)

reward = c1 - c2

print(f'c1: {c1}\nc2: {c2}\nreward: {reward}')

# %%

norm = np.linalg.norm(curr - prev, 1)

res = np.sign(np.linalg.norm(prev - goal, 1) - np.linalg.norm(curr - goal, 1)) * (np.linalg.norm(prev, 1) - np.linalg.norm(curr, 1))**2
res = 1 / np.linalg.norm(goal - curr)

print(res)
# %%

prev = 0
curr1 = 1.6
curr2 = 2.0
x = np.sign(prev - curr1) * (prev - curr1)**2
y = np.sign(prev - curr2) * (prev - curr2)**2
print(f'x: {x}\ny: {y}')
# %%
