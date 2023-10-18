#!/usr/bin/env python3
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

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from typing import Tuple, Callable

# > https://www.gymlibrary.dev/content/environment_creation/
# > https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

# . Policy Learning Network
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

# . Custom policy
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

# . Helper functions
def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z]) # in radians

# . Useful action and state classes
class MagnetoAction (object):
    def __init__(self, idx:int=None, pose:Pose=None, link_idx:str=None) -> None:
        self.idx = idx
        self.pose = pose
        self.link_idx = link_idx
    
    def __repr__(self) -> str:
        return 'MagnetoAction()'
    
    def __str__(self) -> str:
        return f'idx: {self.idx}\npose: {self.pose}\nlink_idx: {self.link_idx}'

class MagnetoFootState (object):
    def __init__(self, state, link_id) -> None:
        self.link_id = link_id
        self.pose = state.pose
        self.magnetic_force = state.magnetic_force
    
    def __repr__(self) -> str:
        return 'MagnetoFootState()'
    
    def __str__(self) -> str:
        return f'link_id: {self.link_id}\npose: {self.pose}\nmagnetic_force: {self.magnetic_force}'
    
class MagnetoState (object):
    def __init__(self, state) -> None:
        self.ground_pose = state.ground_pose
        self.body_pose = state.body_pose
        self.foot0 = MagnetoFootState(state.AR_state, 'AR')
        self.foot1 = MagnetoFootState(state.AL_state, 'AL')
        self.foot2 = MagnetoFootState(state.BL_state, 'BL')
        self.foot3 = MagnetoFootState(state.BR_state, 'BR')
    
    def __repr__(self) -> str:
        return 'MagnetoState()'
    
    def __str__(self) -> str:
        return f'ground_pose: {self.ground_pose}\nbody_pose: {self.body_pose}\nfoot0: {self.foot0}\nfoot1: {self.foot1}\nfoot2: {self.foot2}\nfoot3: {self.foot3}'

# . Magneto Gym Environment
class MagnetoEnv (gym.Env):
    metadata = {"render_modes":[], "render_fps":0}
    
    def __init__ (self, render_mode=None):
        # WIP
        # . Verify that the ROS elements we need to add can be provided here without changing functionality
        # . Probably should add info about wall size and pose here? Or maybe need to make a serice call to the sim to get this?
        
        # ? Can I still do other things I need to in this script like this?
        self.plugin = MagnetoRLPlugin()
        
        self.link_idx_lookup = {0:'AR', 1:'AL', 2:'BL', 3:'BR'}
        
        self.states = []
        self.actions = []
        self.begin_episode()
        
        # TODO add in self.action_space and self.observation_space variable, similar to examples below:
        # > https://www.gymlibrary.dev/api/spaces/
        # + self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # + self.observation_space = spaces.Box(low=0, hgih=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
    
    def _get_obs (self):
        # WIP
        # . Add service call to get state information about the robot from sim
        state = MagnetoState(self.plugin.report_state())
        self.latest_state = state
        
        raise NotImplementedError
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info (self):
        # TODO
        # . Get auxillary information about robot/sim that may be helpful?
        
        raise NotImplementedError
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        # WIP
        # . Allow robot to step in place four times and measure the magnetic forces at each foot location
        # . Fill in state information of robot, goal, etc.
        
        self.terminate_episode()
        self.begin_episode()
        
        # TODO make this function return something in this form:
        # + return observation, info
        
    def step (self, action:MagnetoAction, check_status:bool=False):
        # WIP
        # . Map action output of RL stuff into action for robot in sim to execute
        # & This function may also get the resulting state information due to the nature of the sim, which can be stored for reference by the other relevant functions
        
        self.states.append(MagnetoState(self.plugin.report_state()))
        self.actions.append(action)
        success = self.plugin.update_action(self.link_idx_lookup[action.idx], action.pose)
        
        if check_status:
            post_state = MagnetoState(self.plugin.report_state())
            # if (not self.action_within_tolerance(self.states[-1], self.actions[-1], post_state)):
            #     # print('Error! Action was not executed within set tolerances so episode is terminated and treated as a failure.')
            #     self.reset()
            #     return False
            # elif self.has_fallen():
            if self.has_fallen():
                # print('Error! Robot is estimated to have fallen so episode is terminated and treated as a failure.')
                self.reset()
                return False
            elif not self.making_sufficient_contact(post_state):
                input("Inadequate contact detected!")
                self.reset()
                return False
        
        return success
        # TODO make this function return something in this form
        # + return observation, reward, terminated, truncated, info
    
    def render (self):
        # TODO
        
        raise NotImplementedError
    
    def discretize_geometry (self):
        # TODO
        # . Break geometry we are standing on into a 2d or 3d grid of space to be seeded with variable magnetism
        
        raise NotImplementedError
    
    def seed_magnetism (self): 
        # TODO
        # . Randomly seed magnetism (or more likely weak magnetism in a subtractive sense) to the wall in strong shapes and grow out from all of these accordingly
        # . Add a visualization method to see how this is working
        
        raise NotImplementedError
    
    def begin_episode (self) -> bool:
        # WIP
        # . Using roslaunch api and pyautogui, begin and manage a new simulation episode
        # . Let the robot perform its initial settling before turning control back to code for other functionality
        self.states = []
        self.actions = []
        return self.plugin.begin_sim_episode()

    def terminate_episode (self) -> bool:
        # WIP
        # . Using pyautogui, kill a currently running simulation episode
        return self.plugin.end_sim_episode()
    
    def close (self):
        return self.terminate_episode()
    
    def report_history (self) -> None:
        assert len(self.states) == len(self.actions)
        for i in range(len(self.states)):
            print(f'{i}:\nstate: {self.states[i]}\naction: {self.actions[i]}')
    
    def action_within_tolerance (self, state_i:MagnetoState, action:MagnetoAction, state_f:MagnetoState, tol_abs=0.08, tol_rel=0.5) -> bool:
        # WIP
        # . Transform into body pose frame of state_i, then calculate the difference between the final and intial states and see if it is within some tolerance
        
        r = R.from_quat([state_i.body_pose.orientation.x, state_i.body_pose.orientation.y, state_i.body_pose.orientation.z, state_i.body_pose.orientation.w])
        translation = np.expand_dims(np.array([state_i.body_pose.position.x, state_i.body_pose.position.y, state_i.body_pose.position.z]), 1)
        T = np.linalg.inv(np.concatenate([np.concatenate([r.as_matrix(), translation], axis=1), np.expand_dims(np.array([0, 0, 0, 1]), 1).T], axis=0))
        
        if action.idx == 0:
            foot_state_i = state_i.foot0
            foot_state_f = state_f.foot0
        elif action.idx == 1:
            foot_state_i = state_i.foot1
            foot_state_f = state_f.foot1
        elif action.idx == 2:
            foot_state_i = state_i.foot2
            foot_state_f = state_f.foot2
        else:
            foot_state_i = state_i.foot3
            foot_state_f = state_f.foot3
        
        v_i = np.expand_dims(np.array([foot_state_i.pose.position.x, foot_state_i.pose.position.y, foot_state_i.pose.position.z, 1]), 1)
        v_f = np.expand_dims(np.array([foot_state_f.pose.position.x, foot_state_f.pose.position.y, foot_state_f.pose.position.z, 1]), 1)
        
        v_i_p = T @ v_i
        v_f_p = T @ v_f
        
        v_diff = v_f_p - v_i_p
        
        v_des = np.expand_dims(np.array([action.pose.position.x, action.pose.position.y, action.pose.position.z, 0]), 1)
        
        print(f'action: {v_des}')
        # # print(f'v_i_p: {v_i_p}')
        # # print(f'v_f_p: {v_f_p}')
        print(f'result: {v_diff}')
        print(f'diff: {v_des - v_diff}')
        
        diff_mag = np.linalg.norm(v_des - v_diff)
        print(f'diff_mag: {diff_mag}')
        print(f'rel_diff_mag: {diff_mag / np.linalg.norm(v_des)}')
        if diff_mag > tol_abs:
            print(f"ERROR! Absolute position deviation {diff_mag} exceeds set absolute tolerance of {tol_abs}.")
            return False
        elif diff_mag > tol_rel * np.linalg.norm(v_des):
            print(f"ERROR! Relative position deviation {diff_mag / np.linalg.norm(v_des)} exceeds set relative tolerance of {tol_rel}.")
            return False
        return True
    
    def extract_ground_frame_positions (self, state:MagnetoState):
        r = R.from_quat([state.ground_pose.orientation.x, state.ground_pose.orientation.y, state.ground_pose.orientation.z, state.ground_pose.orientation.w])
        translation = np.expand_dims(np.array([state.ground_pose.position.x, state.ground_pose.position.y, state.ground_pose.position.z]), 1)
        T = np.linalg.inv(np.concatenate([np.concatenate([r.as_matrix(), translation], axis=1), np.expand_dims(np.array([0, 0, 0, 1]), 1).T], axis=0))
        
        pb = np.expand_dims(np.array([state.body_pose.position.x, state.body_pose.position.y, state.body_pose.position.z, 1]), 1)
        p0 = np.expand_dims(np.array([state.foot0.pose.position.x, state.foot0.pose.position.y, state.foot0.pose.position.z, 1]), 1)
        p1 = np.expand_dims(np.array([state.foot1.pose.position.x, state.foot1.pose.position.y, state.foot1.pose.position.z, 1]), 1)
        p2 = np.expand_dims(np.array([state.foot2.pose.position.x, state.foot2.pose.position.y, state.foot2.pose.position.z, 1]), 1)
        p3 = np.expand_dims(np.array([state.foot3.pose.position.x, state.foot3.pose.position.y, state.foot3.pose.position.z, 1]), 1)

        pb_ = T @ pb
        p0_ = T @ p0
        p1_ = T @ p1
        p2_ = T @ p2
        p3_ = T @ p3
        
        # print(f'pb_: {pb_}')
        # print(f'p0_: {p0_}')
        # print(f'p1_: {p1_}')
        # print(f'p2_: {p2_}')
        # print(f'p3_: {p3_}')
        # input('Continue?')
        return {'body':pb_, 'feet': {0:p0_, 1:p1_, 2:p2_, 3:p3_}}

    def has_fallen (self, tol_pos=0.2, tol_ori=1.5):
        # WIP
        # . Check whether robot has fallen (or if foot is too far from its intended position?) after some action has resolved
        if len(self.states) < 2:
            return False
        
        v1 = np.array([self.states[-1].body_pose.position.x, self.states[-1].body_pose.position.y, self.states[-1].body_pose.position.z])
        v0 = np.array([self.states[-2].body_pose.position.x, self.states[-2].body_pose.position.y, self.states[-2].body_pose.position.z])
        
        r1 = euler_from_quaternion(self.states[-1].body_pose.orientation.w, self.states[-1].body_pose.orientation.x, self.states[-1].body_pose.orientation.y, self.states[-1].body_pose.orientation.z)
        r0 = euler_from_quaternion(self.states[-2].body_pose.orientation.w, self.states[-2].body_pose.orientation.x, self.states[-2].body_pose.orientation.y, self.states[-2].body_pose.orientation.z)
        
        if np.linalg.norm(v1 - v0) > tol_pos:
            print(f'Error! Base position deviance of {np.linalg.norm(v1 - v0)} from previous step is greater than set tolerance of {tol_pos}, assuming robot fell and terminating episode...')
            return True
        elif np.linalg.norm(r1 - r0) > tol_ori:
            print(f'Error! Base orientation deviance of {np.linalg.norm(r1 - r0)} from previous step is greater than set tolerance of {tol_ori}, assuming robot fell and terminating episode...')
            return True
        
        return False
    
    def making_sufficient_contact (self, state, tol=0.002):
        if len(self.states) < 1:
            return False
        
        positions = self.extract_ground_frame_positions(state)
        errored = False
        error_msg = 'Error! Robot is making insufficient contact at:'
        for key, value in positions['feet'].items():
            if value[2][:] > tol: # z-coordinate
                error_msg += f'\n   Foot {key}! Value is {value[2][0]} and allowable tolerance is set to {tol}.'
                errored = True
        if errored:
            print(error_msg)
        return not errored
    
# . Testing methods
def iterate (env:MagnetoEnv, num_steps=10, check_status=True):
    pose = Pose()
    pose.orientation.w = 1.
    action = MagnetoAction(pose=pose)
    
    for i in range(num_steps):
        print(f'it: {i}')
        action.idx = random.randint(0,3)
        action.pose.position.x = round(random.uniform(-0.2, 0.2), 2)
        action.pose.position.y = round(random.uniform(-0.2, 0.2), 2)
        # if (action.idx == 0) or (action.idx == 1):
        #     action.pose.position.x = round(random.uniform(0, 0.2), 2)
        #     action.pose.position.y = round(random.uniform(0, 0.2), 2)
        # else:
        #     action.pose.position.x = round(random.uniform(-0.2, 0), 2)
        #     action.pose.position.y = round(random.uniform(-0.2, 0), 2)
        env.step(action, check_status=check_status)
        # if i == 3:
        #     env.report_history()
        #     env.reset()
    env.terminate_episode()

# TODO misc.
# & Think about how to incorporate uncertainty about where the foot will actually land given an instruction since it often does not perfectly map to the goal
# & Think about the algorithm or network that will learn and become the policy (refer to example code, probably want some RNN (LSTM?) that accepts all state info with uncertainty, goal, and perceived magentism and outputs a foot to move and location to which to move it)
# & Think about the RL algorithm we should use to train and other dynamics (continuous vs. discrete action space, state space, etc.)

if __name__ == "__main__":
    env = MagnetoEnv()
    iterate(env, 5)
    # env.report_history()
    
    # TODO finish relocating this into a gym env package we can load so allow us to use the PPO stuff    
    # model = PPO(CustomActroCriticPolicy, 'MagnetoSim-v0', verbose=0)
    # env = gym.make('MagnetoSim-v0', render_mode=None)
    model = PPO(CustomActroCriticPolicy, env=env, verbose=0)
    
    # reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=10)
    # print("Pre-Training")
    # print(f"Reward Mean: {reward_mean:.3f} Reward Std.: {reward_std:.3f}")


# # %%
# from stable_baselines3.common.env_checker import check_env
# env = MagnetoEnv()
# check_env(env)

# # %%
