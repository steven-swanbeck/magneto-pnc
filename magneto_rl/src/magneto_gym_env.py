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

# > https://www.gymlibrary.dev/content/environment_creation/

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
        self.body_pose = state.body_pose
        self.foot0 = MagnetoFootState(state.AR_state, 'AR')
        self.foot1 = MagnetoFootState(state.AL_state, 'AL')
        self.foot2 = MagnetoFootState(state.BL_state, 'BL')
        self.foot3 = MagnetoFootState(state.BR_state, 'BR')
    
    def __repr__(self) -> str:
        return 'MagnetoState()'
    
    def __str__(self) -> str:
        return f'body_pose: {self.body_pose}\nfoot0: {self.foot0}\nfoot1: {self.foot1}\nfoot2: {self.foot2}\nfoot3: {self.foot3}'
    
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
        
        # srv = Trigger()
        # self.plugin.reset_episode_cb(srv)
        self.terminate_episode()
        self.begin_episode()
        
        # raise NotImplementedError
        
    def step (self, action:MagnetoAction, check_status:bool=False) -> bool:
        # WIP
        # . Map action output of RL stuff into action for robot in sim to execute
        # & This function may also get the resulting state information due to the nature of the sim, which can be stored for reference by the other relevant functions
        
        self.states.append(MagnetoState(self.plugin.report_state()))
        self.actions.append(action)
        success = self.plugin.update_action(self.link_idx_lookup[action.idx], action.pose)
        
        if check_status:
            post_state = MagnetoState(self.plugin.report_state())
            self.action_within_tolerance(self.states[-1], self.actions[-1], post_state)
        
        return success
        
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
    
    def report_history (self) -> None:
        assert len(env.states) == len(env.actions)
        for i in range(len(env.states)):
            print(f'{i}:\nstate: {env.states[i]}\naction: {env.actions[i]}')
    
    def action_within_tolerance (self, state_i:MagnetoState, action:MagnetoAction, state_f:MagnetoState):
        # TODO 
        # . Transform into body pose frame of state_i, then calculate the difference between the final and intial states and see if it is within some tolerance
        # r = R.from_quat([root.orientation.x, root.orientation.y, root.orientation.z, root.orientation.w])
        # translation = -1 * np.expand_dims(np.array([root.position.x, root.position.y, root.position.z]), 1)
        # T = np.concatenate([np.concatenate([r.as_matrix(), translation], axis=1), np.expand_dims(np.array([0, 0, 0, 1]), 1).T], axis=0)
        
        r = R.from_quat([state_i.body_pose.orientation.x, state_i.body_pose.orientation.y, state_i.body_pose.orientation.z, state_i.body_pose.orientation.w])
        # translation = -1 * np.expand_dims(np.array([state_i.body_pose.position.x, state_i.body_pose.position.y, state_i.body_pose.position.z]), 1)
        translation = np.expand_dims(np.array([state_i.body_pose.position.x, state_i.body_pose.position.y, state_i.body_pose.position.z]), 1)
        T = np.concatenate([np.concatenate([r.as_matrix(), translation], axis=1), np.expand_dims(np.array([0, 0, 0, 1]), 1).T], axis=0)
        T = np.linalg.inv(T)
        
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
        print(f'v_i_p: {v_i_p}')
        print(f'v_f_p: {v_f_p}')
        print(f'diff: {v_diff}')
        print(f'real diff: {v_des - v_diff}')
        
        # TODO add in tolerancing to determine whether the executed action is good enough or not
        
        
        # raise NotImplementedError
        
    def has_fallen (self):
        # TODO
        # . Check whether robot has fallen (or if foot is too far from its intended position?) after some action has resolved
        
        raise NotImplementedError
        
    def can_continue (self):
        # TODO
        # self.has_fallen()
        # self.action_within_tolerance()
        
        raise NotImplementedError
    
    # TODO misc.
    # & Think about how to incorporate uncertainty about where the foot will actually land given an instruction since it often does not perfectly map to the goal
    # & Think about the algorithm or network that will learn and become the policy (refer to example code, probably want some RNN (LSTM?) that accepts all state info with uncertainty, goal, and perceived magentism and outputs a foot to move and location to which to move it)
    # & Think about the RL algorithm we should use to train and other dynamics (continuous vs. discrete action space, state space, etc.)

if __name__ == "__main__":
    env = MagnetoEnv()
    
    pose = Pose()
    pose.orientation.w = 1.
    action = MagnetoAction(pose=pose)
    
    num_moves = 2
    for i in range(num_moves):
        print(f'it: {i}')
        action.idx = random.randint(0,3)
        action.pose.position.x = round(random.uniform(-0.2, 0.2), 2)
        action.pose.position.y = round(random.uniform(-0.2, 0.2), 2)
        env.step(action, check_status=True)
        # if i == 3:
        #     env.report_history()
        #     env.reset()
    env.terminate_episode()
