#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gym import Env, spaces

from scipy.spatial.transform import Rotation as R

from magneto_plugin import MagnetoRLPlugin
from magneto_utils import *

class MagnetoEnv (gym.Env):
    metadata = {"render_modes":[], "render_fps":0}
    
    # WIP
    def __init__ (self, render_mode=None):
        self.plugin = MagnetoRLPlugin()
        
        self.link_idx_lookup = {0:'AR', 1:'AL', 2:'BL', 3:'BR'}
        
        self.states = []
        self.actions = []
        self.begin_episode()
        
        # TODO add in self.action_space and self.observation_space variable, similar to examples below:
        # > https://www.gymlibrary.dev/api/spaces/
        # + self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # + self.observation_space = spaces.Box(low=0, hgih=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
    
    # WIP
    def _get_obs (self):
        state = MagnetoState(self.plugin.report_state())
        self.latest_state = state
        
        raise NotImplementedError
        return {"agent": self._agent_location, "target": self._target_location}
    
    # TODO
    def _get_info (self):
        # . Get auxillary information about robot/sim that may be helpful?
        
        raise NotImplementedError
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    # ! Mandatory
    # WIP
    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        # WIP
        self.terminate_episode()
        self.begin_episode()
        
        # TODO make this function return something in this form:
        # + return observation, info
    
    # ! Mandatory
    # WIP
    def step (self, action:MagnetoAction, check_status:bool=False):
        self.states.append(MagnetoState(self.plugin.report_state()))
        self.actions.append(action)
        success = self.plugin.update_action(self.link_idx_lookup[action.idx], action.pose)
        
        if check_status:
            post_state = MagnetoState(self.plugin.report_state())
            if self.has_fallen():
                self.reset()
                return False
            elif not self.making_sufficient_contact(post_state):
                input("Inadequate contact detected!")
                self.reset()
                return False
        
        return success
        # TODO make this function return something in this form
        # + return observation, reward, terminated, truncated, info
    
    # TODO
    def render (self):
        
        raise NotImplementedError
    
    # DONE
    def begin_episode (self) -> bool:
        self.states = []
        self.actions = []
        return self.plugin.begin_sim_episode()

    # DONE
    def terminate_episode (self) -> bool:
        return self.plugin.end_sim_episode()
    
    # WIP
    def close (self):
        return self.terminate_episode()
    
    # DONE
    def report_history (self) -> None:
        assert len(self.states) == len(self.actions)
        for i in range(len(self.states)):
            print(f'{i}:\nstate: {self.states[i]}\naction: {self.actions[i]}')
    
    # DONE
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
    
    # DONE
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
        return {'body':pb_, 'feet': {0:p0_, 1:p1_, 2:p2_, 3:p3_}}

    # DONE
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
    
    # DONE
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

