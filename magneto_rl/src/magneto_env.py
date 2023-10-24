#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces

from scipy.spatial.transform import Rotation as R

from magneto_plugin import MagnetoRLPlugin
from magneto_utils import *

class MagnetoEnv (Env):
    metadata = {"render_modes":[], "render_fps":0}
    
    # WIP
    def __init__ (self, render_mode=None):
        super(MagnetoEnv, self).__init__()
        
        self.plugin = MagnetoRLPlugin()
        
        # TODO add in self.action_space and self.observation_space variable, similar to examples below:
        # > https://www.gymlibrary.dev/api/spaces/
        # + self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # + self.observation_space = spaces.Box(low=0, hgih=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        '''
        Action space should be something like this (I suppose just a box of continuous variables):
        1. One value for which leg we are wanting to move (round to nearest whole number between 0 and 4 in the step method)
        2. Continuous xy values (likely bounded by some appropriate thresholds, -0.25 and 0.25, perhaps)
        '''
        # - Format is foot id, x step size, y step size
        # act_low = np.array([-0.5, -0.2, -0.2])
        # act_high = np.array([3.49, 0.2, 0.2])
        act_low = np.array([-1, -1, -1])
        act_high = np.array([1, 1, 1])
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        
        '''
        Observation space should be something like this:
        1. Robot pose (xyz, qxyz) bounded within workspace? Limits xyz to be constrained on plane and qxyz to form unit quaternion
        2. Magnetic force at each foot bounded between 0 and the maximum force specified in the target parameter
        Is that it???
        '''
        # . x, y, yaw, foot0_mag, foot0_x, foot0_y, foot1_mag, foot1_x, foot1_y, foot2_mag, foot2_x, foot2_y, foot3_mag, foot3_x, foot3_y, goal_x, goal_y
        # - x and y should be set based on the limits of the wall it is on, yaw can be full 360, magnetic forces are bounded between 0 and the known max force
        
        x_min_global = -5. # +
        x_max_global = 5. # +
        y_min_global = -5. # +
        y_max_global = 5. # +
        yaw_min = -np.pi
        yaw_max = np.pi
        mag_min = 0. # +
        mag_max = 147. # + this should come from loaded param (add function to plugin that returns dictionary of all these needed values)
        goal_min_x = -5.
        goal_max_x = 5.
        goal_min_y = -5.
        goal_max_y = 5.
        
        # TODO I should also add image support to this to receive the magnetic gradient
        obs_low = np.array([
            x_min_global, y_min_global, yaw_min,
            mag_min, x_min_global, y_min_global,
            mag_min, x_min_global, y_min_global,
            mag_min, x_min_global, y_min_global,
            mag_min, x_min_global, y_min_global,
            goal_min_x, goal_min_y, 
        ])
        obs_high = np.array([
            x_max_global, y_max_global, yaw_max,
            mag_max, x_max_global, y_max_global,
            mag_max, x_max_global, y_max_global,
            mag_max, x_max_global, y_max_global,
            mag_max, x_max_global, y_max_global,
            goal_max_x, goal_max_y,
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        self.link_idx_lookup = {0:'AR', 1:'AL', 2:'BL', 3:'BR'}
        self.max_foot_step_size = 0.2 # ! remember this is here!
        
        self.state_history = []
        self.action_history = []
        self.is_episode_running = False
        
        self.goal = np.array([1,1])

    # DONE
    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        # WIP
        if self.is_episode_running:
            self.terminate_episode()
        self.begin_episode()
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    # DONE
    def step (self, gym_action, check_status:bool=True):
        self.state_history.append(MagnetoState(self.plugin.report_state()))
        
        # . Converting action from Gym format to one used by ROS plugin and other class members
        action = self.gym_2_action(gym_action)
        self.action_history.append(action)
        
        # . Taking specified action
        success = self.plugin.update_action(self.link_idx_lookup[action.idx], action.pose)
        
        if not check_status:
            return success
        
        # . Observation and info
        obs_raw = self._get_obs(format='ros')
        info = self._get_info()
        
        # . Termination determination
        is_terminated:bool = False
        goal_reached:bool = False
        if self.has_fallen(): # if the robot has fallen
            is_terminated = True
            # self.reset()
        elif not self.making_sufficient_contact(obs_raw): # if the robot is not making good contact with the wall
            is_terminated = True
            # self.reset()
        elif self.at_goal(obs_raw): # if neither above true, if the robot has reached its goal position
            goal_reached = True
            is_terminated = True
        
        # . Reward calculation
        if goal_reached:
            reward = 5
        elif is_terminated:
            reward = -10
        else:
            reward = -1
        
        # .Converting observation to format required by Gym
        obs = self.state_2_gym(obs_raw)
        
        return obs, reward, is_terminated, False, info
    
    # DONE
    def _get_obs (self, format='gym'):
        state = MagnetoState(self.plugin.report_state())
        if format == 'gym':
            return self.state_2_gym(state)
        return state
    
    # DONE
    def _get_info (self):
        # . Get auxillary information about robot/sim that may be helpful?
        return {}
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    # TODO
    def render (self):
        raise NotImplementedError
    
    # DONE
    def gym_2_action (self, gym_action:np.array) -> MagnetoAction:
        action = MagnetoAction()
        # action.idx = return_closest(gym_action[0], [0, 1, 2, 3]) # ? deprecated to make symmetric action space for foot id
        action.idx = self.get_foot_from_action(gym_action[0])
        
        action.pose.position.x = self.max_foot_step_size * gym_action[1]
        action.pose.position.y = self.max_foot_step_size * gym_action[2]
        return action
    
    # DONE
    def state_2_gym (self, state:MagnetoState) -> np.array:
        _, _, body_yaw = euler_from_quaternion(state.body_pose.orientation.w, state.body_pose.orientation.x, state.body_pose.orientation.y, state.body_pose.orientation.z)
        
        gym_obs = np.array([state.body_pose.position.x, state.body_pose.position.x, body_yaw,
                            state.foot0.magnetic_force, state.foot0.pose.position.x, state.foot0.pose.position.y,
                            state.foot1.magnetic_force, state.foot1.pose.position.x, state.foot1.pose.position.y,
                            state.foot2.magnetic_force, state.foot2.pose.position.x, state.foot2.pose.position.y,
                            state.foot3.magnetic_force, state.foot3.pose.position.x, state.foot3.pose.position.y,
                            self.goal[0], self.goal[1],
        ], dtype=np.float32)
        return gym_obs
    
    # DONE
    def begin_episode (self) -> bool:
        self.state_history = []
        self.action_history = []
        self.is_episode_running = True
        return self.plugin.begin_sim_episode()

    # DONE
    def terminate_episode (self) -> bool:
        self.is_episode_running = False
        return self.plugin.end_sim_episode()
    
    # DONE
    def close (self):
        self.is_episode_running = False
        return self.terminate_episode()
    
    # DONE
    def report_history (self) -> None:
        assert len(self.state_history) == len(self.action_history)
        for i in range(len(self.state_history)):
            print(f'{i}:\nstate: {self.state_history[i]}\naction: {self.action_history[i]}')
    
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
    def has_fallen (self, tol_pos=0.18, tol_ori=1.2):
        # WIP
        # . Check whether robot has fallen (or if foot is too far from its intended position?) after some action has resolved
        if len(self.state_history) < 2:
            return False
        
        v1 = np.array([self.state_history[-1].body_pose.position.x, self.state_history[-1].body_pose.position.y, self.state_history[-1].body_pose.position.z])
        v0 = np.array([self.state_history[-2].body_pose.position.x, self.state_history[-2].body_pose.position.y, self.state_history[-2].body_pose.position.z])
        
        r1 = euler_from_quaternion(self.state_history[-1].body_pose.orientation.w, self.state_history[-1].body_pose.orientation.x, self.state_history[-1].body_pose.orientation.y, self.state_history[-1].body_pose.orientation.z)
        r0 = euler_from_quaternion(self.state_history[-2].body_pose.orientation.w, self.state_history[-2].body_pose.orientation.x, self.state_history[-2].body_pose.orientation.y, self.state_history[-2].body_pose.orientation.z)
        
        if np.linalg.norm(v1 - v0) > tol_pos:
            # print(f'Error! Base position deviance of {np.linalg.norm(v1 - v0)} from previous step is greater than set tolerance of {tol_pos}, assuming robot fell and terminating episode...')
            return True
        elif np.linalg.norm(r1 - r0) > tol_ori:
            # print(f'Error! Base orientation deviance of {np.linalg.norm(r1 - r0)} from previous step is greater than set tolerance of {tol_ori}, assuming robot fell and terminating episode...')
            return True
        
        return False
    
    # DONE
    def making_sufficient_contact (self, state, tol=0.002):
        if len(self.state_history) < 1:
            return False
        
        positions = self.extract_ground_frame_positions(state)
        insufficient = 0
        error_msg = 'Robot is making insufficient contact at:'
        for key, value in positions['feet'].items():
            if value[2][:] > tol: # z-coordinate
                error_msg += f'\n   Foot {key}! Value is {value[2][0]} and allowable tolerance is set to {tol}.'
                insufficient += 1
        if insufficient > 1:
            error_msg = 'ERROR! ' + error_msg + ' Resetting sim.'
            print(error_msg)
            outcome = False
        else:
            # print(error_msg)
            outcome = True
        return outcome

    # TODO
    def at_goal (self, obs, tol=0.01):
        # . Take in observation and determine if we are at the goal position
        if np.linalg.norm(np.array([obs.body_pose.position.x, obs.body_pose.position.y]) - self.goal) < tol:
            return True
        return False

    # DONE
    def get_foot_from_action (self, x):
        if x > 0.5:
            return 3
        if x > 0.0:
            return 2
        if x < -0.5:
            return 0
        return 1
