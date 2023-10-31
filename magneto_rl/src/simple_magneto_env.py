#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from magneto_plugin import MagnetoRLPlugin
from magneto_utils import *

# from PIL import ImageGrab
import pyscreenshot as ImageGrab
import moviepy.video.io.ImageSequenceClip
from datetime import datetime
import csv

class SimpleMagnetoEnv (Env):
    metadata = {"render_modes":[], "render_fps":0}
    
    # DONE
    def __init__ (self, render_mode=None):
        super(SimpleMagnetoEnv, self).__init__()
        
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
        # - Format is x step size, y step size
        act_low = np.array([-1, -1])
        act_high = np.array([1, 1])
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
        yaw_min = -np.pi
        y_min_global = -5. # +
        y_max_global = 5. # +
        goal_min_x = -5.
        goal_max_x = 5.
        yaw_max = np.pi
        goal_min_y = -5.
        goal_max_y = 5.
        
        # TODO I should also add image support to this to receive the magnetic gradient
        obs_low = np.array([
            x_min_global, y_min_global, 
        ])
        obs_high = np.array([
            x_max_global, y_max_global,
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        self.link_idx_lookup = {0:'AR', 1:'AL', 2:'BL', 3:'BR'}
        self.max_foot_step_size = 0.08 # ! remember this is here!
        
        self.state_history = []
        self.action_history = []
        self.is_episode_running = False
        self.screenshots = []
        # self.goal = np.array([-1, 0]) # ! REMEMBER I'M FIXING THIS
    
    # WIP
    def step (self, gym_action, check_status:bool=True):
        self.state_history.append(MagnetoState(self.plugin.report_state()))
        
        # . Converting action from Gym format to one used by ROS plugin and other class members
        action = self.gym_2_action(gym_action)
        self.action_history.append(action)
        
        # . Taking specified action
        success = self.plugin.update_action(self.link_idx_lookup[0], action.pose)
        # self.screenshot()
        success = self.plugin.update_action(self.link_idx_lookup[1], action.pose)
        # self.screenshot()
        success = self.plugin.update_action(self.link_idx_lookup[3], action.pose)
        # self.screenshot()
        success = self.plugin.update_action(self.link_idx_lookup[2], action.pose)
        # self.screenshot()
        
        # . Observation and info
        obs_raw = self._get_obs(format='ros')
        info = self._get_info()
        
        # . Termination determination
        reward, is_terminated = self.calculate_reward(obs_raw, action)
        
        # .Converting observation to format required by Gym
        obs = self.state_2_gym(obs_raw)
        
        print('-----------------')
        print(f'Step reward: {reward}')
        print(f'Distance from goal: {np.linalg.norm(self.goal - np.array([obs_raw.body_pose.position.x, obs_raw.body_pose.position.y]), 1)}')
        print(f'Body goal: {self.goal}')
        print(f'Body position: {np.array([obs_raw.body_pose.position.x, obs_raw.body_pose.position.y])}')
        print(f'Obs: {obs}')
        print('-----------------')
        self.timesteps += 1
        
        return obs, reward, is_terminated, False, info
    
    # WIP
    def calculate_reward (self, state, action):
        is_terminated:bool = False
        
        if self.has_fallen(state):
            is_terminated = True
            reward = -1000
            # print(f'Fall detected! Reward set to {reward}')
        elif self.at_goal(state):
            is_terminated = True
            reward = 1000
            # print(f'Reached goal! Reward set to {reward}')
        else:
            
            # prev = np.array([self.state_history[-1].body_pose.position.x, self.state_history[-1].body_pose.position.y])
            # curr = np.array([state.body_pose.position.x, state.body_pose.position.y])
            
            # prev_reward = self.reward_paraboloid.eval(prev)
            # curr_reward = self.reward_paraboloid.eval(curr)
            
            # reward = prev_reward - curr_reward
            
            # print(f'prev: {prev}, reward: {prev_reward}')
            # print(f'curr: {curr}, reward: {curr_reward}')
            # print(f'reward: {reward}')
            
            curr = np.array([state.body_pose.position.x, state.body_pose.position.y])
            reward = -1 * self.reward_paraboloid.eval(curr)
            print(f'curr: {curr}, reward: {reward}')

            # proximity_reward_multiplier = 10
            # if len(self.state_history) > 0:
            #     # - check if foot index from action got closer to the goal, generate proximity reward accordingly
            #     proximity_change = self.calculate_distance_change(state)
            #     reward += proximity_reward_multiplier * proximity_change
            #     if reward < 0:
            #         reward -= 1 # ? small penalty to make sure the robot tries to improve
            #     # print(f'Proximity change for foot {action.idx} calculated as {proximity_change}, reward updated to {reward}')
        
        return reward, is_terminated
    
    def screenshot (self):
        self.screenshots.append(np.array(ImageGrab.grab(bbox=(100, 200, 1800, 1050))))

    def export_video (self, fps=10):
        stamp = str(datetime.now())
        
        if len(self.screenshots) > 0:
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(self.screenshots, fps=fps)
            clip.write_videofile('/home/steven/magneto_ws/outputs/single_walking/' + stamp + '.mp4')
            
            fields = [stamp, str(self.timesteps), str(self.goal[0]), str(self.goal[1]), str(self.state_history[-1].body_pose.position.x), str(self.state_history[-1].body_pose.position.y)]
            with open(r'/home/steven/magneto_ws/outputs/single_walking/log.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        
        return stamp
    
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
    def begin_episode (self) -> bool:
        self.state_history = []
        self.action_history = []
        self.is_episode_running = True
        self.timesteps = 0
        self.goal = np.array([random.uniform(-0.5, 0.5),random.uniform(-0.5, 0.5)]) # ! REMEMBER I'M FIXING THIS
        self.reward_paraboloid = paraboloid(self.goal)
        print(f'Sim initialized with a goal postion of {self.goal}')
        return self.plugin.begin_sim_episode()

    # DONE
    def terminate_episode (self) -> bool:
        self.is_episode_running = False
        # self.export_video()
        return self.plugin.end_sim_episode()
    
    # DONE
    def close (self):
        self.is_episode_running = False
        return self.terminate_episode()

    # DONE
    def has_fallen (self, state, tol_pos=0.18, tol_ori=1.2):
        if self.making_insufficient_contact(state) == 4:
            return True
        return False
    
    # DONE
    def making_insufficient_contact (self, state, tol=0.002):
        positions = extract_ground_frame_positions(state)
        insufficient = 0
        error_msg = 'Robot is making insufficient contact at:'
        for key, value in positions['feet'].items():
            if value[2][:] > tol: # z-coordinate
                error_msg += f'\n   Foot {key}! Value is {value[2][0]} and allowable tolerance is set to {tol}.'
                insufficient += 1
        if insufficient > 0:
            # error_msg = 'ERROR! ' + error_msg
            print(error_msg)
        return insufficient

    # DONE
    def at_goal (self, obs, tol=0.05):
        if np.linalg.norm(np.array([obs.body_pose.position.x, obs.body_pose.position.y]) - self.goal) < tol:
            return True
        return False
    
    def foot_at_goal (self, obs, tol=0.01):
        if np.linalg.norm(np.array([obs.foot0.pose.position.x, obs.foot0.pose.position.y]) - self.goal) < tol:
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

    # DONE
    def gym_2_action (self, gym_action:np.array) -> MagnetoAction:
        action = MagnetoAction()        
        action.pose.position.x = self.max_foot_step_size * gym_action[0]
        action.pose.position.y = self.max_foot_step_size * gym_action[1]
        return action
    
    # DONE
    def state_2_gym (self, state:MagnetoState) -> np.array:
        _, _, body_yaw = euler_from_quaternion(state.body_pose.orientation.w, state.body_pose.orientation.x, state.body_pose.orientation.y, state.body_pose.orientation.z)
        
        relative_goal = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, self.goal)
        
        gym_obs = np.array([
            relative_goal[0], relative_goal[1],
        ], dtype=np.float32)
        return gym_obs

    def calculate_distance_change (self, state):
        foot_pos = np.array([state.body_pose.position.x, state.body_pose.position.y])
        prev_pos = np.array([self.state_history[-1].body_pose.position.x, self.state_history[-1].body_pose.position.y])
        prev_dist = np.linalg.norm(prev_pos - self.goal, 1)
        curr_dist = np.linalg.norm(foot_pos - self.goal, 1)
        
        calc = (prev_dist - curr_dist) / prev_dist
        print(f'Prev: {prev_dist}')
        print(f'Curr: {curr_dist}')
        print(f'Calc: {calc}')
        alt = 1
        if calc > 0:
            # print(f'Choosing min between {calc} and {alt}')
            # print(f'Returning {np.min(np.array([calc, alt]))}')
            return np.min(np.array([calc, alt]))
        else:
            # print(f'Choosing max between {calc} and {-1 * alt}')
            # print(f'Returning {np.max(np.array([calc, -1 * alt]))}')
            return np.max(np.array([calc, -1 * alt])) # ? punishing negative movement extra to prevent the robot from adopting any oscillatory behaviors
