#!/usr/bin/env python3
# %%
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
try:
    from magneto_ros_plugin import MagnetoRLPlugin
except ImportError:
    print("Unable to import ROS-based plugin!")
from magneto_utils import *
from magneto_game_plugin import GamePlugin

# from PIL import ImageGrab
import pyscreenshot as ImageGrab
import moviepy.video.io.ImageSequenceClip
from datetime import datetime
import csv

class MagnetoEnv (Env):
    metadata = {"render_modes":["human", "rgb_array"], "render_fps":10}
    
    def __init__ (self, render_mode=None, sim_mode="full", magnetic_seeds=10):
        super(MagnetoEnv, self).__init__()
        
        self.sim_mode = sim_mode
        if self.sim_mode == "full":
            self.plugin = MagnetoRLPlugin()
        else:
            self.plugin = GamePlugin(render_mode, self.metadata["render_fps"], magnetic_seeds)
            self.render_mode = render_mode
        
        act_low = np.array([-1, -1, -1])
        act_high = np.array([1, 1, 1])
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        
        x_min_global = -10. # +
        x_max_global = 10. # +
        yaw_min = -np.pi
        y_min_global = -10. # +
        y_max_global = 10. # +
        goal_min_x = -5.
        goal_max_x = 5.
        yaw_max = np.pi
        goal_min_y = -5.
        goal_max_y = 5.
        mag_min = 0.
        mag_max = 1.
        
        # TODO figure out how I want to add the magnetism (whole robot or legs? maybe projection in front of it?)
        obs_low = np.array([
            x_min_global, y_min_global,
            x_min_global, y_min_global,
            x_min_global, y_min_global,
            x_min_global, y_min_global,
            x_min_global, y_min_global,
        ])
        obs_high = np.array([
            x_max_global, y_max_global,
            x_max_global, y_max_global,
            x_max_global, y_max_global,
            x_max_global, y_max_global,
            x_max_global, y_max_global,
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        self.link_idx_lookup = {0:'AR', 1:'AL', 2:'BL', 3:'BR'}
        self.max_foot_step_size = 0.08 # ! remember this is here!
        
        self.state_history = []
        self.action_history = []
        self.is_episode_running = False
        self.screenshots = []
    
    def step (self, gym_action, check_status:bool=True):
        self.state_history.append(MagnetoState(self.plugin.report_state()))
        
        # . Converting action from Gym format to one used by ROS plugin and other class members
        action = self.gym_2_action(gym_action)
        self.action_history.append(action)
        
        # . Taking specified action
        success = self.plugin.update_action(self.link_idx_lookup[action.idx], action.pose)
        # walk_order = np.random.permutation([0, 1, 2, 3])
        # success = self.plugin.update_action(self.link_idx_lookup[walk_order[0]], action.pose)
        # # self.screenshot()
        # success = self.plugin.update_action(self.link_idx_lookup[walk_order[1]], action.pose)
        # # self.screenshot()
        # success = self.plugin.update_action(self.link_idx_lookup[walk_order[2]], action.pose)
        # # self.screenshot()
        # success = self.plugin.update_action(self.link_idx_lookup[walk_order[3]], action.pose)
        
        # . Observation and info
        obs_raw = self._get_obs(format='ros')
        info = self._get_info()
        
        # . Termination determination
        reward, is_terminated = self.calculate_reward(obs_raw, action)
        
        # .Converting observation to format required by Gym
        obs = self.state_2_gym(obs_raw)
        
        if self.sim_mode == "full":
            print('-----------------')
            print(f'Step reward: {reward}')
            print(f'Distance from goal: {np.linalg.norm(self.goal - np.array([obs_raw.body_pose.position.x, obs_raw.body_pose.position.y]), 1)}')
            print(f'Body goal: {self.goal}')
            print(f'Body position: {np.array([obs_raw.body_pose.position.x, obs_raw.body_pose.position.y])}')
            print(f'Obs: {obs}')
            print('-----------------')
        self.timesteps += 1
        
        if self.sim_mode == "full":
            obs = -1 * obs
        return obs, reward, is_terminated, False, info
    
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
            # curr = np.array([state.body_pose.position.x, state.body_pose.position.y])
            # TODO! provide reward to each foot for where it is instead of the base
            # ! Make sure these are correct mappings!
            if action.idx == 0:
                curr = np.array([state.foot0.pose.position.x, state.foot0.pose.position.y])
            elif action.idx == 1:
                curr = np.array([state.foot1.pose.position.x, state.foot1.pose.position.y])
            elif action.idx == 2:
                curr = np.array([state.foot2.pose.position.x, state.foot2.pose.position.y])
            elif action.idx == 3:
                curr = np.array([state.foot3.pose.position.x, state.foot3.pose.position.y])
                
            reward = -1 * self.reward_paraboloid.eval(curr)
            # print(f'curr: {curr}, reward: {reward}')
        
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
    
    def _get_obs (self, format='gym'):
        state = MagnetoState(self.plugin.report_state())
        if format == 'gym':
            return self.state_2_gym(state)
        return state
    
    def _get_info (self):
        return {}
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def render (self):
        if (self.render_mode == "rgb_array") or (self.render_mode == "human"):
            self.plugin._render_frame()
    
    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        # WIP
        if self.is_episode_running:
            self.terminate_episode()
        self.begin_episode()
        
        obs = self._get_obs()
        info = self._get_info()
        if self.sim_mode == "full":
            obs = -1 * obs
        return obs, info
    
    def begin_episode (self) -> bool:
        self.state_history = []
        self.action_history = []
        self.is_episode_running = True
        self.timesteps = 0
        self.goal = np.array([random.uniform(-4.5, 4.5),random.uniform(-4.5, 4.5)])
        self.reward_paraboloid = paraboloid(self.goal)
        if (self.render_mode == "rgb_array") or (self.render_mode == "human"):
            self.plugin.update_goal(self.goal)
        # print(f'Sim initialized with a goal postion of {self.goal}')
        return self.plugin.begin_sim_episode()

    def terminate_episode (self) -> bool:
        self.is_episode_running = False
        # self.export_video()
        return self.plugin.end_sim_episode()
    
    def close (self):
        self.is_episode_running = False
        return self.terminate_episode()

    def has_fallen (self, state, tol_pos=0.18, tol_ori=1.2):
        if self.making_insufficient_contact(state) == 4:
            return True
        # ! THIS COULDN'T BE BROUGHT OVER FOR SOME REASON!
        if self.sim_mode != "full":
            return self.plugin.has_fallen()
        return False
    
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

    def at_goal (self, obs, tol=0.05):
        if np.linalg.norm(np.array([obs.body_pose.position.x, obs.body_pose.position.y]) - self.goal) < tol:
            return True
        return False
    
    def foot_at_goal (self, obs, tol=0.01):
        if np.linalg.norm(np.array([obs.foot0.pose.position.x, obs.foot0.pose.position.y]) - self.goal) < tol:
                return True
        return False

    def get_foot_from_action (self, x):
        if x > 0.5:
            return 3
        if x > 0.0:
            return 2
        if x < -0.5:
            return 0
        return 1

    def gym_2_action (self, gym_action:np.array) -> MagnetoAction:
        action = MagnetoAction()
        action.pose.position.x = self.max_foot_step_size * gym_action[0]
        action.pose.position.y = self.max_foot_step_size * gym_action[1]
        action.idx = self.get_foot_from_action(gym_action[2])   
        return action
    
    def state_2_gym (self, state:MagnetoState) -> np.array:
        _, _, body_yaw = euler_from_quaternion(state.body_pose.orientation.w, state.body_pose.orientation.x, state.body_pose.orientation.y, state.body_pose.orientation.z)
        
        relative_goal = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, self.goal)
        
        # outs = extract_ground_frame_positions(state)
        # body_pos = np.array([outs['body'][0][0], outs['body'][1][0]])
        # foot0_pos = np.array([outs['feet'][0][0][0], outs['feet'][0][1][0]])
        # foot1_pos = np.array([outs['feet'][1][0][0], outs['feet'][1][1][0]])
        # foot2_pos = np.array([outs['feet'][2][0][0], outs['feet'][2][1][0]])
        # foot3_pos = np.array([outs['feet'][3][0][0], outs['feet'][3][1][0]])
        
        # gym_obs = np.concatenate((body_pos, foot0_pos, foot1_pos, foot2_pos, foot3_pos, relative_goal), dtype=np.float32)
        # gym_obs = np.concatenate((foot0_pos, foot1_pos, foot2_pos, foot3_pos, relative_goal), dtype=np.float32)
        
        # # ? maybe just return the magnetism scaling factor experienced by the foot closest to the goal
        # closest_foot_index = self.foot_closest_to_goal(state)
        # if closest_foot_index == 0:
        #     mag = state.foot0.magnetic_force
        # elif closest_foot_index == 1:
        #     mag = state.foot1.magnetic_force
        # elif closest_foot_index == 2:
        #     mag = state.foot2.magnetic_force
        # elif closest_foot_index == 3:
        #     mag = state.foot3.magnetic_force
        # gym_obs = np.concatenate((relative_goal, np.array([mag])), dtype=np.float32)
        
        relative_foot0 = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, np.array([state.foot0.pose.position.x, state.foot0.pose.position.y]))
        relative_foot1 = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, np.array([state.foot1.pose.position.x, state.foot1.pose.position.y]))
        relative_foot2 = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, np.array([state.foot2.pose.position.x, state.foot2.pose.position.y]))
        relative_foot3 = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, np.array([state.foot3.pose.position.x, state.foot3.pose.position.y]))
        
        gym_obs = np.concatenate((relative_foot0, relative_foot1, relative_foot2, relative_foot3, relative_goal), dtype=np.float32)
        
        return gym_obs
    
    def foot_closest_to_goal (self, state:MagnetoState, body_yaw=0):
        feet_pos = [
            np.array([state.foot0.pose.position.x, state.foot0.pose.position.y]),
            np.array([state.foot1.pose.position.x, state.foot1.pose.position.y]),
            np.array([state.foot2.pose.position.x, state.foot2.pose.position.y]),
            np.array([state.foot3.pose.position.x, state.foot3.pose.position.y]),
        ]
        relative_distances = np.zeros((4,), np.float32)
        for ii in range(len(feet_pos)):
            relative_distances[ii] = np.linalg.norm(global_to_body_frame(self.goal, body_yaw, feet_pos[ii]), 2)
        return np.argmin(relative_distances)

# # %%
# env = SimpleMagnetoEnv(sim_mode="simple")

# # %%
# env.reset()

# # %%
# state = env._get_obs('ros')

# # %%
# min = env.foot_closest_to_goal(state)

# # %%
