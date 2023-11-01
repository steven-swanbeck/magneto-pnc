#!/usr/bin/env python3
# %%
import numpy as np
from magneto_utils import *
from seed_magnetism import MagnetismMapper
import pygame
import time

class SimpleSimPlugin(object):
    
    # WIP
    def __init__(self, render_mode, render_fps) -> None:
        self.link_idx = {
            'AR':0,
            'AL':1,
            'BL':2,
            'BR':3,
        }
        self.wall_size = 5
        self.mag_map = MagnetismMapper(self.wall_size, self.wall_size)
        self.render_mode = render_mode
        self.fps = render_fps
        self.window = None
        self.clock = None
        self.window_size = 500
        self.scale = 500 / 5 #pixels/m
        self.heading_arrow_length = 0.2
        self.leg_length = 0.2
        self.body_radius = 0.08
        self.foot_radius = 0.03
        self.body_width = 0.2 #m
        self.body_width_pixels = self.scale * self.body_width
        self.body_height = 0.3 #m
        self.body_height_pixels = self.scale * self.body_height
        self.goal = np.array([1, 1]) # !
        self.heading = 0
        self.tolerable_foot_displacement = np.array([0.08, 0.35])
    
    # DONE
    def report_state (self) -> MagnetoState:
        if self.has_fallen():
            for ii in range(len(self.foot_poses)):
                self.foot_poses[ii].position.z += 1.
        return StateRep(self.ground_pose, self.body_pose, self.foot_poses, self.foot_mags)
    
    # DONE
    def update_goal (self, goal):
        self.goal = goal
    
    # WIP
    def update_action (self, link_id:str, pose:Pose) -> bool:
        # & THESE NEED TO BE UPDATED IN BODY FRAME!!!
        update = body_to_global_frame(self.heading, np.array([pose.position.x, pose.position.y]))
        # print('-----------')
        # print(f'update: {update}')
        # print(f'before: {np.array([self.foot_poses[self.link_idx[link_id]].position.x, self.foot_poses[self.link_idx[link_id]].position.y])}')
        # self.foot_poses[self.link_idx[link_id]].position.x += update[0]
        # self.foot_poses[self.link_idx[link_id]].position.y += update[1]
        self.foot_poses[self.link_idx[link_id]].position.x += update[0] # ? switching these to try to better correspond with the full sim
        self.foot_poses[self.link_idx[link_id]].position.y += update[1]
        
        # TODO set magnetism property
        
        pos, heading = self.calculate_body_pose()
        self.body_pose.position.x = pos[0]
        self.body_pose.position.y = pos[1]
        self.body_pose.orientation.w = np.sin(heading)
        self.body_pose.orientation.z = np.cos(heading)
        # self.heading = heading # ! adding this back in has a huge impact on performance
        
        # print(f'after: {np.array([self.foot_poses[self.link_idx[link_id]].position.x, self.foot_poses[self.link_idx[link_id]].position.y])}')
    
    # DONE
    def begin_sim_episode (self) -> bool:
        self.ground_pose = Pose()
        self.ground_pose.orientation.w = 1.
        self.body_pose = Pose()
        self.body_pose.orientation.w = 1.
        self.foot_poses = [Pose(), Pose(), Pose(), Pose()]
        for ii in range(len(self.foot_poses)):
            self.foot_poses[ii].orientation.w = 1.
        self.foot_mags = [147., 147., 147., 147.]
        self.heading = 0
        self.spawn_robot()
        _, self.heading = self.calculate_body_pose()

    # DONE
    def end_sim_episode (self) -> bool:
        # TODO reset any storage variables or similar
        # ? should this be here?
        # self.close()
        pass
    
    # DONE
    def _render_frame (self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((100, 100, 100))
        
        body_center = (np.array([self.body_pose.position.x, self.body_pose.position.y])) * self.scale + np.array([self.window_size/2, self.window_size/2])
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            center=body_center,
            radius=self.body_radius * self.scale,
        )
        
        foot_pixel_positions = [(np.array([self.foot_poses[ii].position.x, self.foot_poses[ii].position.y])) * self.scale + np.array([self.window_size/2, self.window_size/2]) for ii in range(len(self.foot_poses))]
        for ii in range(len(self.foot_poses)):
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                center=foot_pixel_positions[ii],
                radius=self.foot_radius * self.scale,
            )
        
        heading_end = (np.array([self.body_pose.position.x, self.body_pose.position.y]) + np.array([self.heading_arrow_length * np.cos(self.heading), self.heading_arrow_length * np.sin(self.heading)])) * self.scale + np.array([self.window_size/2, self.window_size/2])
        pygame.draw.line(
                canvas,
                0,
                start_pos=body_center,
                end_pos=heading_end,
                width=3,
            )
        
        goal_center = self.goal * self.scale + np.array([self.window_size/2, self.window_size/2])
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            center=goal_center,
            radius=self.body_radius * self.scale,
        )
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            
            myfont = pygame.font.SysFont("monospace", 15)
            for ii in range(len(self.foot_poses)):
                label = myfont.render(str(ii), 1, (255,255,0))
                self.window.blit(label, foot_pixel_positions[ii])
        
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps)
            
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    # DONE
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    # DONE
    def spawn_robot (self):
        self.body_pose.position.x = 0
        self.body_pose.position.y = 0
        
        # leg_angles = [5 * np.pi / 4, 7 * np.pi / 4, 1 * np.pi /4, 3 * np.pi / 4]
        leg_angles = [7 * np.pi / 4, 1 * np.pi / 4, 3 * np.pi /4, 5 * np.pi / 4]
        for ii in range(len(self.foot_poses)):
            self.foot_poses[ii].position.x += self.leg_length * np.cos(leg_angles[ii])
            self.foot_poses[ii].position.y += self.leg_length * np.sin(leg_angles[ii])
        
    # DONE
    def calculate_body_pose (self):
        # . Pos is average position of the four feet
        # . Heading is perpendicular to
        
        px = np.mean([pose.position.x for pose in self.foot_poses])
        py = np.mean([pose.position.y for pose in self.foot_poses])
        
        pos = np.array([px, py])
        feet_pos = [[pose.position.x, pose.position.y] for pose in self.foot_poses]
        rel_feet_pos = [np.array(foot_pos) - pos for foot_pos in feet_pos]
        
        front_leg_v = rel_feet_pos[2] - rel_feet_pos[3]
        rear_leg_v = rel_feet_pos[1] - rel_feet_pos[0]
        
        # angle of front minus angle of rear + angle of rear + pi/2
        theta_front = np.arctan2(front_leg_v[0], front_leg_v[1])
        theta_rear = np.arctan2(rear_leg_v[0], rear_leg_v[1])
        theta_average = (theta_front + theta_rear) / 2
        heading = theta_average# + np.pi / 2
        
        return pos, heading
    
    # TODO add utility to trigger state report that would be considered fall if the feet get too far apart or in bad spots
    def has_fallen (self):
        if (np.abs(self.body_pose.position.x) > self.wall_size) or (np.abs(self.body_pose.position.y > self.wall_size)):
            return True
        # TODO Check if feet are in a substantially weird configuration
        # body_pos = np.array([self.body_pose.position.x, self.body_pose.position.y])
        # feet_pos = [np.array([self.foot_poses[ii].position.x, self.foot_poses[ii].position.y]) for ii in range(len(self.foot_poses))]
        
        # for ii in range(len(self.feet_pos)):
        #     norm = np.linalg.norm(feet_pos[ii] - body_pos, 1)
        #     if norm > self.tolerable_foot_displacement[1]:
        #         return True
        #     if norm < self.tolerable_foot_displacement[0]:
        #         return True
        
        return False
    
    # TODO add randomness to where the feet end up


# %%
# sim = SimpleSimPlugin()
# sim.begin_sim_episode()
# sim._render_frame()
# time.sleep(1)
# foot_pose = Pose()
# foot_pose.position.x = -0.08
# foot_pose.position.y = 0.
# sim.update_action('AR', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('AL', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('BR', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('BL', foot_pose)
# sim._render_frame()
# time.sleep(1)

# sim.update_action('AR', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('AL', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('BR', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('BL', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('AR', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('AL', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('BR', foot_pose)
# sim._render_frame()
# time.sleep(1)
# sim.update_action('BL', foot_pose)
# sim._render_frame()
# time.sleep(1)

# sim.close()



# . Compatability standins
class StateRep(object):
    def __init__(self, ground_pose, body_pose, foot_poses, foot_mags) -> None:
        self.ground_pose = ground_pose
        self.body_pose = body_pose
        self.AR_state = FootStateRep(foot_poses[0], foot_mags[0])
        self.AL_state = FootStateRep(foot_poses[1], foot_mags[1])
        self.BL_state = FootStateRep(foot_poses[2], foot_mags[2])
        self.BR_state = FootStateRep(foot_poses[3], foot_mags[3])
        
class FootStateRep(object):
    def __init__(self, pose, force) -> None:
        self.pose = pose
        self.magnetic_force = force

# %%
