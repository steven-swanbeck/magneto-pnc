#!/usr/bin/env python3
import os
import sys
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point, Pose
# from magneto_rl.srv import FootPlacement
from magneto_rl.srv import FootPlacement, FootPlacementResponse
from magneto_rl.srv import UpdateMagnetoAction, UpdateMagnetoActionResponse
from magneto_rl.srv import ReportMagnetoState, ReportMagnetoStateResponse
import roslaunch
import time
import pyautogui
from seed_magnetism import MagnetismMapper


class MagnetoRLPlugin (object):
    
    def __init__(self) -> None:
        rospy.init_node('magneto_rl_manager')
        
        self.next_step_service = rospy.Service('determine_next_step', FootPlacement, self.determine_next_step)
        self.test_action_service = rospy.Service('test_action_command', Trigger, self.test_action_command)
        self.set_magneto_action = rospy.ServiceProxy('set_magneto_action', UpdateMagnetoAction)
        self.command_sleep_duration = rospy.get_param('/magneto/simulation/resume_duration')
        self.test_state_service = rospy.Service('test_state_getter', Trigger, self.test_get_state)
        self.test_begin_episode = rospy.Service('test_episode_begin', Trigger, self.begin_episode_cb)
        self.test_end_episode = rospy.Service('test_episode_end', Trigger, self.end_episode_cb)
        self.test_reset_episode = rospy.Service('test_episode_reset', Trigger, self.reset_episode_cb)
        self.test_map = rospy.Service('test_map', Trigger, self.get_map)
        
        self.link_idx = {
            'AR':rospy.get_param('/magneto/simulation/link_idx/AR'),
            'AL':rospy.get_param('/magneto/simulation/link_idx/AL'),
            'BL':rospy.get_param('/magneto/simulation/link_idx/BL'),
            'BR':rospy.get_param('/magneto/simulation/link_idx/BR'),
        }
        self.naive_walk_order = ['AR', 'AL', 'BL', 'BR']
        self.last_foot_placed = None
        
        self.vertical_pixel_calibration_offset = rospy.get_param('/magneto/simulation/vertical_pixel_calibration_offset')
        
        self.mag_map = MagnetismMapper(rospy.get_param('/magneto/simulation/magnetism_map/wall_geometry/width'), rospy.get_param('/magneto/simulation/magnetism_map/wall_geometry/height'))
    
    # . Testing functions
    # TODO validate the functionality of magnetism stuff
    def get_map (self, msg:Trigger):
        self.mag_map.create_map()
        value = self.mag_map.pixel_grab(1.5, 1)
        print(value)
        return True, ''
    
    def begin_episode_cb (self, msg:Trigger):
        success = self.begin_sim_episode()
        return success, ''
    
    def end_episode_cb (self, msg:Trigger):
        success = self.end_sim_episode()
        return success, ''
    
    def reset_episode_cb (self, msg:Trigger):
        self.end_sim_episode()
        self.begin_sim_episode()
        
        test = Trigger()
        self.test_action_command(test)
        self.test_action_command(test)
        self.test_action_command(test)
        self.test_action_command(test)
        self.test_get_state(test)
        
        return True, ''
    
    def test_get_state (self, msg):
        # srv = ReportMagnetoState()
        res = self.get_magneto_state()
        rospy.loginfo(res)
        rospy.loginfo(f'Ground pose: {res.ground_pose}')
        rospy.loginfo(f'Ground pose: {res.body_pose}')
        return True, ''
    
    def test_action_command (self, msg):
        # srv = UpdateMagnetoAction()
        
        if self.last_foot_placed is not None:
            last_step_index = self.naive_walk_order.index(self.last_foot_placed)
            if last_step_index < len(self.naive_walk_order) - 1:
                self.last_foot_placed = self.naive_walk_order[last_step_index + 1]
            else:
                self.last_foot_placed = self.naive_walk_order[0]
        else:
            self.last_foot_placed = self.naive_walk_order[0]
        
        pose = Pose()
        pose.position.x = -0.1
        pose.position.y = 0.
        pose.position.z = 0.
        pose.orientation.w = 1.
        pose.orientation.x = 0.
        pose.orientation.y = 0.
        pose.orientation.z = 0.
        
        self.set_magneto_action(self.link_idx[self.last_foot_placed], pose)
        
        # ! Recognize this will be crucial to keep for the gym integration!
        rospy.sleep(self.command_sleep_duration)
        return True, ''
    
    # . Simulator Dynamics
    def provide_magnetic_force_modifier (self, msg):
        # TODO
        raise NotImplementedError
    
    def report_state (self):
        res = self.get_magneto_state()
        return res
    
    def update_action (self, link_id:str, pose:Pose) -> bool:
        # rospy.loginfo(f'[update_action] Setting new action for link_id {link_id} (link_idx: {self.link_idx[link_id]}) of pose\n{pose}')
        res = self.set_magneto_action(self.link_idx[link_id], pose)
        rospy.sleep(self.command_sleep_duration)
        return res.success
    
    def begin_sim_episode (self) -> bool:
        node = roslaunch.core.Node('my_simulator', 
                            'magneto_ros') #,
                            # args='config/Magneto/USERCONTROLWALK.yaml')
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        self.sim_process = launch.launch(node)
        # rospy.loginfo(f'[begin_sim_episode] Starting simulation episode. Process is alive: {self.sim_process.is_alive()}')
        time.sleep(3)
        
        self.set_magneto_action = rospy.ServiceProxy('set_magneto_action', UpdateMagnetoAction)
        self.get_magneto_state = rospy.ServiceProxy('get_magneto_state', ReportMagnetoState)
        
        #TODO: JARED: Trigger magnetism seed
        self.mag_map.create_map()
        # > Return "true I created that object"
        # > Then, later, can call a function from seed_magnetism that will grab pixel val
        
        pyautogui.doubleClick(1440 + 500/2, 10 + self.vertical_pixel_calibration_offset)
        pyautogui.click(1440 + 500/2, 500/2)
        pyautogui.press('space')
        time.sleep(1)
        
        pyautogui.press('s')
        # rospy.loginfo('[begin_sim_episode] Finished bringing up simulation.')
        return True
    
    def end_sim_episode (self) -> bool:
        # rospy.loginfo('[end_sim_episode] Ending simulation episode. Forcing shutdown of simulation...')
        pyautogui.click(1440 + 500/2, 500/2)
        # pyautogui.click(1899, 21 + self.vertical_pixel_calibration_offset)
        pyautogui.click(1901, 21 + self.vertical_pixel_calibration_offset)
        self.sim_process.stop()
        time.sleep(1)
        return not self.sim_process.is_alive()
    
    # . Deprecated
    def determine_next_step (self, req:FootPlacement) -> FootPlacementResponse:
        
        rospy.loginfo(f'Base pose:\n{req.base_pose}')
        rospy.loginfo(f'p_al:\n{req.p_al}')
        rospy.loginfo(f'p_ar:\n{req.p_ar}')
        rospy.loginfo(f'p_bl:\n{req.p_bl}')
        rospy.loginfo(f'p_br:\n{req.p_br}')
        
        point = Point()
        point.x = -1 * float(input("Enter x step size: "))
        point.y = -1 * float(input("Enter y step size: "))
        point.z = 0
        
        if self.last_foot_placed is not None:
            last_step_index = self.naive_walk_order.index(self.last_foot_placed)
            if last_step_index < len(self.naive_walk_order) - 1:
                self.last_foot_placed = self.naive_walk_order[last_step_index + 1]
            else:
                self.last_foot_placed = self.naive_walk_order[0]
        else:
            self.last_foot_placed = self.naive_walk_order[0]
        
        return self.link_idx[self.last_foot_placed], point

    def run (self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == "__main__":
    magneto_rl = MagnetoRLPlugin()
    magneto_rl.run()
