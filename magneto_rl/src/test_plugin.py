#!/usr/bin/env python3
import os
import sys
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point
# from magneto_rl.srv import FootPlacement
from magneto_rl.srv import FootPlacement, FootPlacementResponse

class MagnetoRLPlugin (object):
    
    def __init__(self) -> None:
        rospy.init_node('magneto_rl_manager')
        
        self.next_step_service = rospy.Service('determine_next_step', FootPlacement, self.determine_next_step)
        
        self.link_idx = {
            'AR':rospy.get_param('/magneto/simulation/link_idx/AR'),
            'AL':rospy.get_param('/magneto/simulation/link_idx/AL'),
            'BL':rospy.get_param('/magneto/simulation/link_idx/BL'),
            'BR':rospy.get_param('/magneto/simulation/link_idx/BR'),
        }
        self.naive_walk_order = ['AR', 'AL', 'BL', 'BR']
        self.last_foot_placed = None
    
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
