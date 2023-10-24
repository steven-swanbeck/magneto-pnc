#!/usr/bin/env python3
import numpy as np
import math
from geometry_msgs.msg import Pose
import random
import tqdm

class MagnetoAction (object):
    def __init__(self, idx:int=None, pose:Pose=None, link_idx:str=None) -> None:
        self.idx = idx
        if pose is not None:
            self.pose = pose
        else:
            self.pose = Pose()
            self.pose.orientation.w = 1.0
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

def return_closest (n, range):
    return min(range, key=lambda x:abs(x-n))
    
def iterate (env, num_steps=10, check_status=True):
    # pose = Pose()
    # pose.orientation.w = 1.
    # action = MagnetoAction(pose=pose)    
    env.reset()
    
    for i in range(num_steps):
        print(f'it: {i}')
        action = np.empty([3,], dtype=np.float64)
        
        # action[0] = random.randint(0,3)
        # action[1] = round(random.uniform(-0.2, 0.2), 2)
        # action[2] = round(random.uniform(-0.2, 0.2), 2)
        action[0] = random.uniform(-1, 1)
        action[1] = random.uniform(-1, 1)
        action[2] = random.uniform(-1, 1)
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
