#!/usr/bin/env python3
import rospy
import roslaunch
import random
from std_srvs.srv import Trigger
from magneto_rl .srv import ReportMagnetoState
# from mapping_3d .srv import ReportMagnetoState #.For testing because of the weird cmake issues
from numpy import array, int32



#Concept: Grab map dimensions (NxM grid, in meters). 
#Create a map of the same dimensions, with subintervals of X meters
#Create the image

class SeedMagnetism (object):
    
    def __init__(self) -> None:
        rospy.init_node('magnetize_map')
        self.perform_seeding = rospy.Service('seed_magnetism', Trigger, self.create_map)
        self.get_magneto_state = rospy.ServiceProxy('get_magneto_state', ReportMagnetoState)
        
        self.filename = "testing_magnetism2.pgm" #TODO make user input or automatically generate
        
        
    def create_map(self, msg:Trigger):
        # res = self.get_magneto_state()
        # self.height = res.ground_height #TODO
        # self.width = res.ground_width #TODO
        self.height = 10
        self.width = 15
        #TODO: Assuming that height and width are in meters, divide by the desired chunk size (e.g. 10m^2 / 0.25m^2 -> 40x40 grid of magnetism)
        self.pgmwrite(self.width, self.height)
        
        
    def pgmwrite(self, width, height, maxVal=10, magicNum='P2'):
        img = []
        self.f.write(magicNum + '\n')
        self.f.write(str(width) + ' ' + str(height) + '\n')
        self.f.write(str(maxVal) + '\n')
        print("I made it here")
        for i in range(height):
            for j in range(width):
                magnetism = random.gauss(mu=7.0, sigma=1.5)
                round(magnetism)
                if magnetism > 10:
                    magnetism -= 5
                self.f.write(str(magnetism) + ' ')
            self.f.write('\n')
        self.f.close()
        
    
    def run(self):
        while not rospy.is_shutdown():
            with open(self.filename, 'w') as self.f:
                rospy.spin()


if __name__ == "__main__":
    seed = SeedMagnetism()
    seed.run()
