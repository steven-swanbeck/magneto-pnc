#!/usr/bin/env python3
import rospy
import roslaunch
import random
from std_srvs.srv import Trigger
from numpy import array, int32, zeros, sqrt



#Concept: Grab map dimensions (NxM grid, in meters). 
#Create a map of the same dimensions, with subintervals of X meters
#Create the image


#. Plugin has a reset function. 
# Will have a function that creates a map. Owns the map, and has other functions such as lookup value at pixel coordinates, 
# needs to have:
# Generate, which deletes old and builds new
# Access, which looks up pixel values
# could instead use opencv because it can easily grab pixel coords



# Grabs height, width from yaml, pixel size from user
# Calls create_map using pixel size (tuple) as inputs
# Receives pixel requests, returns value

#Recall that the (x,y) in cartesion space needs to be converted to pixel domain

#Should create a safezone in where the robot spawns (which is what it calls 0,0)

# Robot knows its position wrt a global frame (which it spawns at 0,0), 



class MagnetismMapper (object):
    
    def __init__(self, height, width, resolution = 0.25, radius = 0.5) -> None:
        self.res = resolution
        self.h = height/resolution+1
        self.w = width/resolution+1
        self.rad = radius/resolution
        self.filename = "magnetism_map.pgm" #TODO make user input or automatically generate
        
        
    def create_map(self):
        #. The height and width should be odd numbers to allow for uniform size around origin
        # TODO: if h/w are even, add/sub one?
        #. The unit of resolution is meters/pixel
        #. The unit of radius should be in meters
        random.seed()
        self.map = self.pgmwrite(int(self.w), int(self.h))
        #Here, self.map is what we can index to grab magnetism in a specific pixel.
        self.map = self.starting_zone(self.map, self.rad)
        return self.map
        
        
    def pgmwrite(self, width, height, maxVal=10, magicNum='P2'):
        with open(self.filename, 'w') as self.f:
            img = zeros((height,width))
            self.f.write(magicNum + '\n')
            self.f.write(str(width) + ' ' + str(height) + '\n')
            self.f.write(str(maxVal) + '\n')
            for i in range(height):
                for j in range(width):
                    magnetism = random.gauss(mu=7.0, sigma=1.5)
                    magnetism = round(magnetism)
                    if magnetism > 10:
                        magnetism -= 5
                    self.f.write(str(magnetism) + ' ')
                    img[i][j] = magnetism
                self.f.write('\n')
            self.f.close()
            print("File closed at", self.filename)
            return img
    
    def pix2cart(self, x, y, w, h):
        x = x - (w-1)/2
        y = -(y-(h-1)/2)
        return x,y
        
    def cart2pix(self, x, y, w, h):
        x = x+(w-1)/2
        y = -(y-(h-1)/2)
        return x,y
        
    def starting_zone(self, map, radius):
        radius = int(radius)
        x_cen = int((self.w-1)/2)
        y_cen = int((self.h-1)/2)
        map[x_cen][y_cen] = 10
        for i in range(1,radius+1):
            for j in range(1,radius+1):
                map[x_cen+i][y_cen] = 10
                map[x_cen][y_cen+j] = 10
                map[x_cen-i][y_cen] = 10
                map[x_cen][y_cen-j] = 10
                if sqrt(i*i+j*j) <= radius:
                    map[x_cen+i][y_cen+j] = 10
                    map[x_cen+i][y_cen-j] = 10
                    map[x_cen-i][y_cen-j] = 10
                    map[x_cen-i][y_cen+j] = 10
        return self.map
        
    def pixel_grab(self, x, y):
        x,y = self.cart2pix(x,y,self.w,self.h)
        return self.map[int(x)][int(y)]
