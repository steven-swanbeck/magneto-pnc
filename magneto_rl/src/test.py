#!/usr/bin/env python3
# * Goals
# . Launch node using roslaunch api
# . Click on screen in correct location to focus window then unpause simulation
# . Kill node and repeat

import rospy
import roslaunch
# from subprocess import Popen, PIPE
import pyautogui
import time

if __name__ == "__main__":
    
    rospy.init_node('roslaunch_test')
    
    node = roslaunch.core.Node('my_simulator', 
                            'magneto_ros',
                            args='config/Magneto/USERCONTROLWALK.yaml')
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()
    
    process = launch.launch(node)
    print(process.is_alive)
    
    # rospy.sleep(5)
    time.sleep(3)
    
    pyautogui.doubleClick(1440 + 500/2, 10)
    pyautogui.click(1440 + 500/2, 500/2)
    pyautogui.press('space')
    time.sleep(1)
    
    
    pyautogui.press('s')
    time.sleep(5)
    
    print("Forcing shutdown!")
    pyautogui.click(1899, 21)
    process.stop()
    time.sleep(1)
    pyautogui.click(1440 + 500/2, 500/2)
    with pyautogui.hold('ctrl'):
        pyautogui.press('c')
    
    while not rospy.is_shutdown():
        rospy.spin()

# %%
import pyautogui
screenWidth, screenHeight = pyautogui.size()
print(screenWidth, screenHeight)

currentMouseX, currentMouseY = pyautogui.position()
print(currentMouseX, currentMouseY)

# # %%
# # pyautogui.moveTo(1650, 150)
# pyautogui.click(1650, 150)
# pyautogui.press('space')

# # %%
# pyautogui.doubleClick(1650, 150)

# %%
