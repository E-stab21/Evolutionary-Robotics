import pybullet as p
#personal imports
from world import WORLD
from robot import ROBOT

class SIMULATION:
    def __init__(self):
        #feilds
        self.world = WORLD()
        self.robot = ROBOT()

    def __del__(self):
        p.disconnect()



