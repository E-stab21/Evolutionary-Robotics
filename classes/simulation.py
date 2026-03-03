import pybullet as p
#personal imports
from classes.world import WORLD
from classes.robot import ROBOT

class SIMULATION:
    def __init__(self):
        #feilds
        self.world = WORLD()
        self.robot = ROBOT()

    def __del__(self):
        p.disconnect()



