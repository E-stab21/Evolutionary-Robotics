import pybullet as p
import time
#personal imports
from classes.world import WORLD
from classes.robot import ROBOT

class SIMULATION:
    def __init__(self):
        #feilds
        self.world = WORLD()
        self.robot = ROBOT()

    def Run(self):
        for t in range(400):
            p.stepSimulation()
            self.robot.Sense(t)
            self.robot.Think()
            self.robot.Act()
            time.sleep(1 / 100)

    def __del__(self):
        p.disconnect()



