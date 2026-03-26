import pybullet as p
import time
#personal imports
from classes.world import WORLD
from classes.robot import ROBOT

class SIMULATION:
    def __init__(self, directOrGUI):
        if directOrGUI == "GUI":
            self.gui = True
            p.connect(p.GUI)
        elif directOrGUI == "DIRECT":
            self.gui = False
            p.connect(p.DIRECT)

        #fields
        self.world = WORLD()
        self.robot = ROBOT()

    def Run(self):
        for t in range(400):
            p.stepSimulation()
            self.robot.Sense(t)
            self.robot.Think()
            self.robot.Act()
            if self.gui:
                time.sleep(1 / 100)
            else:
                time.sleep(1 / 500)
        self.Get_Fitness()

    def Get_Fitness(self):
        self.robot.Get_Fitness()

    def __del__(self):
        p.disconnect()



