import pybullet as p
import pybullet_data
#personal imports
from classes.robot import ROBOT
import time

class SIMULATION:
    def __init__(self, directOrGUI, solutionID):
        if directOrGUI == "GUI":
            self.gui = True
            p.connect(p.GUI)
        elif directOrGUI == "DIRECT":
            self.gui = False
            p.connect(p.DIRECT)

        #world
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadSDF("world.sdf")
        p.loadURDF("plane.urdf")

        #body
        self.robot = ROBOT(solutionID)

    def Run(self):
        for t in range(1000):
            p.stepSimulation()
            self.robot.Sense(t)
            self.robot.Think()
            self.robot.Act()
            if self.gui:
                time.sleep(1 / 100)
        self.robot.Get_Fitness()

    def __del__(self):
        p.disconnect()



