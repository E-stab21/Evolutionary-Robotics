import pybullet as p
import pybullet_data
import sys
import time
from classes.robot import Robot

class Simulation:
    def __init__(self, direct_or_gui, design_id, body_file, brain_file):
        if direct_or_gui == "GUI":
            self.gui = True
            p.connect(p.GUI)
        elif direct_or_gui == "DIRECT":
            self.gui = False
            p.connect(p.DIRECT)

        #world
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadSDF("dfs/world.sdf")
        p.loadURDF("plane.urdf")

        #body
        self.robot = Robot(design_id, body_file, brain_file)

    def run(self):
        for t in range(1000):
            p.stepSimulation()
            self.robot.sense()
            self.robot.act()
            if self.gui:
                time.sleep(1 / 100)
        self.robot.write_fitness()

    def __del__(self):
        p.disconnect()

if __name__ == "__main__":
    #run sim
    sim = Simulation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    sim.run()




