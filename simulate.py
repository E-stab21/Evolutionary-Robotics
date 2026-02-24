import pybullet as p
import time
#personal
from classes.simulation import SIMULATION

def Run():
    simulation = SIMULATION()

    for t in range(1000):
        p.stepSimulation()
        simulation.robot.Sense(t)
        simulation.robot.Act(t)
        time.sleep(1 / 100)

if __name__ == "__main__":
    Run()