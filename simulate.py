#personal
from classes.simulation import SIMULATION
import sys


if __name__ == "__main__":
    solutionID = sys.argv[2]
    sim = SIMULATION(sys.argv[1], solutionID)
    sim.Run()