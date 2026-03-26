#personal
from classes.simulation import SIMULATION
import sys


if __name__ == "__main__":
    sim = SIMULATION(sys.argv[1])
    sim.Run()