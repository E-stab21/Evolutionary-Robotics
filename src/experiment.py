"""
Experiment for the CS3060 project.
"""
import os
import matplotlib.pyplot as plt
from pyrosim import pyrosim
from src.classes.design import Ctrnn, TrdNet
import constants as c


def create_world():
    pyrosim.Start_SDF("dfs/world.sdf")
    pyrosim.End()

def test(show_best=False):
    trd_population = []
    euler_population = []
    rtk4_population = []
    trd_fitnesses = []
    euler_fitnesses = []
    rtk4_fitnesses = []

    #create the world
    create_world()

    #create populations
    for _ in range(c.POPULATION_SIZE):
        trd_population.append(TrdNet())
        euler_population.append(Ctrnn(True))
        rtk4_population.append(Ctrnn(False))

    #evolve
    for _ in range(c.NUMBER_OF_GENERATIONS):

        #simulating
        for index in range(c.POPULATION_SIZE):
            trd_population[index].simulate(direct_or_gui="DIRECT")
            euler_population[index].simulate(direct_or_gui="DIRECT")
            rtk4_population[index].simulate(direct_or_gui="DIRECT")
        for index in range(c.POPULATION_SIZE):
            trd_population[index].wait()
            euler_population[index].wait()
            rtk4_population[index].wait()

        #sort populations
        trd_population.sort()
        euler_population.sort()
        rtk4_population.sort()

        #clean bottom half of populations
        trd_population = trd_population[:c.POPULATION_SIZE // 2]
        euler_population = euler_population[:c.POPULATION_SIZE // 2]
        rtk4_population = rtk4_population[:c.POPULATION_SIZE // 2]

        #spawn new population
        for index in range(c.POPULATION_SIZE // 2):
            trd_population.append(trd_population[index].spawn())
            euler_population.append(euler_population[index].spawn())
            rtk4_population.append(rtk4_population[index].spawn())

        #append top fitnesses to list
        trd_fitnesses.append(trd_population[0].fitness)
        euler_fitnesses.append(euler_population[0].fitness)
        rtk4_fitnesses.append(rtk4_population[0].fitness)

    #show best
    if show_best:
        trd_population[0].simulate(direct_or_gui="GUI")
        trd_population[0].wait()
        euler_population[0].simulate(direct_or_gui="GUI")
        euler_population[0].wait()
        rtk4_population[0].simulate(direct_or_gui="GUI")
        rtk4_population[0].wait()

    #cleanup
    if os.path.exists("fitness*.txt"):
        os.remove("fitness*.txt")

    return trd_fitnesses, euler_fitnesses, rtk4_fitnesses

if __name__ == "__main__":
    trd_trials = []
    euler_trials = []
    rtk4_trials = []

    for _ in range(c.NUMBER_OF_TRIALS):
        test_fitnesses = test()
        trd_trials.append(test_fitnesses[0])
        euler_trials.append(test_fitnesses[1])
        rtk4_trials.append(test_fitnesses[2])

    for i in range(c.NUMBER_OF_TRIALS):
        plt.plot(trd_trials[i], label=f"TRD Trial {i + 1}", linewidth=1)
        plt.plot(euler_trials[i], label=f"EULER Trial {i + 1}", linewidth=2, linestyle="dashed")
        plt.plot(rtk4_trials[i], label=f"RK4 Trial {i + 1}", linewidth=3, linestyle="dotted")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig("data/experiment.png")
