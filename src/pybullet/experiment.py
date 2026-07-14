"""
Experiment for the CS3060 project.
"""
import os
import glob
import matplotlib.pyplot as plt
from pyrosim import pyrosim
from src.classes.design import Ctrnn, TrdNet
import constants as c

def create_world():
    pyrosim.Start_SDF("dfs/world.sdf")
    pyrosim.End()

def run_trial():
    trd_population = []
    euler_population = []
    rtk4_population = []
    trd_fitnesses = []
    euler_fitnesses = []
    rtk4_fitnesses = []

    #create populations
    for _ in range(c.POPULATION_SIZE):
        trd_population.append(TrdNet())
        euler_population.append(Ctrnn(True))
        rtk4_population.append(Ctrnn(False))

    #evolve
    for _ in range(c.NUMBER_OF_GENERATIONS):
        print(f"Generation {_}")

        #simulating
        for index in range(c.POPULATION_SIZE):
            trd_population[index].simulate(direct_or_gui="DIRECT")
            euler_population[index].simulate(direct_or_gui="DIRECT")
            rtk4_population[index].simulate(direct_or_gui="DIRECT")
        for index in range(c.POPULATION_SIZE):
            trd_population[index].wait()
            euler_population[index].wait()
            rtk4_population[index].wait()

        #clean up
        files_ = glob.glob("dfs/brain*.nndf")
        for file_ in files_:
            os.remove(file_)
        files_ = glob.glob("dfs/brain*.pkl")
        for file_ in files_:
            os.remove(file_)

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

    return ((trd_fitnesses, euler_fitnesses, rtk4_fitnesses),
            (trd_population[0], euler_population[0], rtk4_population[0]))

if __name__ == "__main__":
    trd_trials = []
    euler_trials = []
    rtk4_trials = []
    trd_best = []
    euler_best = []
    rtk4_best = []

    # create the world
    create_world()

    #cleanup
    files = glob.glob("dfs/brain*.nndf")
    for file in files:
        os.remove(file)
    files = glob.glob("dfs/brain*.pkl")
    for file in files:
        os.remove(file)

    for trial in range(c.NUMBER_OF_TRIALS):
        (trd_trail_fitness, euler_trail_fitness, rtk4_trail_fitness), \
        (trd_trail_best, euler_trail_best, rtk4_trail_best) = run_trial()
        trd_trials.append(trd_trail_fitness)
        euler_trials.append(euler_trail_fitness)
        rtk4_trials.append(rtk4_trail_fitness)
        trd_best.append(trd_trail_best)
        euler_best.append(euler_trail_best)
        rtk4_best.append(rtk4_trail_best)
        print(f"Trail {trial + 1} done")

    #saving the best over all the trails
    trd_best_overall = trd_best[0]
    euler_best_overall = euler_best[0]
    rtk4_best_overall = rtk4_best[0]
    for i in range(len(trd_best)):
        if trd_best[i].fitness > trd_best_overall.fitness:
            trd_best_overall = trd_best[i]
        if euler_best[i].fitness > euler_best_overall.fitness:
            euler_best_overall = euler_best[i]
        if rtk4_best[i].fitness > rtk4_best_overall.fitness:
            rtk4_best_overall = rtk4_best[i]
    trd_best_overall.save()
    euler_best_overall.save()
    rtk4_best_overall.save()

    #plotting
    for i in range(c.NUMBER_OF_TRIALS):
        plt.plot(trd_trials[i], label=f"TRD Trial {i + 1}", linewidth=1, color="red")
        plt.plot(euler_trials[i], label=f"EULER Trial {i + 1}", linewidth=2, linestyle="dashed", color="blue")
        plt.plot(rtk4_trials[i], label=f"RK4 Trial {i + 1}", linewidth=3, linestyle="dotted", color="green")
    #plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig("data/experiment_.png")
