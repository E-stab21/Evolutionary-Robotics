from src.classes.design import Ctrnn
import glob
import os

POPULATION_SIZE = 10
NUMBER_OF_GENERATIONS = 50

rtk4_population = []

print("generate.py")

for _ in range(10):
    rtk4_population.append(Ctrnn(False))

for _ in range(NUMBER_OF_GENERATIONS):

    #simulating
    for index in range(POPULATION_SIZE):
        rtk4_population[index].simulate(direct_or_gui="DIRECT")
    for index in range(POPULATION_SIZE):
        rtk4_population[index].wait()

    #clean up
    files_ = glob.glob("dfs/brain*.nndf")
    for file_ in files_:
        os.remove(file_)
    files_ = glob.glob("dfs/brain*.pkl")
    for file_ in files_:
        os.remove(file_)

    #sort populations
    rtk4_population.sort()

    #clean bottom half of populations
    rtk4_population = rtk4_population[:POPULATION_SIZE // 2]

    #spawn new population
    for index in range(POPULATION_SIZE // 2):
        rtk4_population.append(rtk4_population[index].spawn())

print(rtk4_population[0].fitness)
rtk4_population[0].save()


