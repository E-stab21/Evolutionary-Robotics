import random
import time
from cmath import inf
from classes.solution import SOLUTION
import constants as c
import copy
import os

class PARALLEL_HILL_CLIMBER:
    def __init__(self):
        os.system("del brain*.nndf")
        os.system("del fitness*.nndf")
        self.nextAvailableID = 0
        self.parents = {}
        self.children = None
        for index in range(c.populationSize):
            self.parents[index] = SOLUTION(self.nextAvailableID)
            self.nextAvailableID += 1


    def Evolve(self):
        self.Evaluate(self.parents)
        for currentGeneration in range(c.numberOfGenerations):
            self.Evolve_For_One_Generation()

    def Evolve_For_One_Generation(self):
        self.Spawn()
        self.Mutate()
        self.Evaluate(self.children)
        self.Print()
        self.Select()

    def Spawn(self):
        self.children = {}
        for parentID in self.parents.keys():
            self.children[parentID] = copy.deepcopy(self.parents[parentID])


    def Mutate(self):
        for child in self.children.values():
            child.Mutate()

    def Select(self):
        for key in self.parents.keys():
            if self.children[key].fitness < self.parents[key].fitness:
                self.parents[key] = copy.deepcopy(self.children[key])

    def Print(self):
        print()
        print()
        for key in self.parents.keys():
            print(f"{self.parents[key].fitness:.2f} | {self.children[key].fitness:.2f}")
        print()

    def Show_Best(self):
        bestParentKey = None
        bestFitness = inf
        for key in self.parents.keys():
            if self.parents[key].fitness < bestFitness:
                bestFitness = self.parents[key].fitness
                bestParentKey = key
        self.parents[bestParentKey].Evaluate("GUI")

    def Evaluate(self, solutions):
        for solution in solutions.values():
            solution.Start_Simulation("DIRECT")
        for solution in solutions.values():
            solution.Wait_For_Simulation_To_End()




