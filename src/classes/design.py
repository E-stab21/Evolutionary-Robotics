import pickle
import subprocess
import random
import os
import numpy as np
from pyrosim import pyrosim

class Design:
    next_id = 0

    def __init__(self):
        self.id = Design.next_id
        Design.next_id += 1
        self.body_file = "dfs/quadruped.urdf"
        self.num_of_sensor_neurons = 9
        self.num_of_motor_neurons = 8
        self.process = None
        self.fitness = None

    def wait(self):
        if self.process:
            self.process.wait()
            self.process = None
            with open(f"fitness{self.id}.txt", "r", encoding="utf-8") as f:
                self.fitness = float(f.read())
            os.remove(f"fitness{self.id}.txt")
        else:
            print("no process to wait on")
    
    def __lt__(self, other):
        return (
            self.fitness > other.fitness
            and self.fitness is not None
            and other.fitness is not None
        )

    def generate_body(self):
        pyrosim.Start_URDF(self.body_file)
        #torso
        pyrosim.Send_Cube(name="Torso", pos=[0, 0, 1], size=[1, 1, 1])
        #frontleg
        pyrosim.Send_Joint(name="Torso_FrontLeg", parent="Torso", child="FrontLeg", type="revolute",
                           position=[0, 0.5, 1], jointAxis="1 0 0")
        pyrosim.Send_Cube(name="FrontLeg", pos=[0, 0.5, 0], size=[0.2, 1, 0.2])
        pyrosim.Send_Joint(name="FrontLeg_FrontFoot", parent="FrontLeg", child="FrontFoot", type="revolute",
                           position=[0, 1, 0], jointAxis="1 0 0")
        pyrosim.Send_Cube(name="FrontFoot", pos=[0, 0, -0.5], size=[0.2, 0.2, 1])
        #backleg
        pyrosim.Send_Joint(name="Torso_BackLeg", parent="Torso", child="BackLeg", type="revolute",
                           position=[0, -0.5, 1], jointAxis="1 0 0")
        pyrosim.Send_Cube(name="BackLeg", pos=[0, -0.5, 0], size=[0.2, 1, 0.2])
        pyrosim.Send_Joint(name="BackLeg_BackFoot", parent="BackLeg", child="BackFoot", type="revolute",
                           position=[0, -1, 0], jointAxis="1 0 0")
        pyrosim.Send_Cube(name="BackFoot", pos=[0, 0, -0.5], size=[0.2, 0.2, 1])
        #leftleg
        pyrosim.Send_Joint(name="Torso_LeftLeg", parent="Torso", child="LeftLeg", type="revolute",
                           position=[-0.5, 0, 1], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="LeftLeg", pos=[-0.5, 0, 0], size=[1, 0.2, 0.2])
        pyrosim.Send_Joint(name="LeftLeg_LeftFoot", parent="LeftLeg", child="LeftFoot", type="revolute",
                           position=[-1, 0, 0], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="LeftFoot", pos=[0, 0, -0.5], size=[0.2, 0.2, 1])
        #rightleg
        pyrosim.Send_Joint(name="Torso_RightLeg", parent="Torso", child="RightLeg", type="revolute",
                           position=[0.5, 0, 1], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="RightLeg", pos=[0.5, 0, 0], size=[1, 0.2, 0.2])
        pyrosim.Send_Joint(name="RightLeg_RightFoot", parent="RightLeg", child="RightFoot", type="revolute",
                           position=[1, 0, 0], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="RightFoot", pos=[0, 0, -0.5], size=[0.2, 0.2, 1])
        pyrosim.End()

class Ctrnn(Design):
    def __init__(self, is_euler):
        super().__init__()
        self.brain_file = f"dfs/brain{self.id}.pkl"
        self.is_euler = is_euler
        self.n = 20
        self.time_constant = 1
        self.sensor_neurons = list(range(self.n))[:self.num_of_sensor_neurons]
        self.motor_neurons = list(range(self.n))[-self.num_of_motor_neurons:]
        self.weights = (np.random.rand(self.n, self.n) * 2) - 1
        self.bias = (np.random.rand(self.n) * 2) - 1
        self.vfunc = np.vectorize(np.tanh)

    def simulate(self, direct_or_gui="DIRECT"):
        self.generate_body()
        with open(self.brain_file, "wb") as f:
            pickle.dump(self, f)
        self.process = (
            subprocess.Popen(["python", "src/simulation.py", direct_or_gui,
            str(self.id), self.body_file, self.brain_file])
        )

    def mutate(self):
        random_row = random.randint(0, self.n - 1)
        random_column = random.randint(0, self.n - 1)
        random_bias = random.randint(0, self.n - 1)
        self.weights[random_row][random_column] = random.random() * 2 - 1
        self.bias[random_bias] = random.random() * 2 - 1

    def spawn(self):
        child = Ctrnn(self.is_euler)
        child.weights = self.weights.copy()
        child.bias = self.bias.copy()
        child.mutate()
        return child

class TrdNet(Design):
    def __init__(self):
        super().__init__()
        self.brain_file = f"dfs/brain{self.id}.nndf"
        self.weights = np.random.rand(self.num_of_sensor_neurons, self.num_of_motor_neurons) * 2 - 1

    def simulate(self, direct_or_gui="DIRECT"):
        self.generate_body()
        self.generate_brain()
        self.process = subprocess.Popen(["python", "src/simulation.py", direct_or_gui, str(self.id),
                                         self.body_file, self.brain_file])

    def mutate(self):
        random_row = random.randint(0, self.num_of_sensor_neurons - 1)
        random_column = random.randint(0, self.num_of_motor_neurons - 1)
        self.weights[random_row][random_column] = random.random() * 2 - 1

    def spawn(self):
        child = TrdNet()
        child.weights = self.weights.copy()
        child.mutate()
        return child

    def generate_brain(self):
        #sensor neurons
        pyrosim.Start_NeuralNetwork(self.brain_file)
        pyrosim.Send_Sensor_Neuron(name=0, linkName="Torso")
        pyrosim.Send_Sensor_Neuron(name=1, linkName="FrontLeg")
        pyrosim.Send_Sensor_Neuron(name=2, linkName="BackLeg")
        pyrosim.Send_Sensor_Neuron(name=3, linkName="LeftLeg")
        pyrosim.Send_Sensor_Neuron(name=4, linkName="RightLeg")
        pyrosim.Send_Sensor_Neuron(name=5, linkName="FrontFoot")
        pyrosim.Send_Sensor_Neuron(name=6, linkName="BackFoot")
        pyrosim.Send_Sensor_Neuron(name=7, linkName="LeftFoot")
        pyrosim.Send_Sensor_Neuron(name=8, linkName="RightFoot")

        #motor neurons
        pyrosim.Send_Motor_Neuron(name=9, jointName="Torso_FrontLeg")
        pyrosim.Send_Motor_Neuron(name=10, jointName="Torso_BackLeg")
        pyrosim.Send_Motor_Neuron(name=11, jointName="Torso_LeftLeg")
        pyrosim.Send_Motor_Neuron(name=12, jointName="Torso_RightLeg")
        pyrosim.Send_Motor_Neuron(name=13, jointName="FrontLeg_FrontFoot")
        pyrosim.Send_Motor_Neuron(name=14, jointName="BackLeg_BackFoot")
        pyrosim.Send_Motor_Neuron(name=15, jointName="LeftLeg_LeftFoot")
        pyrosim.Send_Motor_Neuron(name=16, jointName="RightLeg_RightFoot")

        #synapses
        for currentRow in range(self.num_of_sensor_neurons):
            for currentColumn in range(self.num_of_motor_neurons):
                pyrosim.Send_Synapse(sourceNeuronName=currentRow,
                                     targetNeuronName=currentColumn + self.num_of_sensor_neurons,
                                     weight=self.weights[currentRow][currentColumn])
        pyrosim.End()