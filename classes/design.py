from pyrosim import pyrosim
import numpy as np
import pickle
import subprocess
import random
import os

class Design:
    def __init__(self, id_):
        self.id = id_
        self.body_file = "dfs/quadruped.urdf"
        self.num_of_sensor_neurons = 9
        self.num_of_motor_neurons = 8
        self.process = None
        self.fitness = None

    def set_id(self, id_):
        self.id = id_

    def wait(self):
        if self.process:
            self.process.wait()
            self.process = None
            with open(f"fitness{self.id}.txt", "r", encoding="utf-8") as f:
                self.fitness = float(f.read())
            os.remove(f"fitness{self.id}.txt")
        else:
            print("no process to wait on")

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
        pyrosim.Send_Cube(name="FrontFoot", pos=[0, 0.5, 0], size=[0.2, 1, 0.2])
        #backleg
        pyrosim.Send_Joint(name="Torso_BackLeg", parent="Torso", child="BackLeg", type="revolute",
                           position=[0, -0.5, 1], jointAxis="1 0 0")
        pyrosim.Send_Cube(name="BackLeg", pos=[0, -0.5, 0], size=[0.2, 1, 0.2])
        pyrosim.Send_Joint(name="BackLeg_BackFoot", parent="BackLeg", child="BackFoot", type="revolute",
                           position=[0, -1, 0], jointAxis="1 0 0")
        pyrosim.Send_Cube(name="BackFoot", pos=[0, -0.5, 0], size=[0.2, 1, 0.2])
        #leftleg
        pyrosim.Send_Joint(name="Torso_LeftLeg", parent="Torso", child="LeftLeg", type="revolute",
                           position=[-0.5, 0, 1], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="LeftLeg", pos=[-0.5, 0, 0], size=[1, 0.2, 0.2])
        pyrosim.Send_Joint(name="LeftLeg_LeftFoot", parent="LeftLeg", child="LeftFoot", type="revolute",
                           position=[-1, 0, 0], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="LeftFoot", pos=[-0.5, 0, 0], size=[1, 0.2, 0.2])
        #rightleg
        pyrosim.Send_Joint(name="Torso_RightLeg", parent="Torso", child="RightLeg", type="revolute",
                           position=[0.5, 0, 1], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="RightLeg", pos=[0.5, 0, 0], size=[1, 0.2, 0.2])
        pyrosim.Send_Joint(name="RightLeg_RightFoot", parent="RightLeg", child="RightFoot", type="revolute",
                           position=[1, 0, 0], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="RightFoot", pos=[0.5, 0, 0], size=[1, 0.2, 0.2])
        pyrosim.End()

class Ctrnn(Design):
    def __init__(self, id_):
        super().__init__(id_)
        self.n = 30
        self.time_constant = 1
        self.sensor_neurons = list(range(self.n))[:self.num_of_sensor_neurons]
        self.motor_neurons = list(range(self.n))[-self.num_of_motor_neurons:]
        self.weights = (np.random.rand(self.n, self.n) * 2) - 1
        self.bias = (np.random.rand(self.n) * 2) - 1
        self.vfunc = np.vectorize(np.tanh)

    def simulate(self, direct_or_gui="DIRECT"):
        self.generate_body()
        with open(f"brain{self.id}.pkl", "wb") as f:
            pickle.dump(self, f)
        self.process = subprocess.Popen(["python", "simulation.py", direct_or_gui, str(self.id),
                                         self.body_file, f"brain{self.id}.pkl"])

    def mutate(self):
        randomRow = random.randint(0, 2)
        randomColumn = random.randint(0, 1)
        self.weights[randomRow][randomColumn] = random.random() * 2 - 1

class TrdNet(Design):
    def __init__(self, id_):
        super().__init__(id_)
        self.weights = np.random.rand(self.num_of_sensor_neurons, self.num_of_motor_neurons) * 2 - 1

    def simulate(self, direct_or_gui="DIRECT"):
        self.generate_body()
        self.generate_brain()
        self.process = subprocess.Popen(["python", "simulation.py", direct_or_gui, str(self.id),
                                         self.body_file, f"brain{self.id}.nndf"])

    def mutate(self):
        randomRow = random.randint(0, 2)
        randomColumn = random.randint(0, 1)
        self.weights[randomRow][randomColumn] = random.random() * 2 - 1

    def generate_brain(self):
        #sensor neurons
        pyrosim.Start_NeuralNetwork(f"brain{self.id}.nndf")
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