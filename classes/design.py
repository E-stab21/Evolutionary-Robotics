import numpy as np
from pyrosim import pyrosim
import random
import os
import time
import constants as c

class SOLUTION:
    def __init__(self, ID):
        self.myID = ID
        self.weights = np.random.rand(c.numSensorNeurons, c.numMotorNeurons) * 2 - 1
        self.fitness = None

    def Evaluate(self, directOrGUI):
        self.Start_Simulation(directOrGUI)
        self.Wait_For_Simulation_To_End()

    def Mutate(self):
        randomRow = random.randint(0, 2)
        randomColumn = random.randint(0, 1)
        self.weights[randomRow][randomColumn] = random.random() * 2 - 1

    def Create_World(self):
        pyrosim.Start_SDF("../world.sdf")
        pyrosim.Send_Cube(name="Box", pos=[5, 0, 0.5], size=[1, 1, 1])
        pyrosim.End()

    def Create_Robot(self):
        self.Generate_Body()
        self.Generate_Brain()

    def Generate_Body(self):
        pyrosim.Start_URDF("../body.urdf")
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


    def Generate_Brain(self):
        #sensor neurons
        pyrosim.Start_NeuralNetwork("brain" + str(self.myID) + ".nndf")
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
        for currentRow in range(c.numSensorNeurons):
            for currentColumn in range(c.numMotorNeurons):
                pyrosim.Send_Synapse(sourceNeuronName=currentRow,
                                     targetNeuronName=currentColumn + c.numSensorNeurons,
                                     weight=self.weights[currentRow][currentColumn])
        pyrosim.End()

    def Set_ID(self, id):
        self.myID = id

    def Start_Simulation(self, directOrGUI):
        self.Create_World()
        self.Create_Robot()
        os.system(f"start /B python simulate.py {directOrGUI} {str(self.myID)}")

    def Wait_For_Simulation_To_End(self):
        while not os.path.exists(f"fitness{str(self.myID)}.txt"):
            time.sleep(0.01)
        with open(f"fitness{str(self.myID)}.txt") as f:
            self.fitness = float(f.read())
        os.system(f"del fitness{str(self.myID)}.txt")
