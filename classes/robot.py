import pybullet as p
from pyrosim import pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
import os
#personal
from classes.sensor import SENSOR
from classes.motor import MOTOR
import constants as c

class ROBOT:
    def __init__(self, design):
        #feilds
        self.desId = design.id
        self.motors = {}
        self.sensors = {}
        self.id = None

        #body
        self.id = p.loadURDF("../body.urdf")
        pyrosim.Prepare_To_Simulate(self.id)
        #prep for sense
        for linkName in pyrosim.linkNamesToIndices:
            self.sensors[linkName] = SENSOR(linkName)
        #prep for act
        for jointName in pyrosim.jointNamesToIndices:
            self.motors[jointName] = MOTOR(jointName)

        #brain
        if design.isCTRN:

        else:
            self.nn = NEURAL_NETWORK("brain" + str(design.id) + ".nndf")
            os.system("del brain" + str(design.id) + ".nndf")

    def Sense(self, t):
        for sensor in self.sensors.values():
            sensor.Get_Value(t)

    def Act(self):
        for neuronName in self.nn.Get_Neuron_Names():
            if self.nn.Is_Motor_Neuron(neuronName):
                jointName = self.nn.Get_Motor_Neuron_Joint(neuronName).encode('utf-8')
                desiredAngle = self.nn.Get_Value_Of(neuronName) * c.motorJointRange
                self.motors[jointName].Set_Value(desiredAngle, self.id)

    def Think(self):
        self.nn.Update()

    def Get_Fitness(self):
        basePositionAndOrientation = p.getBasePositionAndOrientation(self.robot)
        basePosition = basePositionAndOrientation[0]
        xPosition = basePosition[0]
        with open(f"tmp{self.desId}.txt", "w") as f:
            f.write(str(xPosition))
        os.rename("tmp" + str(self.desId) + ".txt", "fitness" + str(self.desId) + ".txt")
