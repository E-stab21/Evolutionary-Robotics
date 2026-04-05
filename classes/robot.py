from pyrosim import pyrosim
import pybullet as p
from pyrosim.neuralNetwork import NEURAL_NETWORK
#personal
import generate
from classes.sensor import SENSOR
from classes.motor import MOTOR
import os
import constants as c

class ROBOT:
    def __init__(self, solutionID):
        #feilds
        self.solID = solutionID
        self.motors = {}
        self.sensors = {}
        self.id = None

        #body
        self.id = p.loadURDF("../body.urdf")
        pyrosim.Prepare_To_Simulate(self.id)
        self.Prepare_To_Sense()
        self.Prepare_To_Act()

        #brain
        self.nn = NEURAL_NETWORK("brain" + str(solutionID) + ".nndf")

        os.system("del brain" + str(solutionID) + ".nndf")

    def Prepare_To_Sense(self):
        for linkName in pyrosim.linkNamesToIndices:
            self.sensors[linkName] = SENSOR(linkName)

    def Sense(self, t):
        for sensor in self.sensors.values():
            sensor.Get_Value(t)

    def Prepare_To_Act(self):
        for jointName in pyrosim.jointNamesToIndices:
            self.motors[jointName] = MOTOR(jointName)

    def Act(self):
        for neuronName in self.nn.Get_Neuron_Names():
            if self.nn.Is_Motor_Neuron(neuronName):
                jointName = self.nn.Get_Motor_Neuron_Joint(neuronName).encode('utf-8')
                desiredAngle = self.nn.Get_Value_Of(neuronName) * c.motorJointRange
                self.motors[jointName].Set_Value(desiredAngle, self.id)

    def Think(self):
        self.nn.Update()

    def Get_Fitness(self):
        stateOfLinkZero = p.getLinkState(self.id, 0)
        positionOfLinkZero = stateOfLinkZero[0]
        xCoordinateOfLinkZero = positionOfLinkZero[0]
        with open(f"tmp{self.solID}.txt", "w") as f:
            f.write(str(xCoordinateOfLinkZero))
        os.rename("tmp" + str(self.solID) + ".txt", "fitness" + str(self.solID) + ".txt")
