import pybullet as p
from pyrosim import pyrosim
import pickle
from pyrosim.neuralNetwork import NEURAL_NETWORK
import numpy as np
import os
import constants as c

class Robot:
    def __init__(self, design_id, body_file, brain_file):
        #feilds
        self.id = p.loadURDF(body_file)
        self.design_id = design_id
        self.sensors = []
        self.motors = []

        #brain logic
        if brain_file[-4:] == "nndf":
            self.is_ctrnn = False
            self.nn = NEURAL_NETWORK(brain_file)
            os.remove(brain_file)
        else:
            self.is_ctrnn = True
            with open(brain_file, "rb") as f:
                self.brain = pickle.load(f)
            os.remove(brain_file)
            self.brain_state = np.zeros(self.brain.n)

        #set senors and motors
        pyrosim.Prepare_To_Simulate(self.id)
        for linkName in pyrosim.linkNamesToIndices:
            self.sensors.append(linkName)
        for jointName in pyrosim.jointNamesToIndices:
            self.motors.append(jointName)

    def sense(self):
        if not self.is_ctrnn:
            self.nn.Update()
        else:
            #Eulers's
            inputs = np.zeros(self.brain.n)
            inputs[self.brain.sensor_neurons] = [pyrosim.Get_Touch_Sensor_Value_For_Link(self.sensors[i])
                                                 for i in self.brain.sensor_neurons]
            self.brain_state = (self.brain_state + (c.deltaTime / self.brain.time_constant) *
                          (-self.brain_state + self.brain.weights @
                           self.brain.vfunc(self.brain_state + self.brain.bias) +
                           inputs))

    def act(self):
        if not self.is_ctrnn:
            for neuronName in self.nn.Get_Neuron_Names():
                if self.nn.Is_Motor_Neuron(neuronName):
                    jointName = self.nn.Get_Motor_Neuron_Joint(neuronName).encode('utf-8')
                    desiredAngle = self.nn.Get_Value_Of(neuronName) * c.motorJointRange
                    pyrosim.Set_Motor_For_Joint(
                        bodyIndex=self.id,
                        jointName=jointName,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=desiredAngle,
                        maxForce=50
                    )
        else:
            for i, jointName in enumerate(self.motors):
                desiredAngle = self.brain_state[self.brain.motor_neurons[i]] * c.motorJointRange
                pyrosim.Set_Motor_For_Joint(
                    bodyIndex=self.id,
                    jointName=jointName,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=desiredAngle,
                    maxForce=50
                )

    def write_fitness(self):
        basePositionAndOrientation = p.getBasePositionAndOrientation(self.id)
        basePosition = basePositionAndOrientation[0]
        with open(f"fitness{self.design_id}.txt", "w") as f:
            f.write(str(basePosition[0]))

