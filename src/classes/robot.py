import pybullet as p
from pyrosim import pyrosim
import pickle
from pyrosim.neuralNetwork import NEURAL_NETWORK
import numpy as np
import os
import constants as c

class Robot:
    def __init__(self, fitness_id, body_file, brain_file):
        #fields
        self.id = p.loadURDF(body_file)
        self.fitness_id = fitness_id
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
        for link_name in pyrosim.linkNamesToIndices:
            self.sensors.append(link_name)
        for joint_name in pyrosim.jointNamesToIndices:
            self.motors.append(joint_name)

    def sense(self):
        if not self.is_ctrnn:
            self.nn.Update()
        elif self.brain.is_euler:
            #Euler's
            #print(f"Euler: {self.brain_state}")
            inputs = np.zeros(self.brain.n)
            inputs[self.brain.sensor_neurons] = (
                [pyrosim.Get_Touch_Sensor_Value_For_Link(self.sensors[i]) for i in self.brain.sensor_neurons]
            )
            self.brain_state = (
                    self.brain_state + (c.DELTA_TIME / self.brain.time_constant) * self.f(self.brain_state, inputs)
            )
        else:
            #Runge-Kutta 4
            #print(f"RTK4: {self.brain_state}")
            inputs = np.zeros(self.brain.n)
            inputs[self.brain.sensor_neurons] = (
                [pyrosim.Get_Touch_Sensor_Value_For_Link(self.sensors[i]) for i in self.brain.sensor_neurons]
            )
            k1 = self.f(self.brain_state, inputs)
            k2 = self.f(self.brain_state + (c.DELTA_TIME / 2) * k1, inputs)
            k3 = self.f(self.brain_state + (c.DELTA_TIME / 2) * k2, inputs)
            k4 = self.f(self.brain_state + c.DELTA_TIME * k3, inputs)
            self.brain_state = self.brain_state + (c.DELTA_TIME / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def act(self):
        if not self.is_ctrnn:
            for neuronName in self.nn.Get_Neuron_Names():
                if self.nn.Is_Motor_Neuron(neuronName):
                    joint_name = self.nn.Get_Motor_Neuron_Joint(neuronName).encode('utf-8')
                    desired_angle = (
                        max(c.LOWER_JOINT_LIMIT, min(self.nn.Get_Value_Of(neuronName), c.UPPER_JOINT_LIMIT))
                    )
                    pyrosim.Set_Motor_For_Joint(
                        bodyIndex=self.id,
                        jointName=joint_name,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=desired_angle,
                        maxForce=50.0
                    )
        else:
            for i, jointName in enumerate(self.motors):
                desired_angle = (
                    max(c.LOWER_JOINT_LIMIT, min(self.brain_state[self.brain.motor_neurons[i]], c.UPPER_JOINT_LIMIT))
                )
                pyrosim.Set_Motor_For_Joint(
                    bodyIndex=self.id,
                    jointName=jointName,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=desired_angle,
                    maxForce=50.0
                )

    def write_fitness(self):
        base_position_and_orientation = p.getBasePositionAndOrientation(self.id)
        base_position = base_position_and_orientation[0]
        with open(f"fitness{self.fitness_id}.txt", "w") as f:
            f.write(str(base_position[0] * -1))

    def f(self, y, inputs):
        return -y + self.brain.weights @ self.brain.vfunc(y + self.brain.bias) + inputs

