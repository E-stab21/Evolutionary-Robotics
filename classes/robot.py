from pyrosim import pyrosim
import pybullet as p
#personal
import generate
from classes.sensor import SENSOR
from classes.motor import MOTOR

class ROBOT:
    def __init__(self):
        #feilds
        self.motors = {}
        self.sensors = {}
        self.id = None

        generate.Create_Robot()
        self.id = p.loadURDF("../body.urdf")
        pyrosim.Prepare_To_Simulate(self.id)
        self.Prepare_To_Sense()
        self.Prepare_To_Act()

    def Prepare_To_Sense(self):
        for linkName in pyrosim.linkNamesToIndices:
            self.sensors[linkName] = SENSOR(linkName)

    def Sense(self, t):
        for sensor in self.sensors.values():
            sensor.Get_Value(t)

    def Prepare_To_Act(self):
        for jointName in pyrosim.jointNamesToIndices:
            self.motors[jointName] = MOTOR(jointName)

    def Act(self, t):
        for motor in self.motors.values():
            motor.Set_Value(t, self.id)




