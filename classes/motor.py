import numpy
from pyrosim import pyrosim
import pybullet as p
#personal

class MOTOR:
    def __init__(self, name):
        #feilds
        self.jointName = name
        self.amplitude = None
        self.frequency = None
        self.phase = None
        self.motorValues = None

        self.Prepare_To_Act()


    def Prepare_To_Act(self):
        self.amplitude = numpy.pi / 4
        if self.jointName == b"Torso_BackLeg":
            self.frequency = 8
        else:
            self.frequency = 4
        self.phase = numpy.pi / 4
        self.motorValues = self.amplitude * numpy.sin(self.frequency * numpy.linspace(0, 2 * numpy.pi, 1000) + self.phase)

    def Set_Value(self, t, robotId):
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robotId,
            jointName=self.jointName,
            controlMode=p.POSITION_CONTROL,
            targetPosition=self.motorValues[t],
            maxForce=50
        )

    def Save_Values(self):
        numpy.save(f"data/{self.jointName}Values.npy", self.motorValues)

