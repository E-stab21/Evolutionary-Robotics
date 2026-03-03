import numpy
from pyrosim import pyrosim
import pybullet as p
#personal

class MOTOR:
    def __init__(self, name):
        #feilds
        self.jointName = name

    def Set_Value(self, desiredAngle, robotId):
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robotId,
            jointName=self.jointName,
            controlMode=p.POSITION_CONTROL,
            targetPosition=desiredAngle,
            maxForce=50
        )

