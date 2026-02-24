from pyrosim import pyrosim
import pybullet as p
import pybullet_data
import numpy
import time

import generate

#generate
generate.Create_World()
generate.Create_Robot()

#set up connect
physicsClient = p.connect(p.GUI)

#sim configs
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("body.urdf")
p.loadSDF("world.sdf")
pyrosim.Prepare_To_Simulate(robotId)

#vars
backLegSensorValues = numpy.zeros(1000)
frontLegSensorValues = numpy.zeros(1000)

amplitudeB = numpy.pi / 6
frequencyB = 4
phaseOffsetB = numpy.pi / 2
targetAnglesB = amplitudeB * numpy.sin(frequencyB * numpy.linspace(0, 2 * numpy.pi, 1000) + phaseOffsetB )

amplitudeF = numpy.pi / 4
frequencyF = 4
phaseOffsetF = 0
targetAnglesF = amplitudeF * numpy.sin(frequencyF * numpy.linspace(0, 2 * numpy.pi, 1000) + phaseOffsetF )

for i in range(1000):
    p.stepSimulation()
    backLegSensorValues[i] = pyrosim.Get_Touch_Sensor_Value_For_Link("BackLeg")
    frontLegSensorValues[i] = pyrosim.Get_Touch_Sensor_Value_For_Link("FrontLeg")
    pyrosim.Set_Motor_For_Joint(
        bodyIndex=robotId,
        jointName=b"Torso_BackLeg",
        controlMode=p.POSITION_CONTROL,
        targetPosition=targetAnglesB[i],
        maxForce=50
    )
    pyrosim.Set_Motor_For_Joint(
        bodyIndex=robotId,
        jointName=b"Torso_FrontLeg",
        controlMode=p.POSITION_CONTROL,
        targetPosition=targetAnglesF[i],
        maxForce=50
    )
    time.sleep(1/100)
p.disconnect()

numpy.save("data/backLegSensorValues.npy", backLegSensorValues)
numpy.save("data/frontLegSensorValues.npy", frontLegSensorValues)
numpy.save("data/targetAnglesB.npy", targetAnglesB)
numpy.save("data/targetAnglesF.npy", targetAnglesF)


