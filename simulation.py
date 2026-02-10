import generate
import pybullet as p
import pybullet_data
from pyrosim import pyrosim
import time

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

for i in range(3000):
    p.stepSimulation()
    backLegTouch = pyrosim.Get_Touch_Sensor_Value_For_Link("BackLeg")
    print(backLegTouch)
    time.sleep(1/100)
p.disconnect()


