import generate
import pybullet as p
import pybullet_data
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

for i in range(3000):
    p.stepSimulation()
    time.sleep(1/100)
p.disconnect()


