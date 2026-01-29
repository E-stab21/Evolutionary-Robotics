import generate
import pybullet as p
import pybullet_data
import time

#generate
generate.generate()

#set up connect
physicsClient = p.connect(p.GUI)

#sim configs
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
p.loadSDF("box.sdf")

for i in range(3000):
    p.stepSimulation()
    time.sleep(1/100)
p.disconnect()


