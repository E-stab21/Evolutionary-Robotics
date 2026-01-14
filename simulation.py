import pybullet as p
import time

#set up connect
physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

for i in range(1000):
    p.stepSimulation()

    time.sleep(1/60)
    print(i)


p.disconnect()


