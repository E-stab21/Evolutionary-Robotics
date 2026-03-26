import pybullet as p
import pybullet_data
#personal
import generate

class WORLD:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadSDF("world.sdf")
        p.loadURDF("plane.urdf")