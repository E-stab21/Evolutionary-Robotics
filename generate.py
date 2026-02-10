from pyrosim import pyrosim


def Create_World():
    pyrosim.Start_SDF("world.sdf")
    pyrosim.Send_Cube(name="Box", pos=[5,0,0.5], size=[1,1,1])
    pyrosim.End()

def Create_Robot():
    pyrosim.Start_URDF("body.urdf")
    pyrosim.Send_Cube(name="Torso", pos=[0,0,1.5], size=[1,1,1])
    pyrosim.Send_Joint(name="Torso_Front", parent="Torso", child="Front", type="revolute", position=[0.5, 0, 1])
    pyrosim.Send_Cube(name="Front", pos=[0.5, 0, -0.5], size=[1, 1, 1])
    pyrosim.Send_Joint(name="Torso_Back", parent="Torso", child="Back", type="revolute", position=[-0.5, 0, 1])
    pyrosim.Send_Cube(name="Back", pos=[-0.5, 0, -0.5], size=[1, 1, 1])
    pyrosim.End()

