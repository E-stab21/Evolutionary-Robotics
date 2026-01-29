import pyrosim.pyrosim as pyrosim

def generate():
    pyrosim.Start_SDF("box.sdf")
    for k in range(5):
        for j in range(5):
            for i in range(10):
                pyrosim.Send_Cube(name="Box", pos=[k,j,0.5 + i], size=[1 * (0.9 ** i),1 * (0.9 ** i),1 * (0.9 ** i)])
    pyrosim.Send_Cube(name="Box", pos=[0,-5,1.5], size=[1,2,3])
    pyrosim.End()

