from pyrosim import pyrosim
from classes.design import Ctrnn, TrdNet

def create_world():
    pyrosim.Start_SDF("dfs/world.sdf")
    pyrosim.Send_Cube(name="Box", pos=[5, 0, 0.5], size=[1, 1, 1])
    pyrosim.End()

if __name__ == "__main__":
    create_world()
    ctrnn = Ctrnn(0)
    ctrnn.simulate(direct_or_gui="GUI")
    trd = TrdNet(1)
    trd.simulate(direct_or_gui="GUI")
    trd.wait()

