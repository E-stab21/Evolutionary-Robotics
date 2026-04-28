from pyrosim import pyrosim
import random

def Create_World():
    pyrosim.Start_SDF("../dfs/world.sdf")
    pyrosim.Send_Cube(name="Box", pos=[5,0,0.5], size=[1,1,1])
    pyrosim.End()

def Create_Robot():
    Generate_Body()
    Generate_Brain()

def Generate_Body():
    pyrosim.Start_URDF("dfs/body.urdf")
    pyrosim.Send_Cube(name="Torso", pos=[0, 0, 1.5], size=[1, 1, 1])
    pyrosim.Send_Joint(name="Torso_FrontLeg", parent="Torso", child="FrontLeg", type="revolute", position=[0.5, 0, 1])
    pyrosim.Send_Cube(name="FrontLeg", pos=[0.5, 0, -0.5], size=[1, 1, 1])
    pyrosim.Send_Joint(name="Torso_BackLeg", parent="Torso", child="BackLeg", type="revolute", position=[-0.5, 0, 1])
    pyrosim.Send_Cube(name="BackLeg", pos=[-0.5, 0, -0.5], size=[1, 1, 1])
    pyrosim.End()

def Generate_Brain():
    pyrosim.Start_NeuralNetwork("brain.nndf")
    pyrosim.Send_Sensor_Neuron(name=0, linkName="Torso")
    pyrosim.Send_Sensor_Neuron(name=1, linkName="BackLeg")
    pyrosim.Send_Sensor_Neuron(name=2, linkName="FrontLeg")
    pyrosim.Send_Motor_Neuron(name=3, jointName="Torso_BackLeg")
    pyrosim.Send_Motor_Neuron(name=4, jointName="Torso_FrontLeg")
    for sensor_name in range(4):
        for motor_name in range(3, 5):
            pyrosim.Send_Synapse(sourceNeuronName=sensor_name, targetNeuronName=motor_name, weight= (random.random() - 0.5) * 2 )

    pyrosim.End()
