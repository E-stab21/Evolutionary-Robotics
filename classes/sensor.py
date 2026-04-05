import numpy
from pyrosim import pyrosim
#personal

class SENSOR:
    def __init__(self, name):
        self.linkName = name
        self.sensorValues = numpy.zeros(1000)

    def Get_Value(self, t):
        self.sensorValues[t] = pyrosim.Get_Touch_Sensor_Value_For_Link(self.linkName)

    def Save_Values(self):
        numpy.save(f"data/{self.linkName}Values.npy", self.sensorValues)