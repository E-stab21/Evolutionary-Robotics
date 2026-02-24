import numpy
from matplotlib import pyplot

backLegSensorValues = numpy.load("data/backLegSensorValues.npy", allow_pickle=True)
frontLegSensorValues = numpy.load("data/frontLegSensorValues.npy", allow_pickle=True)
targetAnglesB = numpy.load("data/targetAnglesB.npy", allow_pickle=True)
targetAnglesF = numpy.load("data/targetAnglesF.npy", allow_pickle=True)
pyplot.plot(targetAnglesB, linewidth=3)
pyplot.plot(targetAnglesF, linewidth=1)
#pyplot.plot(backLegSensorValues, linewidth=3.5)
#pyplot.plot(frontLegSensorValues, linewidth=2)
pyplot.legend(["BackMotor", "FrontMotor"])
pyplot.show()




