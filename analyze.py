import numpy
import matplotlib.pyplot
from matplotlib import pyplot

backLegSensorValues = numpy.load("data/backLegSensorValues.npy", allow_pickle=True)
frontLegSensorValues = numpy.load("data/frontLegSensorValues.npy", allow_pickle=True)
pyplot.plot(backLegSensorValues, linewidth=3.5)
pyplot.plot(frontLegSensorValues, linewidth=2)
pyplot.legend(["BackLeg", "FrontLeg"])
pyplot.show()




