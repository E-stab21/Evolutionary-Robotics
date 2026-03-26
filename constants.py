import numpy

numberOfGenerations = 10

amplitudeB = numpy.pi / 6
frequencyB = 4
phaseOffsetB = numpy.pi / 2
targetAnglesB = amplitudeB * numpy.sin(frequencyB * numpy.linspace(0, 2 * numpy.pi, 1000) + phaseOffsetB )

amplitudeF = numpy.pi / 4
frequencyF = 4
phaseOffsetF = 0
targetAnglesF = amplitudeF * numpy.sin(frequencyF * numpy.linspace(0, 2 * numpy.pi, 1000) + phaseOffsetF )