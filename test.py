import numpy as np
from simulation import Simulation

class TEST_DESIGN:
    def __init__(self):
        self.id = 2
        self.isCTRN = True
        self.n = 10

        #brain
        self.timeConstant = 1
        self.sensorNeurons = [0, 1]
        self.motorNeurons = [8, 9]
        self.weights = (np.random.rand(self.n, self.n) * 2) -1
        self.bias = (np.random.rand(self.n) * 2) - 1
        def activation_function(x):
            return np.tanh(x)
        self.vfunc = np.vectorize(activation_function)


if __name__ == "__main__":
    #solutionID = sys.argv[2]

    design = TEST_DESIGN()
    sim = Simulation("GUI", design)
    sim.run()