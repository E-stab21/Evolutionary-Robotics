import numpy as np
from src.simulation import Simulation

class TestDesign:
    def __init__(self, is_euler):
        self.id = 2
        self.is_ctrnn = True
        self.is_euler = is_euler
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
    euler = TestDesign(True)
    rk4 = TestDesign(False)
    sim = Simulation("DIRECT", "dfs/quadruped.urdf", "")
    sim.run()