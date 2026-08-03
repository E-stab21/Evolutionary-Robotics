"""
Tool to generate agent connectors
"""

# imports
import random
import numpy as np


# globals
NUM_OF_CONNECTORS = 1
NUM_OF_ACTS = 5
NUM_OF_POINTS = 11
MAX_X = 7
MAX_Y = 4
INPUT_SIZE = 12
HIDDEN_LAYER_SIZE = 20


if __name__ == "__main__":
    for id in range(NUM_OF_CONNECTORS):
        # locals
        positions = []
        weights1 = np.random.rand(HIDDEN_LAYER_SIZE, INPUT_SIZE)
        weights2 = np.random.rand(NUM_OF_ACTS, HIDDEN_LAYER_SIZE)

        # create endpoints
        for i in range(NUM_OF_ACTS):
            positions.append(
                [
                    [random.randint(0, MAX_X), random.randint(0, MAX_Y)],
                    [random.randint(0, MAX_X), random.randint(0, MAX_Y)],
                ]
            )

        positions = np.array(positions)

        # saving to file
        np.savez(
            f"connectors/connector{id}.npz",
            pos=positions,
            weights1=weights1,
            weights2=weights2,
        )
