"""
script that generates all 100 agents body's
and brain's
"""

# imports
import random
import numpy as np


# globals
NUM_OF_AGENTS = 1
MAX_X = 7
MAX_Y = 4
NUM_OF_POINTS = 11
NUM_OF_SPRINGS = 21
NUM_OF_ACTIVE = 11
INPUT_SIZE = 54
HIDDEN_LAYER_SIZE = 32


# generation
for id in range(NUM_OF_AGENTS):
    points = [(random.randint(1, MAX_X), random.randint(1, MAX_Y))]
    springs = []

    # point generation
    for _ in range(NUM_OF_POINTS):
        check = True
        while check:
            possible = []
            rand_index = random.randint(0, len(points) - 1)
            x, y = points[rand_index]
            for x_ in [-1, 1]:
                if not ((x + x_, y) in points or x_ > MAX_X or x_ < 1):
                    possible.append((x + x_, y))
            for y_ in [-1, 1]:
                if not ((x, y + y_) in points or y_ > MAX_Y or y_ < 1):
                    possible.append((x, y + y_))
            if possible:
                rand_point = random.choice(possible)
                springs.append([rand_index, len(points), 0])
                points.append(rand_point)
                check = False

    # spring generation
    for _ in range(NUM_OF_SPRINGS - NUM_OF_POINTS + 1):
        check = True
        while check:
            rand_spring = [
                random.randint(0, len(points) - 1),
                random.randint(0, len(points) - 1),
                0,
            ]
            if rand_spring not in springs:
                springs.append(rand_spring)
                check = False

    # choosing active or passive springs
    chosen = []
    for _ in range(NUM_OF_ACTIVE):
        check = True
        while check:
            rand_index = random.randint(0, len(springs) - 1)
            if rand_index not in chosen:
                chosen.append(rand_index)
                springs[rand_index][2] = 1
                check = False

    # brain generation
    weights1 = np.random.rand(HIDDEN_LAYER_SIZE, INPUT_SIZE)
    weights2 = np.random.rand(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
    weights3 = np.random.rand(NUM_OF_ACTIVE, HIDDEN_LAYER_SIZE)

    # file generation
    with open(
        f"/home/ethan/Projects/Evolutionary-Robotics/src/taichi/replication/agents/agent{id}.csv",
        "w",
    ) as f:
        f.write("Points: \n")
        for point in points:
            f.write(f"{point[0]}, {point[1]}\n")

        f.write("Springs:\n")
        for spring in springs:
            f.write(f"{spring[0]}, {spring[1]}, {spring[2]}\n")

        f.write("Matrix 1:\n")
        for row in weights1:
            f.write(f"{row[0]}")
            for num in row[1:]:
                f.write(f",{num}")
            f.write("\n")

        f.write("Matrix 2:\n")
        for row in weights2:
            f.write(f"{row[0]}")
            for num in row[1:]:
                f.write(f",{num}")
            f.write("\n")

        f.write("Matrix 3:\n")
        for row in weights3:
            f.write(f"{row[0]}")
            for num in row[1:]:
                f.write(f",{num}")
            f.write("\n")
