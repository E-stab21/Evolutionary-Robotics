"""
Script that generates agent's and connector's
"""

# imports
import random
import numpy as np


# globals
MAX_X = 6
MAX_Y = 4

NUM_OF_AGENT_VERTICES = 11
NUM_OF_AGENT_EDGES = 21
NUM_OF_AGENT_ACTIVE_EDGES = 11
AGENT_INPUT_SIZE = 54
AGENT_HIDDEN_LAYER_SIZE = 32

NUM_OF_CONNECTOR_EDGES = 5
CONNECTOR_INPUT_SIZE = 12
CONNECTOR_HIDDEN_LAYER_SIZE = 20


# agent generation
def generate_agents(num_of_agents):
    for id in range(num_of_agents):
        vertices = [(random.randint(0, MAX_X), random.randint(0, MAX_Y))]
        edges = []
        chosen = []

        # vertex generation
        for _ in range(NUM_OF_AGENT_VERTICES):
            check = True
            while check:
                possible = []
                rand_index = random.randint(0, len(vertices) - 1)
                x, y = vertices[rand_index]
                for x_ in [-1, 1]:
                    if not ((x + x_, y) in vertices or x + x_ > MAX_X or x + x_ < 0):
                        possible.append((x + x_, y))
                for y_ in [-1, 1]:
                    if not ((x, y + y_) in vertices or y + y_ > MAX_Y or y + y_ < 0):
                        possible.append((x, y + y_))
                if possible:
                    rand_vertex = random.choice(possible)
                    edges.append([rand_index, len(vertices), 0])
                    vertices.append(rand_vertex)
                    check = False

        # edge generation
        for _ in range(NUM_OF_AGENT_EDGES - NUM_OF_AGENT_VERTICES + 1):
            check = True
            while check:
                rand_edge = [
                    random.randint(0, len(vertices) - 1),
                    random.randint(0, len(vertices) - 1),
                    0,
                ]
                if rand_edge[0] != rand_edge[1] and rand_edge not in edges:
                    edges.append(rand_edge)
                    check = False

        # choosing active or passive edges
        for _ in range(NUM_OF_AGENT_ACTIVE_EDGES):
            check = True
            while check:
                rand_index = random.randint(0, len(edges) - 1)
                if rand_index not in chosen:
                    chosen.append(rand_index)
                    edges[rand_index][2] = 1
                    check = False

        # brain generation
        weights1 = np.random.rand(AGENT_HIDDEN_LAYER_SIZE, AGENT_INPUT_SIZE)
        weights2 = np.random.rand(AGENT_HIDDEN_LAYER_SIZE, AGENT_HIDDEN_LAYER_SIZE)
        weights3 = np.random.rand(NUM_OF_AGENT_ACTIVE_EDGES, AGENT_HIDDEN_LAYER_SIZE)

        vertices = np.array(vertices)

        np.savez(
            f"agents/agent{id}.npz",
            points=vertices,
            springs=edges,
            status="untrained",
            weights1=weights1,
            weights2=weights2,
            weights3=weights3,
        )


# connector generation
def generate_connectors(num_of_connectors, connector_type):
    if connector_type != "uniform" and connector_type != "diverse":
        raise Exception("Incorrect connector type")

    for id in range(num_of_connectors):
        # locals
        vertices = []
        weights1 = np.random.rand(CONNECTOR_HIDDEN_LAYER_SIZE, CONNECTOR_INPUT_SIZE)
        weights2 = np.random.rand(NUM_OF_CONNECTOR_EDGES, CONNECTOR_HIDDEN_LAYER_SIZE)

        # create endpoints
        for _ in range(NUM_OF_CONNECTOR_EDGES):
            vertices.append(
                [
                    [random.randint(0, MAX_X), random.randint(0, MAX_Y)],
                    [random.randint(0, MAX_X), random.randint(0, MAX_Y)],
                ]
            )

        positions = np.array(vertices)

        # saving to file
        np.savez(
            f"connectors/connector{id}.npz",
            pos=positions,
            connector_type=connector_type,
            weights1=weights1,
            weights2=weights2,
        )
