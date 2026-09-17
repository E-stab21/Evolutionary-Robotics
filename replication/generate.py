"""
Script that generates agent's and connector's
"""

# imports
import os
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
def generate_one_agent(agent_id, directory="agents"):
    os.makedirs(directory, exist_ok=True)
    vertices = [(random.randint(0, MAX_X), random.randint(0, MAX_Y))]
    edges = []
    chosen = []

    for _ in range(NUM_OF_AGENT_VERTICES - 1):
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

    for _ in range(NUM_OF_AGENT_EDGES - len(edges)):
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

    for _ in range(NUM_OF_AGENT_ACTIVE_EDGES):
        check = True
        while check:
            rand_index = random.randint(0, len(edges) - 1)
            if rand_index not in chosen:
                chosen.append(rand_index)
                edges[rand_index][2] = 1
                check = False

    weights1 = np.random.rand(AGENT_HIDDEN_LAYER_SIZE, AGENT_INPUT_SIZE).astype(
        np.float32
    )
    weights2 = np.random.rand(
        AGENT_HIDDEN_LAYER_SIZE, AGENT_HIDDEN_LAYER_SIZE
    ).astype(np.float32)
    weights3 = np.random.rand(
        NUM_OF_AGENT_ACTIVE_EDGES, AGENT_HIDDEN_LAYER_SIZE
    ).astype(np.float32)

    path = f"{directory}/agent{agent_id}.npz"
    np.savez(
        path,
        points=np.array(vertices),
        springs=edges,
        status="untrained",
        fitness=0.0,
        weights1=weights1,
        weights2=weights2,
        weights3=weights3,
    )
    return path


def generate_agents(num_of_agents, directory="agents"):
    for agent_id in range(num_of_agents):
        generate_one_agent(agent_id, directory=directory)


# Fixed grid-region targets for connector anchors (same areas on every agent).
REGION_TARGETS = [(0, 0), (1, 2), (3, 2), (5, 2), (6, 4)]


def region_anchors(points):
    anchors = []
    used = set()
    for tx, ty in REGION_TARGETS:
        ranked = sorted(
            range(len(points)),
            key=lambda i: (points[i][0] - tx) ** 2 + (points[i][1] - ty) ** 2,
        )
        pick = next((i for i in ranked if i not in used), ranked[0])
        anchors.append(pick)
        used.add(pick)
    return anchors


# connector generation
def generate_connectors(num_per_set=10, directory="connectors"):
    os.makedirs(directory, exist_ok=True)
    connector_id = 0
    for strength in ("weak", "strong"):
        for pairing in ("uniform", "diverse"):
            for _ in range(num_per_set):
                weights1 = np.random.rand(
                    CONNECTOR_HIDDEN_LAYER_SIZE, CONNECTOR_INPUT_SIZE
                )
                weights2 = np.random.rand(
                    NUM_OF_CONNECTOR_EDGES, CONNECTOR_HIDDEN_LAYER_SIZE
                )
                np.savez(
                    f"{directory}/connector{connector_id}.npz",
                    connector_type=pairing,
                    strength=strength,
                    weights1=weights1,
                    weights2=weights2,
                )
                connector_id += 1
    return connector_id
