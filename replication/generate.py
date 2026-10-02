"""
Script that generates agent's and connector's
"""

# imports
import os
import random
import numpy as np
import runtime as rt


# globals
MAX_X = 6  # 7 columns
MAX_Y = 3  # 4 rows -- matches the paper's "4 rows x 7 columns" grid exactly.
# MAX_Y=4 (5 rows) let bodies come out taller than wide by pure random-walk
# luck (~41% of a sample pool) -- those consistently jumped/toppled, since a
# tall, narrow structure is far less stable than a short, wide one.

NUM_OF_AGENT_VERTICES = 11
NUM_OF_AGENT_EDGES = 21
NUM_OF_AGENT_ACTIVE_EDGES = 11
AGENT_INPUT_SIZE = 54
AGENT_HIDDEN_LAYER_SIZE = 32

NUM_OF_CONNECTOR_EDGES = 5
CONNECTOR_INPUT_SIZE = 11  # per-spring: 4 relative positions + 4 velocities + 2 touch + 1 length
CONNECTOR_HIDDEN_LAYER_SIZE = 32


def _init_weights(shape, fan_in):
    """Zero-mean, fan-in-scaled uniform init so tanh layers don't saturate at birth."""
    bound = 1.0 / np.sqrt(fan_in)
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


MAX_LAYOUT_ATTEMPTS = 500


def _neighbor_pairs(vertices):
    """Index pairs whose grid points are adjacent, including diagonals -- the
    only edges a real leg/segment should have, and enough to triangulate a
    square cell so it can't just shear."""
    pairs = []
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        for j in range(i + 1, len(vertices)):
            xj, yj = vertices[j]
            if max(abs(xi - xj), abs(yi - yj)) == 1:
                pairs.append((i, j))
    return pairs


def _generate_layout():
    """A connected random-walk body plus every local edge needed to fill out
    NUM_OF_AGENT_EDGES, all between grid-adjacent nodes -- no springs
    shortcutting across the body. Returns None if this layout is too sparse
    to have enough local edges."""
    vertices = [(random.randint(0, MAX_X), random.randint(0, MAX_Y))]
    edges = []

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

    existing = {(a, b) if a < b else (b, a) for a, b, _ in edges}
    candidates = [pair for pair in _neighbor_pairs(vertices) if pair not in existing]
    needed = NUM_OF_AGENT_EDGES - len(edges)
    if len(candidates) < needed:
        return None

    random.shuffle(candidates)
    for i, j in candidates[:needed]:
        edges.append([i, j, 0])
    return vertices, edges


# agent generation
def generate_one_agent(agent_id, directory="agents"):
    directory = rt.resolve_path(directory)
    os.makedirs(directory, exist_ok=True)
    chosen = []

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        layout = _generate_layout()
        if layout is not None:
            break
    else:
        raise RuntimeError(
            f"Could not lay out a locally-connected {NUM_OF_AGENT_VERTICES}-node, "
            f"{NUM_OF_AGENT_EDGES}-edge body after {MAX_LAYOUT_ATTEMPTS} attempts"
        )
    vertices, edges = layout

    for _ in range(NUM_OF_AGENT_ACTIVE_EDGES):
        check = True
        while check:
            rand_index = random.randint(0, len(edges) - 1)
            if rand_index not in chosen:
                chosen.append(rand_index)
                edges[rand_index][2] = 1
                check = False

    weights1 = _init_weights(
        (AGENT_HIDDEN_LAYER_SIZE, AGENT_INPUT_SIZE), AGENT_INPUT_SIZE
    )
    weights2 = _init_weights(
        (AGENT_HIDDEN_LAYER_SIZE, AGENT_HIDDEN_LAYER_SIZE), AGENT_HIDDEN_LAYER_SIZE
    )
    weights3 = _init_weights(
        (NUM_OF_AGENT_ACTIVE_EDGES, AGENT_HIDDEN_LAYER_SIZE), AGENT_HIDDEN_LAYER_SIZE
    )

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
    directory = rt.resolve_path(directory)
    os.makedirs(directory, exist_ok=True)
    connector_id = 0
    for strength in ("weak", "strong"):
        for pairing in ("uniform", "diverse"):
            for _ in range(num_per_set):
                weights1 = _init_weights(
                    (CONNECTOR_HIDDEN_LAYER_SIZE, CONNECTOR_INPUT_SIZE),
                    CONNECTOR_INPUT_SIZE,
                )
                weights2 = _init_weights(
                    (CONNECTOR_HIDDEN_LAYER_SIZE, CONNECTOR_HIDDEN_LAYER_SIZE),
                    CONNECTOR_HIDDEN_LAYER_SIZE,
                )
                weights3 = _init_weights(
                    (1, CONNECTOR_HIDDEN_LAYER_SIZE),
                    CONNECTOR_HIDDEN_LAYER_SIZE,
                )
                np.savez(
                    f"{directory}/connector{connector_id}.npz",
                    connector_type=pairing,
                    strength=strength,
                    weights1=weights1,
                    weights2=weights2,
                    weights3=weights3,
                )
                connector_id += 1
    return connector_id
