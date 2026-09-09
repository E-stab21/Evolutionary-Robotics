"""
Simulation constants and kernels
"""

# imports
import secrets
import numpy as np
import taichi as ti
from generate import (
    NUM_OF_AGENT_VERTICES,
    NUM_OF_AGENT_EDGES,
    NUM_OF_AGENT_ACTIVE_EDGES,
    AGENT_INPUT_SIZE,
    AGENT_HIDDEN_LAYER_SIZE,
    NUM_OF_CONNECTOR_EDGES,
    CONNECTOR_INPUT_SIZE,
    CONNECTOR_HIDDEN_LAYER_SIZE,
)


# globals
SEED = secrets.randbits(31)
ti.init(arch=ti.gpu, random_seed=464525965, debug=False, unrolling_limit=0)

# constants
TIME_STEPS = 1000
DT = 0.005
GRAVITY = ti.Vector([0, -9.8])
SPRING_K = 1500.0
SPRING_DAMPING = 7.0
MOTOR_FORCE = 100
START_MARGIN = 0.05

GROUND_K = 8000.0
GROUND_DAMPING = 150.0
GROUND_MARGIN = 0.02
FRICTION_SMOOTHING = 0.05
FRICTION_MU = 0.99

SENSOR_SMOOTHING = 0.01
CPG_FREQUENCY = 20.0

LR = 0.01
GRAD_CLIP = 10.0
GENERATIONS = 30

SCALE = 10.0


# kernels
@ti.kernel
def compute_spring_forces(
    t: int,
    vertices: ti.template(),
    edges: ti.template(),
    velocities: ti.template(),
    forces: ti.template(),
    resting_lengths: ti.template(),
):
    # finding spring forces
    for i in range(NUM_OF_AGENT_EDGES):
        a, b = edges[i]
        delta = vertices[t, b] - vertices[t, a]
        current_length = delta.norm()
        spring_force = SPRING_K * (current_length - resting_lengths[i])
        relative_velocity = velocities[t, b] - velocities[t, a]
        damping_force = SPRING_DAMPING * relative_velocity.dot(delta.normalized())
        force = delta.normalized() * (spring_force + damping_force)
        forces[t, a] += force
        forces[t, b] -= force


@ti.kernel
def compute_ground_forces(
    t: int,
    vertices: ti.template(),
    velocities: ti.template(),
    forces: ti.template(),
):
    for i in range(NUM_OF_AGENT_VERTICES):
        if vertices[t, i][1] < GROUND_MARGIN:
            # normal forces
            vely = velocities[t, i][1]
            penetration = GROUND_MARGIN - vertices[t, i][1]
            normal_force = penetration * GROUND_K
            if vely < 0.0:
                normal_force += -vely * GROUND_DAMPING
            forces[t, i][1] += normal_force

            # friction forces
            velx = velocities[t, i][0]
            friction_dir = -ti.tanh(velx / FRICTION_SMOOTHING)
            friction_force = normal_force * friction_dir * FRICTION_MU
            forces[t, i][0] += friction_force


@ti.kernel
def compute_agent_motor_forces(
    t: int,
    vertices: ti.template(),
    edges: ti.template(),
    velocities: ti.template(),
    forces: ti.template(),
    center_of_mass: ti.template(),
    input_state: ti.template(),
    neural_state1: ti.template(),
    neural_state2: ti.template(),
    output_state: ti.template(),
    weights1: ti.template(),
    weights2: ti.template(),
    weights3: ti.template(),
    motor_indices: ti.template(),
):
    # for i in range(NUM_OF_AGENT_VERTICES):
    #     penetration = GROUND_MARGIN - vertices[t, sensor_indices[i]][1]
    #     brain_state[t, i] = 0.5 + 0.5 * ti.tanh(penetration / SENSOR_SMOOTHING)

    # center of mass
    for _ in range(1):
        center_of_mass[None] = ti.Vector([0.0, 0.0])
    for i in range(NUM_OF_AGENT_VERTICES):
        center_of_mass[None] += vertices[t, i]
    for _ in range(1):
        center_of_mass[None] /= NUM_OF_AGENT_VERTICES

    # setting vertice and velocitiy neurons
    for i in range(NUM_OF_AGENT_VERTICES):
        input_state[t, i] = vertices[t, i][0] - center_of_mass[None][0]
        input_state[t, i + NUM_OF_AGENT_VERTICES] = (
            vertices[t, i][1] - center_of_mass[None][1]
        )
        input_state[t, i + NUM_OF_AGENT_VERTICES * 2] = velocities[t, i][0]
        input_state[t, i + NUM_OF_AGENT_VERTICES * 3] = velocities[t, i][1]

    # setting cpg neurons
    for i in range(10):
        input_state[t, i + NUM_OF_AGENT_VERTICES * 4] = ti.sin(
            t * DT * CPG_FREQUENCY * i + i * 0.35
        )

    # forward pass 1
    for i in range(AGENT_HIDDEN_LAYER_SIZE):
        sum = 0.0
        for j in ti.static(range(AGENT_INPUT_SIZE)):
            sum += input_state[t, j] * weights1[i, j]
        neural_state1[t, i] = ti.tanh(sum)

    # forward pass 2
    for i in range(AGENT_HIDDEN_LAYER_SIZE):
        sum = 0.0
        for j in ti.static(range(AGENT_HIDDEN_LAYER_SIZE)):
            sum += neural_state1[t, j] * weights2[i, j]
        neural_state2[t, i] = ti.tanh(sum)

    # forward pass 3
    for i in range(NUM_OF_AGENT_ACTIVE_EDGES):
        sum = 0.0
        for j in ti.static(range(AGENT_HIDDEN_LAYER_SIZE)):
            sum += neural_state2[t, j] * weights3[i, j]
        output_state[t, i] = ti.tanh(sum)

    # converting motor values to forces
    for i in range(NUM_OF_AGENT_ACTIVE_EDGES):
        a, b = edges[motor_indices[i]]
        delta = vertices[t, b] - vertices[t, a]
        force = delta.normalized() * output_state[t, i] * MOTOR_FORCE
        forces[t, a] += force
        forces[t, b] -= force


@ti.kernel
def compute_connector_motor_forces(
    t: int,
    agent1_vertices: ti.template(),
    agent1_velocities: ti.template(),
    agent1_forces: ti.template(),
    agent2_vertices: ti.template(),
    agent2_velocities: ti.template(),
    agent2_forces: ti.template(),
    weights1: ti.template(),
    weights2: ti.template(),
):
    agent1_center = ti.Vector([0.0, 0.0])
    agent2_center = ti.Vector([0.0, 0.0])
    agent1_avg_velocity = ti.Vector([0.0, 0.0])
    agent2_avg_velocity = ti.Vector([0.0, 0.0])
    input_state = ti.Vector.zero(float, CONNECTOR_INPUT_SIZE)
    hidden_state = ti.Vector.zero(float, CONNECTOR_HIDDEN_LAYER_SIZE)
    output_state = ti.Vector.zero(float, NUM_OF_CONNECTOR_EDGES)

    for i in range(NUM_OF_AGENT_VERTICES):
        agent1_center += agent1_vertices[t, i]
        agent2_center += agent2_vertices[t, i]
        agent1_avg_velocity += agent1_velocities[t, i]
        agent2_avg_velocity += agent2_velocities[t, i]

    agent1_center /= NUM_OF_AGENT_VERTICES
    agent2_center /= NUM_OF_AGENT_VERTICES
    agent1_avg_velocity /= NUM_OF_AGENT_VERTICES
    agent2_avg_velocity /= NUM_OF_AGENT_VERTICES

    for i in range(2):
        input_state[i] = agent1_center[i]
        input_state[i + 2] = agent2_center[i]
        input_state[i + 4] = agent1_avg_velocity[i]
        input_state[i + 6] = agent2_avg_velocity[i]
        input_state[i + 8] = agent2_center[i] - agent1_center[i]
        input_state[i + 10] = agent2_avg_velocity[i] - agent1_avg_velocity[i]

    for i in ti.static(range(CONNECTOR_HIDDEN_LAYER_SIZE)):
        sum = 0.0
        for j in ti.static(range(CONNECTOR_INPUT_SIZE)):
            sum += input_state[j] * weights1[i, j]
        hidden_state[i] = ti.tanh(sum)

    for i in ti.static(range(NUM_OF_CONNECTOR_EDGES)):
        sum = 0.0
        for j in ti.static(range(CONNECTOR_HIDDEN_LAYER_SIZE)):
            sum += hidden_state[j] * weights2[i, j]
        output_state[i] = ti.tanh(sum)

    for i in ti.static(range(NUM_OF_CONNECTOR_EDGES)):
        delta = agent2_vertices[t, i] - agent1_vertices[t, i]
        force = delta.normalized() * output_state[i] * MOTOR_FORCE
        agent1_forces[t, i] += force
        agent2_forces[t, i] -= force


@ti.kernel
def apply_forces(
    t: int, vertices: ti.template(), velocities: ti.template(), forces: ti.template()
):
    for i in range(NUM_OF_AGENT_VERTICES):
        velocities[t + 1, i] = velocities[t, i] + (GRAVITY + forces[t, i]) * DT
        vertices[t + 1, i] = vertices[t, i] + velocities[t + 1, i] * DT


@ti.kernel
def compute_loss(
    agent1_vertices: ti.template(),
    agent2_vertices: ti.template(),
    loss: ti.template(),
):
    for _ in range(1):
        loss[None] = 0
    for i in range(NUM_OF_AGENT_VERTICES):
        loss[None] -= agent1_vertices[TIME_STEPS - 1, i][0]
        loss[None] -= agent2_vertices[TIME_STEPS - 1, i][0]
    for _ in range(1):
        loss[None] /= 2 * NUM_OF_AGENT_VERTICES
    # print(loss[None])


@ti.kernel
def update_agent_weights(
    weights1: ti.template(), weights2: ti.template(), weights3: ti.template()
):
    for i, j in weights1:
        grad = weights1.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights1[i, j] -= grad * LR

    for i, j in weights2:
        grad = weights2.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights2[i, j] -= grad * LR

    for i, j in weights3:
        grad = weights3.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights3[i, j] -= grad * LR
        # print(grad)


@ti.kernel
def update_connector_weights(weights1: ti.template(), weights2: ti.template()):
    for i, j in weights1:
        grad = weights1.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights1[i, j] -= grad * LR

    for i, j in weights2:
        grad = weights2.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights2[i, j] -= grad * LR


def set_agent(agent):
    for t, i in ti.ndrange(TIME_STEPS, NUM_OF_AGENT_VERTICES):
        agent.velocities[t, i] = ti.Vector([0.0, 0.0])
        agent.forces[t, i] = ti.Vector([0.0, 0.0])


class Agent:
    def __init__(self, agent_id):
        self.path = f"agents/agent{agent_id}.npz"
        loaded = np.load(self.path)
        self.points = loaded["points"].tolist()
        self.springs = loaded["springs"].tolist()
        self.active_springs = []

        self.velocities = ti.Vector.field(
            n=2,
            dtype=float,
            shape=(TIME_STEPS, NUM_OF_AGENT_VERTICES),
            needs_grad=True,
        )
        self.forces = ti.Vector.field(
            n=2,
            dtype=float,
            shape=(TIME_STEPS, NUM_OF_AGENT_VERTICES),
            needs_grad=True,
        )
        self.vertices = ti.Vector.field(
            n=2,
            dtype=float,
            shape=(TIME_STEPS, NUM_OF_AGENT_VERTICES),
            needs_grad=True,
        )
        self.edges = ti.Vector.field(n=2, dtype=int, shape=(NUM_OF_AGENT_EDGES,))
        self.resting_lengths = ti.field(dtype=float, shape=(NUM_OF_AGENT_EDGES,))
        self.motor_indices = ti.field(dtype=int, shape=(NUM_OF_AGENT_ACTIVE_EDGES,))
        self.center_of_mass = ti.Vector.field(
            n=2, dtype=float, shape=(), needs_grad=True
        )
        self.input_state = ti.field(
            dtype=float, shape=(TIME_STEPS, AGENT_INPUT_SIZE), needs_grad=True
        )
        self.neural_state1 = ti.field(
            dtype=float,
            shape=(TIME_STEPS, AGENT_HIDDEN_LAYER_SIZE),
            needs_grad=True,
        )
        self.neural_state2 = ti.field(
            dtype=float,
            shape=(TIME_STEPS, AGENT_HIDDEN_LAYER_SIZE),
            needs_grad=True,
        )
        self.output_state = ti.field(
            dtype=float,
            shape=(TIME_STEPS, NUM_OF_AGENT_ACTIVE_EDGES),
            needs_grad=True,
        )
        self.weights1 = ti.field(
            dtype=float,
            shape=(AGENT_HIDDEN_LAYER_SIZE, AGENT_INPUT_SIZE),
            needs_grad=True,
        )
        self.weights2 = ti.field(
            dtype=float,
            shape=(AGENT_HIDDEN_LAYER_SIZE, AGENT_HIDDEN_LAYER_SIZE),
            needs_grad=True,
        )
        self.weights3 = ti.field(
            dtype=float,
            shape=(NUM_OF_AGENT_ACTIVE_EDGES, AGENT_HIDDEN_LAYER_SIZE),
            needs_grad=True,
        )
        self.loss = ti.field(dtype=float, shape=(), needs_grad=True)

        self.weights1.from_numpy(loaded["weights1"])
        self.weights2.from_numpy(loaded["weights2"])
        self.weights3.from_numpy(loaded["weights3"])

        j = 0
        for i, spring in enumerate(self.springs):
            self.edges[i] = ti.Vector([spring[0], spring[1]])
            if spring[2] == 1:
                self.motor_indices[j] = i
                j += 1

        initial_points = []
        for x, y in self.points:
            if y == 1.0 or y == 3.0:
                initial_points.append((x + 0.5, y))
            else:
                initial_points.append((x, y))

        min_x = min(x for x, _ in initial_points)
        min_y = min(y for _, y in initial_points)

        for i, (x, y) in enumerate(initial_points):
            self.vertices[0, i] = (
                x - min_x + START_MARGIN,
                y - min_y + START_MARGIN,
            )

        for i in range(NUM_OF_AGENT_EDGES):
            a, b = self.edges[i]
            diff = self.vertices[0, a] - self.vertices[0, b]
            self.resting_lengths[i] = diff.norm()

    def write(self):
        np.savez(
            self.path,
            points=np.asarray(self.points),
            springs=np.asarray(self.springs),
            status="trained",
            weights1=self.weights1.to_numpy(),
            weights2=self.weights2.to_numpy(),
            weights3=self.weights3.to_numpy(),
        )


class Connector:
    def __init__(self, connector_id, agent1, agent2):
        self.path = f"connectors/connector{connector_id}.npz"
        loaded = np.load(self.path)
        self.type = loaded["connector_type"].item()
        self.agent1 = agent1
        self.agent2 = agent2
        self.weights1 = ti.field(
            dtype=float,
            shape=(CONNECTOR_HIDDEN_LAYER_SIZE, CONNECTOR_INPUT_SIZE),
            needs_grad=True,
        )
        self.weights2 = ti.field(
            dtype=float,
            shape=(NUM_OF_CONNECTOR_EDGES, CONNECTOR_HIDDEN_LAYER_SIZE),
            needs_grad=True,
        )

        self.weights1.from_numpy(loaded["weights1"])
        self.weights2.from_numpy(loaded["weights2"])

    def write(self):
        loaded = np.load(self.path)
        np.savez(
            self.path,
            connector_type=self.type,
            pos=loaded["pos"],
            weights1=self.weights1.to_numpy(),
            weights2=self.weights2.to_numpy(),
        )
