"""
Simulation constants and kernels
"""

import secrets
import numpy as np
import taichi as ti
import runtime as rt
from generate import (
    NUM_OF_AGENT_VERTICES,
    NUM_OF_AGENT_EDGES,
    NUM_OF_AGENT_ACTIVE_EDGES,
    AGENT_INPUT_SIZE,
    AGENT_HIDDEN_LAYER_SIZE,
    NUM_OF_CONNECTOR_EDGES,
    CONNECTOR_INPUT_SIZE,
    CONNECTOR_HIDDEN_LAYER_SIZE,
    region_anchors,
)


SEED = secrets.randbits(31)


def _init_taichi():
    common = dict(random_seed=464525965, debug=False, unrolling_limit=0)
    if rt.ARCH != "gpu":
        ti.init(arch=ti.cpu, **common)
        return "cpu"

    # Prefer Vulkan here (Intel Arc). CUDA only works with proprietary NVIDIA.
    errors = []
    for arch_name, arch in (("vulkan", ti.vulkan), ("cuda", ti.cuda)):
        try:
            kw = dict(arch=arch, **common)
            if arch_name == "cuda":
                kw["device_memory_fraction"] = rt.GPU_MEMORY_FRACTION
            ti.init(**kw)
            return arch_name
        except Exception as exc:
            errors.append(f"{arch_name}: {exc}")
    raise RuntimeError("No GPU backend available: " + " | ".join(errors))


_GPU_BACKEND = _init_taichi()
print(f"[runtime] taichi arch={rt.ARCH} backend={_GPU_BACKEND}", flush=True)

TIME_STEPS = rt.TIME_STEPS
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

# Paper-style actuation smoothing: o_t = α o_raw + (1-α) o_{t-1}
MOTOR_SMOOTH = 0.15
LR = 0.01
LR_END = 0.001
GRAD_CLIP = 10.0
GENERATIONS = rt.GENERATIONS

SCALE = 10.0

WEAK_CONNECTOR_K = 200.0
STRONG_CONNECTOR_K = 2000.0
CONNECTOR_DAMPING = 7.0

# Environment parameters (mutated by environments.set_env)
env_friction = ti.field(dtype=float, shape=())
env_ground_damping = ti.field(dtype=float, shape=())
env_gravity_scale = ti.field(dtype=float, shape=())
env_wind = ti.field(dtype=float, shape=())
env_slope = ti.field(dtype=float, shape=())
env_step_x = ti.field(dtype=float, shape=())
env_step_h = ti.field(dtype=float, shape=())
env_gap_start = ti.field(dtype=float, shape=())
env_gap_end = ti.field(dtype=float, shape=())
env_drag = ti.field(dtype=float, shape=())


def reset_flat_env():
    env_friction[None] = FRICTION_MU
    env_ground_damping[None] = GROUND_DAMPING
    env_gravity_scale[None] = 1.0
    env_wind[None] = 0.0
    env_slope[None] = 0.0
    env_step_x[None] = 1e9
    env_step_h[None] = 0.0
    env_gap_start[None] = 1e9
    env_gap_end[None] = 1e9
    env_drag[None] = 0.0


reset_flat_env()


@ti.kernel
def compute_spring_forces(
    t: int,
    vertices: ti.template(),
    edges: ti.template(),
    velocities: ti.template(),
    forces: ti.template(),
    resting_lengths: ti.template(),
):
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
        x = vertices[t, i][0]
        y = vertices[t, i][1]
        gap = 0
        if x >= env_gap_start[None] and x <= env_gap_end[None]:
            gap = 1
        gh = GROUND_MARGIN + env_slope[None] * x
        if x >= env_step_x[None]:
            gh += env_step_h[None]
        if gap == 0 and y < gh:
            vely = velocities[t, i][1]
            penetration = gh - y
            normal_force = penetration * GROUND_K
            if vely < 0.0:
                normal_force += -vely * env_ground_damping[None]
            forces[t, i][1] += normal_force

            velx = velocities[t, i][0]
            friction_dir = -ti.tanh(velx / FRICTION_SMOOTHING)
            friction_force = normal_force * friction_dir * env_friction[None]
            forces[t, i][0] += friction_force

        forces[t, i][0] += env_wind[None]
        forces[t, i][0] -= env_drag[None] * velocities[t, i][0]
        forces[t, i][1] -= env_drag[None] * velocities[t, i][1]


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
    for _ in range(1):
        center_of_mass[None] = ti.Vector([0.0, 0.0])
    for i in range(NUM_OF_AGENT_VERTICES):
        center_of_mass[None] += vertices[t, i]
    for _ in range(1):
        center_of_mass[None] /= NUM_OF_AGENT_VERTICES

    for i in range(NUM_OF_AGENT_VERTICES):
        input_state[t, i] = vertices[t, i][0] - center_of_mass[None][0]
        input_state[t, i + NUM_OF_AGENT_VERTICES] = (
            vertices[t, i][1] - center_of_mass[None][1]
        )
        input_state[t, i + NUM_OF_AGENT_VERTICES * 2] = velocities[t, i][0]
        input_state[t, i + NUM_OF_AGENT_VERTICES * 3] = velocities[t, i][1]

    for i in range(10):
        input_state[t, i + NUM_OF_AGENT_VERTICES * 4] = ti.sin(
            t * DT * CPG_FREQUENCY * i + i * 0.35
        )

    for i in range(AGENT_HIDDEN_LAYER_SIZE):
        sum = 0.0
        for j in ti.static(range(AGENT_INPUT_SIZE)):
            sum += input_state[t, j] * weights1[i, j]
        neural_state1[t, i] = ti.tanh(sum)

    for i in range(AGENT_HIDDEN_LAYER_SIZE):
        sum = 0.0
        for j in ti.static(range(AGENT_HIDDEN_LAYER_SIZE)):
            sum += neural_state1[t, j] * weights2[i, j]
        neural_state2[t, i] = ti.tanh(sum)

    for i in range(NUM_OF_AGENT_ACTIVE_EDGES):
        sum = 0.0
        for j in ti.static(range(AGENT_HIDDEN_LAYER_SIZE)):
            sum += neural_state2[t, j] * weights3[i, j]
        raw = ti.tanh(sum)
        if t == 0:
            output_state[t, i] = raw
        else:
            output_state[t, i] = (
                MOTOR_SMOOTH * raw + (1.0 - MOTOR_SMOOTH) * output_state[t - 1, i]
            )

    for i in range(NUM_OF_AGENT_ACTIVE_EDGES):
        a, b = edges[motor_indices[i]]
        delta = vertices[t, b] - vertices[t, a]
        force = delta.normalized() * output_state[t, i] * MOTOR_FORCE
        forces[t, a] += force
        forces[t, b] -= force


@ti.kernel
def compute_connector_nn(
    t: int,
    agent1_vertices: ti.template(),
    agent1_velocities: ti.template(),
    agent2_vertices: ti.template(),
    agent2_velocities: ti.template(),
    agent1_center: ti.template(),
    agent2_center: ti.template(),
    agent1_avg_velocity: ti.template(),
    agent2_avg_velocity: ti.template(),
    input_state: ti.template(),
    hidden_state: ti.template(),
    output_state: ti.template(),
    weights1: ti.template(),
    weights2: ti.template(),
):
    for _ in range(1):
        agent1_center[None] = ti.Vector([0.0, 0.0])
        agent2_center[None] = ti.Vector([0.0, 0.0])
        agent1_avg_velocity[None] = ti.Vector([0.0, 0.0])
        agent2_avg_velocity[None] = ti.Vector([0.0, 0.0])

    for i in range(NUM_OF_AGENT_VERTICES):
        agent1_center[None] += agent1_vertices[t, i]
        agent2_center[None] += agent2_vertices[t, i]
        agent1_avg_velocity[None] += agent1_velocities[t, i]
        agent2_avg_velocity[None] += agent2_velocities[t, i]

    for _ in range(1):
        agent1_center[None] /= NUM_OF_AGENT_VERTICES
        agent2_center[None] /= NUM_OF_AGENT_VERTICES
        agent1_avg_velocity[None] /= NUM_OF_AGENT_VERTICES
        agent2_avg_velocity[None] /= NUM_OF_AGENT_VERTICES

    for i in range(2):
        input_state[i] = agent1_center[None][i]
        input_state[i + 2] = agent2_center[None][i]
        input_state[i + 4] = agent1_avg_velocity[None][i]
        input_state[i + 6] = agent2_avg_velocity[None][i]
        input_state[i + 8] = agent2_center[None][i] - agent1_center[None][i]
        input_state[i + 10] = agent2_avg_velocity[None][i] - agent1_avg_velocity[None][i]

    for i in range(CONNECTOR_HIDDEN_LAYER_SIZE):
        sum = 0.0
        for j in ti.static(range(CONNECTOR_INPUT_SIZE)):
            sum += input_state[j] * weights1[i, j]
        hidden_state[i] = ti.tanh(sum)

    for i in range(NUM_OF_CONNECTOR_EDGES):
        sum = 0.0
        for j in ti.static(range(CONNECTOR_HIDDEN_LAYER_SIZE)):
            sum += hidden_state[j] * weights2[i, j]
        output_state[i] = ti.tanh(sum)


@ti.kernel
def apply_connector_forces(
    t: int,
    agent1_vertices: ti.template(),
    agent1_velocities: ti.template(),
    agent1_forces: ti.template(),
    agent2_vertices: ti.template(),
    agent2_velocities: ti.template(),
    agent2_forces: ti.template(),
    agent1_anchors: ti.template(),
    agent2_anchors: ti.template(),
    resting_lengths: ti.template(),
    output_state: ti.template(),
    connector_k: float,
):
    for i in range(NUM_OF_CONNECTOR_EDGES):
        a = agent1_anchors[i]
        b = agent2_anchors[i]
        delta = agent2_vertices[t, b] - agent1_vertices[t, a]
        length = delta.norm() + 1e-6
        direction = delta / length
        spring_force = connector_k * (length - resting_lengths[i])
        rel_vel = agent2_velocities[t, b] - agent1_velocities[t, a]
        damping_force = CONNECTOR_DAMPING * rel_vel.dot(direction)
        motor = output_state[i] * MOTOR_FORCE
        force = direction * (spring_force + damping_force + motor)
        agent1_forces[t, a] += force
        agent2_forces[t, b] -= force


@ti.kernel
def apply_forces(
    t: int, vertices: ti.template(), velocities: ti.template(), forces: ti.template()
):
    for i in range(NUM_OF_AGENT_VERTICES):
        g = GRAVITY * env_gravity_scale[None]
        velocities[t + 1, i] = velocities[t, i] + (g + forces[t, i]) * DT
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


@ti.kernel
def update_agent_weights(
    weights1: ti.template(),
    weights2: ti.template(),
    weights3: ti.template(),
    lr: float,
):
    for i, j in weights1:
        grad = weights1.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights1[i, j] -= grad * lr

    for i, j in weights2:
        grad = weights2.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights2[i, j] -= grad * lr

    for i, j in weights3:
        grad = weights3.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights3[i, j] -= grad * lr


def agent_lr(gen, generations=None):
    """Linear decay from LR -> LR_END over training."""
    total = generations if generations is not None else GENERATIONS
    if total <= 1:
        return LR_END
    frac = gen / (total - 1)
    return LR + (LR_END - LR) * frac


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


@ti.kernel
def clear_agent_state(
    velocities: ti.template(),
    forces: ti.template(),
):
    for t, i in ti.ndrange(TIME_STEPS, NUM_OF_AGENT_VERTICES):
        velocities[t, i] = ti.Vector([0.0, 0.0])
        forces[t, i] = ti.Vector([0.0, 0.0])


@ti.kernel
def copy_initial_pose(src: ti.template(), dst: ti.template()):
    for i in range(NUM_OF_AGENT_VERTICES):
        dst[0, i] = src[i]


def set_agent(agent):
    clear_agent_state(agent.velocities, agent.forces)
    if hasattr(agent, "initial_pose"):
        copy_initial_pose(agent.initial_pose, agent.vertices)


def mean_x(vertices_field, t):
    pts = vertices_field.to_numpy()[t]
    return float(pts[:, 0].mean())


def agent_fitness(agent):
    return mean_x(agent.vertices, TIME_STEPS - 1) - mean_x(agent.vertices, 0)


class Agent:
    def __init__(self, agent_id, directory="agents"):
        self.path = f"{directory}/agent{agent_id}.npz"
        loaded = np.load(self.path)
        self.points = loaded["points"].tolist()
        self.springs = loaded["springs"].tolist()
        self.fitness = float(loaded["fitness"]) if "fitness" in loaded.files else 0.0
        self.anchors_np = np.array(region_anchors(self.points), dtype=np.int32)

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
        self.anchors = ti.field(dtype=int, shape=(NUM_OF_CONNECTOR_EDGES,))
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

        self.weights1.from_numpy(loaded["weights1"].astype(np.float32))
        self.weights2.from_numpy(loaded["weights2"].astype(np.float32))
        self.weights3.from_numpy(loaded["weights3"].astype(np.float32))
        self.anchors.from_numpy(self.anchors_np)

        j = 0
        edges_np = np.zeros((NUM_OF_AGENT_EDGES, 2), dtype=np.int32)
        motors_np = np.zeros(NUM_OF_AGENT_ACTIVE_EDGES, dtype=np.int32)
        for i, spring in enumerate(self.springs):
            edges_np[i, 0] = int(spring[0])
            edges_np[i, 1] = int(spring[1])
            if int(spring[2]) == 1:
                motors_np[j] = i
                j += 1
        if j != NUM_OF_AGENT_ACTIVE_EDGES:
            raise ValueError(
                f"expected {NUM_OF_AGENT_ACTIVE_EDGES} active springs, got {j}"
            )
        self.edges_np = edges_np
        self.edges.from_numpy(edges_np)
        self.motor_indices.from_numpy(motors_np)

        initial_points = []
        for x, y in self.points:
            if y == 1.0 or y == 3.0:
                initial_points.append((x + 0.5, y))
            else:
                initial_points.append((x, y))

        min_x = min(x for x, _ in initial_points)
        min_y = min(y for _, y in initial_points)

        self.initial_pose = ti.Vector.field(n=2, dtype=float, shape=(NUM_OF_AGENT_VERTICES,))
        pose_np = np.zeros((NUM_OF_AGENT_VERTICES, 2), dtype=np.float32)
        for i, (x, y) in enumerate(initial_points):
            pose_np[i] = (x - min_x + START_MARGIN, y - min_y + START_MARGIN)
        self.initial_pose.from_numpy(pose_np)
        self.vertices.from_numpy(
            np.repeat(pose_np[None, :, :], TIME_STEPS, axis=0)
        )

        rests = np.zeros(NUM_OF_AGENT_EDGES, dtype=np.float32)
        for i in range(NUM_OF_AGENT_EDGES):
            a, b = int(edges_np[i, 0]), int(edges_np[i, 1])
            rests[i] = float(np.linalg.norm(pose_np[a] - pose_np[b]))
        self.resting_lengths.from_numpy(rests)

    def write(self, fitness=None):
        if fitness is not None:
            self.fitness = float(fitness)
        np.savez(
            self.path,
            points=np.asarray(self.points),
            springs=np.asarray(self.springs),
            status="trained",
            fitness=self.fitness,
            weights1=self.weights1.to_numpy(),
            weights2=self.weights2.to_numpy(),
            weights3=self.weights3.to_numpy(),
        )


class Connector:
    def __init__(self, connector_id, agent1, agent2, directory="connectors"):
        self.path = f"{directory}/connector{connector_id}.npz"
        loaded = np.load(self.path)
        self.type = loaded["connector_type"].item()
        self.strength = (
            loaded["strength"].item() if "strength" in loaded.files else "weak"
        )
        self.k = (
            WEAK_CONNECTOR_K if self.strength == "weak" else STRONG_CONNECTOR_K
        )
        self.agent1 = agent1
        self.agent2 = agent2
        self.resting_lengths = ti.field(dtype=float, shape=(NUM_OF_CONNECTOR_EDGES,))
        self.agent1_center = ti.Vector.field(n=2, dtype=float, shape=(), needs_grad=True)
        self.agent2_center = ti.Vector.field(n=2, dtype=float, shape=(), needs_grad=True)
        self.agent1_avg_velocity = ti.Vector.field(
            n=2, dtype=float, shape=(), needs_grad=True
        )
        self.agent2_avg_velocity = ti.Vector.field(
            n=2, dtype=float, shape=(), needs_grad=True
        )
        self.input_state = ti.field(
            dtype=float, shape=(CONNECTOR_INPUT_SIZE,), needs_grad=True
        )
        self.hidden_state = ti.field(
            dtype=float, shape=(CONNECTOR_HIDDEN_LAYER_SIZE,), needs_grad=True
        )
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
        self.output_state = ti.field(
            dtype=float, shape=(NUM_OF_CONNECTOR_EDGES,), needs_grad=True
        )

        self.weights1.from_numpy(loaded["weights1"].astype(np.float32))
        self.weights2.from_numpy(loaded["weights2"].astype(np.float32))
        self.resting_lengths.from_numpy(
            np.ones(NUM_OF_CONNECTOR_EDGES, dtype=np.float32)
        )

    def write(self):
        np.savez(
            self.path,
            connector_type=self.type,
            strength=self.strength,
            weights1=self.weights1.to_numpy(),
            weights2=self.weights2.to_numpy(),
        )
