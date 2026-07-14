# imports
import taichi as ti
import secrets


# init
SEED = secrets.randbits(31)
ti.init(arch=ti.gpu, random_seed=464525965)  # secrets.randbits(31))
gui = ti.GUI()


# globals
NUM_OF_VERTICES = 10
NUM_OF_EDGES = 20

TIME_STEPS = 1000
DT = 0.001
GRAVITY = ti.Vector([0, -9.8])
SPRING_K = 2500.0
SPRING_DAMPING = 10.0
MOTOR_FORCE = 170

GROUND_K = 8000.0
GROUND_DAMPING = 150.0
GROUND_MARGIN = 0.02
FRICTION_SMOOTHING = 0.05
FRICTION_MU = 0.99

SENSOR_INDEXS = [0, 1, 2, 3, 5, 6]
MOTOR_INDEXS = [1, 4, 7, 10, 12, 15, 16, 18, 19]
NUM_OF_SENSOR_NEURONS = len(SENSOR_INDEXS)
NUM_OF_MOTOR_NEURONS = len(MOTOR_INDEXS)
SENSOR_SMOOTHING = 0.01
CPG_FREQUENCY = 20.0

LR = 0.01
GRAD_CLIP = 10.0
GENERATIONS = 80


# fields
vertices = ti.Vector.field(
    n=2, dtype=float, shape=(TIME_STEPS, NUM_OF_VERTICES), needs_grad=True
)
edges = ti.Vector.field(n=2, dtype=int, shape=(NUM_OF_EDGES,))
resting_lengths = ti.field(dtype=float, shape=(NUM_OF_EDGES,))

velocities = ti.Vector.field(
    n=2, dtype=float, shape=(TIME_STEPS, NUM_OF_VERTICES), needs_grad=True
)
forces = ti.Vector.field(
    n=2, dtype=float, shape=(TIME_STEPS, NUM_OF_VERTICES), needs_grad=True
)

brain_state = ti.field(
    dtype=float, shape=(TIME_STEPS, NUM_OF_SENSOR_NEURONS + 1), needs_grad=True
)
motor_state = ti.field(
    dtype=float,
    shape=(
        TIME_STEPS,
        NUM_OF_MOTOR_NEURONS,
    ),
    needs_grad=True,
)
weights = ti.field(
    dtype=float,
    shape=(NUM_OF_MOTOR_NEURONS + 1, NUM_OF_SENSOR_NEURONS),
    needs_grad=True,
)
loss = ti.field(dtype=float, shape=(), needs_grad=True)

sensor_indices = ti.field(dtype=int, shape=(NUM_OF_SENSOR_NEURONS,))
motor_indices = ti.field(dtype=int, shape=(NUM_OF_MOTOR_NEURONS,))

for i, e in enumerate(SENSOR_INDEXS):
    sensor_indices[i] = e
for i, e in enumerate(MOTOR_INDEXS):
    motor_indices[i] = e


@ti.kernel
def reset_robot():
    # robot vertice positions
    vertices[0, 0] = (0.2, 0.05)
    vertices[0, 1] = (0.3, 0.05)
    vertices[0, 2] = (0.5, 0.05)
    vertices[0, 3] = (0.6, 0.05)
    vertices[0, 4] = (0.2, 0.2)
    vertices[0, 5] = (0.3, 0.2)
    vertices[0, 6] = (0.5, 0.2)
    vertices[0, 7] = (0.6, 0.2)
    vertices[0, 8] = (0.3, 0.4)
    vertices[0, 9] = (0.5, 0.4)

    # robot edges
    edges[0] = (0, 1)
    edges[1] = (0, 4)
    edges[2] = (0, 5)
    edges[3] = (1, 4)
    edges[4] = (1, 5)
    edges[5] = (4, 5)
    edges[6] = (2, 3)
    edges[7] = (2, 6)
    edges[8] = (2, 7)
    edges[9] = (3, 6)
    edges[10] = (3, 7)
    edges[11] = (6, 7)
    edges[12] = (4, 8)
    edges[13] = (5, 8)
    edges[14] = (6, 9)
    edges[15] = (7, 9)
    edges[16] = (8, 9)
    edges[17] = (5, 6)
    edges[18] = (6, 8)
    edges[19] = (5, 9)

    # calculate resting positions
    for i in range(NUM_OF_EDGES):
        a, b = edges[i]
        diff = vertices[0, a] - vertices[0, b]
        resting_lengths[i] = diff.norm()

    # reseting all fields
    for t, i in ti.ndrange(TIME_STEPS, NUM_OF_VERTICES):
        velocities[t, i] = ti.Vector([0.0, 0.0])
        forces[t, i] = ti.Vector([0.0, 0.0])


@ti.kernel
def initialize_weights():
    for i, j in weights:
        weights[i, j] = ti.random() * 2.0 - 1.0


@ti.kernel
def compute_spring_forces(t: int):
    # finding spring forces
    for i in range(NUM_OF_EDGES):
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
def compute_ground_forces(t: int):
    for i in range(NUM_OF_VERTICES):
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
def compute_motor_forces(t: int):
    # getting senor values
    for i in range(NUM_OF_SENSOR_NEURONS):
        penetration = GROUND_MARGIN - vertices[t, sensor_indices[i]][1]
        brain_state[t, i] = 0.5 + 0.5 * ti.tanh(penetration / SENSOR_SMOOTHING)

    # update cpg neuron
    for i in range(1):
        brain_state[t, NUM_OF_SENSOR_NEURONS] = ti.sin(t * DT * CPG_FREQUENCY)

    # forward pass
    for i in range(NUM_OF_MOTOR_NEURONS):
        sum = 0.0
        for j in ti.static(range(NUM_OF_SENSOR_NEURONS + 1)):
            sum += brain_state[t, j] * weights[i, j]
        motor_state[t, i] = ti.tanh(sum)

    # applying activation funciton Leaky ReLu
    for i in range(NUM_OF_MOTOR_NEURONS):
        motor_state[t, i] = ti.tanh(motor_state[t, i])

    # converting motor values to forces
    for i in range(NUM_OF_MOTOR_NEURONS):
        a, b = edges[motor_indices[i]]
        delta = vertices[t, b] - vertices[t, a]
        force = delta.normalized() * motor_state[t, i] * MOTOR_FORCE
        forces[t, a] += force
        forces[t, b] -= force


@ti.kernel
def apply_forces(t: int):
    for i in range(NUM_OF_VERTICES):
        velocities[t + 1, i] = velocities[t, i] + (GRAVITY + forces[t, i]) * DT
        vertices[t + 1, i] = vertices[t, i] + velocities[t + 1, i] * DT


@ti.kernel
def compute_loss():
    loss[None] = 0
    for i in ti.static(range(NUM_OF_VERTICES)):
        loss[None] -= vertices[TIME_STEPS - 1, i][0]
    loss[None] /= NUM_OF_VERTICES
    print(loss[None])


@ti.kernel
def update_weights():
    for i, j in weights:
        grad = weights.grad[i, j]
        grad = ti.max(ti.min(grad, GRAD_CLIP), -GRAD_CLIP)
        weights[i, j] -= grad * LR
        # print(grad)


def display(t: int, video_manager=None):
    gui.line([0.0, 0.0], [100.0, 0.0], radius=2, color=0xFFFFFF)
    for i in range(NUM_OF_EDGES):
        a, b = edges[i]
        gui.line(
            [vertices[t, a][0], vertices[t, a][1]],
            [vertices[t, b][0], vertices[t, b][1]],
            radius=3,
            color=0x068587,
        )
    if video_manager is not None:
        video_manager.write_frame(gui.get_image())
    gui.show()


# Simulations
def simulate(with_display=False, video_manager=None):
    for t in range(TIME_STEPS - 1):
        compute_spring_forces(t)
        compute_ground_forces(t)
        compute_motor_forces(t)
        apply_forces(t)
        if with_display:
            display(t, video_manager)


if __name__ == "__main__":
    initialize_weights()
    for _ in range(GENERATIONS):
        reset_robot()
        with ti.ad.Tape(loss=loss):
            simulate(with_display=False)
            compute_loss()
        update_weights()

    reset_robot()

    # Save a video of the final simulation
    video_manager = ti.tools.VideoManager(
        output_dir="/home/ethan/Projects/Evolutionary-Robotics/output",
        framerate=60,
        automatic_build=False,
    )
    simulate(with_display=True, video_manager=video_manager)
    print("Exporting video...")
    video_manager.make_video(gif=False, mp4=True)

    print(weights)
    print(SEED)
