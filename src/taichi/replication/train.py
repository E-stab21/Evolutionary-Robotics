# imports
import taichi as ti
import csv
import secrets


# init
SEED = secrets.randbits(31)
ti.init(
    arch=ti.gpu, random_seed=464525965, debug=True, unrolling_limit=0
)  # secrets.randbits(31))
gui = ti.GUI("Sim", res=600)


# constants
TIME_STEPS = 2000
DT = 0.005
GRAVITY = ti.Vector([0, -9.8])
SPRING_K = 2500.0
SPRING_DAMPING = 10.0
MOTOR_FORCE = 100

GROUND_K = 8000.0
GROUND_DAMPING = 150.0
GROUND_MARGIN = 0.02
FRICTION_SMOOTHING = 0.05
FRICTION_MU = 0.99

SENSOR_SMOOTHING = 0.01
CPG_FREQUENCY = 20.0

LR = 0.01
GRAD_CLIP = 10.0
GENERATIONS = 20

SCALE = 10.0


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
    # for i in range(NUM_OF_VERTICES):
    #     penetration = GROUND_MARGIN - vertices[t, sensor_indices[i]][1]
    #     brain_state[t, i] = 0.5 + 0.5 * ti.tanh(penetration / SENSOR_SMOOTHING)

    # center of mass
    for _ in range(1):
        center_of_mass[None] = ti.Vector([0.0, 0.0])
    for i in range(NUM_OF_VERTICES):
        center_of_mass[None] += vertices[t, i]
    for _ in range(1):
        center_of_mass[None] /= NUM_OF_VERTICES

    # setting vertice and velocitiy neurons
    for i in range(NUM_OF_VERTICES):
        input_state[t, i] = vertices[t, i][0] - center_of_mass[None][0]
        input_state[t, i + NUM_OF_VERTICES] = (
            vertices[t, i][1] - center_of_mass[None][1]
        )
        input_state[t, i + NUM_OF_VERTICES * 2] = velocities[t, i][0]
        input_state[t, i + NUM_OF_VERTICES * 3] = velocities[t, i][1]

    # setting cpg neurons
    for i in range(10):
        input_state[t, i + NUM_OF_VERTICES * 4] = ti.sin(
            t * DT * CPG_FREQUENCY * i + i * 0.35
        )

    # forward pass 1
    for i in range(MIDDLE_LAYER_SIZE):
        sum = 0.0
        for j in ti.static(range(NUM_OF_INPUTS)):
            sum += input_state[t, j] * weights1[i, j]
        neural_state1[t, i] = ti.tanh(sum)

    # forward pass 2
    for i in range(MIDDLE_LAYER_SIZE):
        sum = 0.0
        for j in ti.static(range(MIDDLE_LAYER_SIZE)):
            sum += neural_state1[t, j] * weights2[i, j]
        neural_state2[t, i] = ti.tanh(sum)

    # forward pass 3
    for i in range(NUM_OF_OUTPUTS):
        sum = 0.0
        for j in ti.static(range(MIDDLE_LAYER_SIZE)):
            sum += neural_state2[t, j] * weights3[i, j]
        output_state[t, i] = ti.tanh(sum)

    # converting motor values to forces
    for i in range(NUM_OF_OUTPUTS):
        a, b = edges[motor_indices[i]]
        delta = vertices[t, b] - vertices[t, a]
        force = delta.normalized() * output_state[t, i] * MOTOR_FORCE
        forces[t, a] += force
        forces[t, b] -= force


@ti.kernel
def apply_forces(t: int):
    for i in range(NUM_OF_VERTICES):
        velocities[t + 1, i] = velocities[t, i] + (GRAVITY + forces[t, i]) * DT
        vertices[t + 1, i] = vertices[t, i] + velocities[t + 1, i] * DT


@ti.kernel
def compute_loss():
    for _ in range(1):
        loss[None] = 0
    for i in range(NUM_OF_VERTICES):
        loss[None] -= vertices[TIME_STEPS - 1, i][0]
    for _ in range(1):
        loss[None] /= NUM_OF_VERTICES
    # print(loss[None])


@ti.kernel
def update_weights():
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


# python functions
def read_in_agent():
    # reading in the file
    with open(f"agents/agent{id}.csv", "r") as f:
        data = csv.reader(f)
        state = 0
        for row in data:
            # skip over the start
            if not state:
                state += 1
                continue

            try:
                row = [float(i.strip()) for i in row]
            except ValueError:
                state += 1
                continue

            # reading
            if state == 1:
                points.append(row)
            elif state == 2:
                if row[2]:
                    active_springs.append(len(springs))
                springs.append((int(row[0]), int(row[1])))
            elif state == 3:
                matrix1.append(row)
            elif state == 4:
                matrix2.append(row)
            elif state == 5:
                matrix3.append(row)

    # offsetting grid
    for i in range(len(points)):
        if points[i][1] == 2.0 or points[i][1] == 4.0:
            points[i][0] += 0.5


def overwriting_weights(n, m, weights, lines, line_i):
    for i in range(n):
        temp = []
        for j in range(m):
            temp.append(str(weights[i, j]))
        lines[line_i + i] = temp


def writing_out_agent():
    lines = None
    with open(f"agents/agent{id}.csv", "r") as f:
        lines = list(csv.reader(f))
        state = 0
        for line_i in range(len(lines)):
            if lines[line_i][:-1] == "Matrix":
                state += 1

            if state == 1:
                overwriting_weights(
                    MIDDLE_LAYER_SIZE, NUM_OF_INPUTS, weights1, lines, line_i
                )
            if state == 2:
                overwriting_weights(
                    MIDDLE_LAYER_SIZE, MIDDLE_LAYER_SIZE, weights2, lines, line_i
                )
            if state == 3:
                overwriting_weights(
                    NUM_OF_OUTPUTS, MIDDLE_LAYER_SIZE, weights3, lines, line_i
                )

    with open(f"agents/agent{id}.csv", "w") as f:
        csv.writer(f).writerows(lines)


def set_weights():
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            weights1[i, j] = matrix1[i][j]
    for i in range(len(matrix2)):
        for j in range(len(matrix2[0])):
            weights2[i, j] = matrix2[i][j]
    for i in range(len(matrix3)):
        for j in range(len(matrix3[0])):
            weights3[i, j] = matrix3[i][j]


def set_sim():
    # setting vertice positions
    for i, point in enumerate(points):
        x, y = point
        vertices[0, i] = (x, y)

    # setting edges
    for i, spring in enumerate(springs):
        p1, p2 = spring
        edges[i] = ti.Vector([p1, p2])

    # calculate resting positions
    for i in range(NUM_OF_EDGES):
        a, b = edges[i]
        diff = vertices[0, a] - vertices[0, b]
        resting_lengths[i] = diff.norm()

    # setting active edges
    for i, active_spring in enumerate(active_springs):
        motor_indices[i] = active_spring

    # reseting all fields
    for t, i in ti.ndrange(TIME_STEPS, NUM_OF_VERTICES):
        velocities[t, i] = ti.Vector([0.0, 0.0])
        forces[t, i] = ti.Vector([0.0, 0.0])


def display(t: int, video_manager=None):
    gui.line([0.0, 0.0], [1.0, 0.0], radius=2, color=0xFFFFFF)
    for i in range(NUM_OF_EDGES):
        a, b = edges[i]
        gui.line(
            [vertices[t, a][0] / SCALE, vertices[t, a][1] / SCALE],
            [vertices[t, b][0] / SCALE, vertices[t, b][1] / SCALE],
            radius=3,
            color=0x068587,
        )
    if video_manager is not None:
        video_manager.write_frame(gui.get_image())
    gui.show()


def simulate(with_display=False, video_manager=None):
    for t in range(TIME_STEPS - 1):
        compute_spring_forces(t)
        compute_ground_forces(t)
        compute_motor_forces(t)
        apply_forces(t)
        if with_display:
            display(t, video_manager)
        # for i in range(NUM_OF_VERTICES):
        # print(vertices[t, i])


def save_video(with_display=False):
    # Save a video of the final simulation
    video_manager = ti.tools.VideoManager(
        output_dir="/home/ethan/Projects/Evolutionary-Robotics/vids",
        framerate=60,
        automatic_build=True,
    )
    simulate(with_display=with_display, video_manager=video_manager)
    print("Exporting video...")
    video_manager.make_video(gif=False, mp4=True)


if __name__ == "__main__":
    for id in range(1):
        points = []
        springs = []
        active_springs = []
        matrix1 = []
        matrix2 = []
        matrix3 = []

        read_in_agent()

        # constants
        NUM_OF_VERTICES = len(points)
        NUM_OF_EDGES = len(springs)
        NUM_OF_ACTIVE = len(active_springs)

        NUM_OF_INPUTS = len(matrix1[0])
        MIDDLE_LAYER_SIZE = len(matrix1)
        NUM_OF_OUTPUTS = len(matrix3)

        # sim
        velocities = ti.Vector.field(
            n=2, dtype=float, shape=(TIME_STEPS, NUM_OF_VERTICES), needs_grad=True
        )
        forces = ti.Vector.field(
            n=2, dtype=float, shape=(TIME_STEPS, NUM_OF_VERTICES), needs_grad=True
        )

        # body
        vertices = ti.Vector.field(
            n=2, dtype=float, shape=(TIME_STEPS, NUM_OF_VERTICES), needs_grad=True
        )
        edges = ti.Vector.field(n=2, dtype=int, shape=(NUM_OF_EDGES,))
        resting_lengths = ti.field(dtype=float, shape=(NUM_OF_EDGES,))
        motor_indices = ti.field(dtype=int, shape=(NUM_OF_ACTIVE,))
        center_of_mass = ti.Vector.field(n=2, dtype=float, shape=(), needs_grad=True)

        # brain
        input_state = ti.field(
            dtype=float, shape=(TIME_STEPS, NUM_OF_INPUTS), needs_grad=True
        )
        neural_state1 = ti.field(
            dtype=float, shape=(TIME_STEPS, MIDDLE_LAYER_SIZE), needs_grad=True
        )
        neural_state2 = ti.field(
            dtype=float, shape=(TIME_STEPS, MIDDLE_LAYER_SIZE), needs_grad=True
        )
        output_state = ti.field(
            dtype=float, shape=(TIME_STEPS, NUM_OF_OUTPUTS), needs_grad=True
        )
        weights1 = ti.field(
            dtype=float, shape=(MIDDLE_LAYER_SIZE, NUM_OF_INPUTS), needs_grad=True
        )
        weights2 = ti.field(
            dtype=float, shape=(MIDDLE_LAYER_SIZE, MIDDLE_LAYER_SIZE), needs_grad=True
        )
        weights3 = ti.field(
            dtype=float, shape=(NUM_OF_OUTPUTS, MIDDLE_LAYER_SIZE), needs_grad=True
        )
        loss = ti.field(dtype=float, shape=(), needs_grad=True)

        set_weights()
        for _ in range(GENERATIONS):
            set_sim()
            with ti.ad.Tape(loss=loss):
                simulate()
                compute_loss()
            update_weights()

        set_sim()

        #  ffmpeg -framerate 30 -i 'frame_%06d.png' -c:v libx264 -pix_fmt yuv420p output.mp4
