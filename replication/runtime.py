"""
Shared runtime settings. Configure before importing simulation.
"""

import os
import subprocess
import sys

ARCH = "gpu"
TIME_STEPS = 1000
GENERATIONS = 80
GPU_MEMORY_FRACTION = 0.4

# Body/actuation/contact physics. Read into simulation.py's globals at import
# time, so they're baked into Taichi's compiled kernels for the lifetime of
# the process -- change them (via configure_physics or CLI flags) BEFORE
# `import simulation`, not after.
SPRING_K = 1500.0
SPRING_DAMPING = 25.0
MOTOR_FORCE = 300.0
GROUND_K = 4000.0
GROUND_DAMPING = 106.0
FRICTION_SMOOTHING = 0.15
FRICTION_MU = 0.99
GRAVITY = -4.8  # matches the paper's default; real Earth gravity (-9.8) was 2x too strong
# Per-node total-force safety clamp. MOTOR_FORCE already hard-caps each motor
# spring (NN output is tanh-bounded before scaling), but the passive spring
# force, ground contact force, and several motors converging on one node are
# all otherwise unbounded -- this catches those blow-ups before they hit the integrator.
FORCE_CLAMP = 700.0  # every grounded (zero height-rise) agent in a 27-agent pool survey topped out at 561; every hopping one exceeded 897
# Matches SPRING_DAMPING: was 7 (damping ratio 0.08-0.25 depending on weak/strong
# connector stiffness -- same underdamping problem agent springs had before tuning).
CONNECTOR_DAMPING = 25.0
# Rewards variance in each connector spring's output over time (subtracted from
# loss), since with none the network converges to a constant, input-ignoring
# +-1 output rather than an actual responsive muscle.
CONNECTOR_VARIANCE_BONUS = 1.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    """Anchor a data directory/file to the replication folder, not the caller's cwd."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def configure(arch=None, time_steps=None, generations=None, gpu_memory_fraction=None):
    global ARCH, TIME_STEPS, GENERATIONS, GPU_MEMORY_FRACTION
    if arch is not None:
        ARCH = "gpu" if arch.lower() in ("gpu", "cuda", "vulkan") else "cpu"
    if time_steps is not None:
        TIME_STEPS = int(time_steps)
    if generations is not None:
        GENERATIONS = int(generations)
    if gpu_memory_fraction is not None:
        GPU_MEMORY_FRACTION = float(gpu_memory_fraction)


def configure_physics(
    spring_k=None,
    spring_damping=None,
    motor_force=None,
    ground_k=None,
    ground_damping=None,
    friction_smoothing=None,
    friction_mu=None,
    gravity=None,
    force_clamp=None,
    connector_damping=None,
    connector_variance_bonus=None,
):
    global SPRING_K, SPRING_DAMPING, MOTOR_FORCE
    global GROUND_K, GROUND_DAMPING, FRICTION_SMOOTHING, FRICTION_MU, GRAVITY, FORCE_CLAMP
    global CONNECTOR_DAMPING, CONNECTOR_VARIANCE_BONUS
    if spring_k is not None:
        SPRING_K = float(spring_k)
    if spring_damping is not None:
        SPRING_DAMPING = float(spring_damping)
    if motor_force is not None:
        MOTOR_FORCE = float(motor_force)
    if ground_k is not None:
        GROUND_K = float(ground_k)
    if ground_damping is not None:
        GROUND_DAMPING = float(ground_damping)
    if friction_smoothing is not None:
        FRICTION_SMOOTHING = float(friction_smoothing)
    if friction_mu is not None:
        FRICTION_MU = float(friction_mu)
    if gravity is not None:
        GRAVITY = float(gravity)
    if force_clamp is not None:
        FORCE_CLAMP = float(force_clamp)
    if connector_damping is not None:
        CONNECTOR_DAMPING = float(connector_damping)
    if connector_variance_bonus is not None:
        CONNECTOR_VARIANCE_BONUS = float(connector_variance_bonus)


def add_physics_args(parser):
    parser.add_argument("--spring-k", type=float, default=None)
    parser.add_argument("--spring-damping", type=float, default=None)
    parser.add_argument("--motor-force", type=float, default=None)
    parser.add_argument("--connector-damping", type=float, default=None)
    parser.add_argument("--connector-variance-bonus", type=float, default=None)
    parser.add_argument("--ground-k", type=float, default=None)
    parser.add_argument("--ground-damping", type=float, default=None)
    parser.add_argument("--friction-smoothing", type=float, default=None)
    parser.add_argument("--friction-mu", type=float, default=None)
    parser.add_argument("--gravity", type=float, default=None)
    parser.add_argument("--force-clamp", type=float, default=None)
    return parser


def add_arch_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gpu",
        action="store_true",
        help="Run Taichi on GPU (NVIDIA CUDA preferred; default).",
    )
    group.add_argument(
        "--cpu",
        action="store_true",
        help="Run Taichi on CPU (forces CPU mode).",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=None,
        help="Fraction of GPU memory Taichi may reserve (default 0.4).",
    )
    return parser


def add_train_args(parser):
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to sleep after each generation (limits sustained GPU load, reduces fan noise).",
    )
    return parser


def arch_from_args(args):
    if getattr(args, "cpu", False):
        return "cpu"
    return "gpu"


def print_gpu_power():
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "-i", "0",
                "--query-gpu=enforced.power.limit,clocks.current.graphics,clocks.max.graphics",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        power, clock, clock_max = (p.strip() for p in result.stdout.strip().split(","))
        print(
            f"[runtime] GPU power limit: {power} | graphics clock: {clock} (max {clock_max})",
            flush=True,
        )
    except FileNotFoundError:
        pass


def configure_from_args(args):
    configure(
        arch=arch_from_args(args),
        time_steps=getattr(args, "time_steps", None),
        generations=getattr(args, "generations", None),
        gpu_memory_fraction=getattr(args, "gpu_memory_fraction", None),
    )
    configure_physics(
        spring_k=getattr(args, "spring_k", None),
        spring_damping=getattr(args, "spring_damping", None),
        motor_force=getattr(args, "motor_force", None),
        ground_k=getattr(args, "ground_k", None),
        ground_damping=getattr(args, "ground_damping", None),
        friction_smoothing=getattr(args, "friction_smoothing", None),
        friction_mu=getattr(args, "friction_mu", None),
        gravity=getattr(args, "gravity", None),
        force_clamp=getattr(args, "force_clamp", None),
        connector_damping=getattr(args, "connector_damping", None),
        connector_variance_bonus=getattr(args, "connector_variance_bonus", None),
    )
    if ARCH == "gpu":
        print_gpu_power()
    return float(getattr(args, "pause", None) or 0.0)
