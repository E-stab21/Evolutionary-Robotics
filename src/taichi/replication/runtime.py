"""
Shared runtime settings. Configure before importing simulation.
"""

ARCH = "gpu"
TIME_STEPS = 1000
GENERATIONS = 30
GPU_MEMORY_FRACTION = 0.4


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
    return parser


def arch_from_args(args):
    if getattr(args, "cpu", False):
        return "cpu"
    return "gpu"


def configure_from_args(args):
    configure(
        arch=arch_from_args(args),
        time_steps=getattr(args, "time_steps", None),
        generations=getattr(args, "generations", None),
        gpu_memory_fraction=getattr(args, "gpu_memory_fraction", None),
    )
