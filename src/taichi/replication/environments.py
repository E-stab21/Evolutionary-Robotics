"""
Novel deployment environments.
"""

import simulation as sim


def set_env(name="flat"):
    sim.reset_flat_env()
    if name == "flat":
        return
    if name == "incline":
        sim.env_slope[None] = 0.15
    elif name == "decline":
        sim.env_slope[None] = -0.15
    elif name == "ice":
        sim.env_friction[None] = 0.05
    elif name == "sand":
        sim.env_drag[None] = 8.0
        sim.env_ground_damping[None] = 400.0
    elif name == "wind":
        sim.env_wind[None] = -5.0
    elif name == "step_up":
        sim.env_step_x[None] = 1.5
        sim.env_step_h[None] = 0.35
    elif name == "gap":
        sim.env_gap_start[None] = 1.2
        sim.env_gap_end[None] = 1.7
    elif name == "low_gravity":
        sim.env_gravity_scale[None] = 0.35
    else:
        raise ValueError(f"Unknown environment: {name}")


ENVIRONMENTS = [
    "incline",
    "decline",
    "ice",
    "sand",
    "wind",
    "step_up",
    "gap",
    "low_gravity",
]
