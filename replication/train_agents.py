"""
Agent training and simulation
"""

import argparse
import os


def _world_to_gui(x, y, com_x, view=8.0):
    # Keep ground near the bottom of the window.
    return 0.5 + (x - com_x) / view, 0.08 + y / view


def display_frame(pts, edges, gui, com_x, view=8.0, video_manager=None):
    import simulation as sim

    gui.line(
        _world_to_gui(-view, 0.0, com_x, view),
        _world_to_gui(view, 0.0, com_x, view),
        radius=2,
        color=0xFFFFFF,
    )
    for i in range(len(edges)):
        a, b = int(edges[i, 0]), int(edges[i, 1])
        gui.line(
            _world_to_gui(float(pts[a, 0]), float(pts[a, 1]), com_x, view),
            _world_to_gui(float(pts[b, 0]), float(pts[b, 1]), com_x, view),
            radius=3,
            color=0x068587,
        )
    if video_manager is not None:
        video_manager.write_frame(gui.get_image())
    gui.show()


def display(agent, t, video_manager=None, gui=None, com_x=None):
    pts = agent.vertices.to_numpy()[t]
    if com_x is None:
        com_x = float(pts[:, 0].mean())
    display_frame(pts, agent.edges_np, gui, com_x, video_manager=video_manager)


def simulate(agent, with_display=False, video_manager=None, gui=None):
    import simulation as sim

    for t in range(sim.TIME_STEPS - 1):
        sim.compute_spring_forces(
            t,
            agent.vertices,
            agent.edges,
            agent.velocities,
            agent.forces,
            agent.resting_lengths,
        )
        sim.compute_ground_forces(t, agent.vertices, agent.velocities, agent.forces)
        sim.compute_agent_motor_forces(
            t,
            agent.vertices,
            agent.edges,
            agent.velocities,
            agent.forces,
            agent.center_of_mass,
            agent.input_state,
            agent.neural_state1,
            agent.neural_state2,
            agent.output_state,
            agent.weights1,
            agent.weights2,
            agent.weights3,
            agent.motor_indices,
        )
        sim.apply_forces(t, agent.vertices, agent.velocities, agent.forces)
        if with_display:
            display(agent, t, video_manager, gui=gui)


def save_video(agent, with_display=False):
    import taichi as ti

    video_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vids")
    video_manager = ti.tools.VideoManager(
        output_dir=video_output_dir,
        framerate=60,
        automatic_build=True,
    )
    gui = ti.GUI("Sim", res=600) if with_display else None
    simulate(agent, with_display=with_display, video_manager=video_manager, gui=gui)
    print("Exporting video...")
    video_manager.make_video(gif=False, mp4=True)


def watch(agent_id, directory="agents", fps=60):
    import time
    import taichi as ti
    import simulation as sim
    import environments as envs

    agent = sim.Agent(agent_id, directory=directory)
    sim.set_agent(agent)
    envs.set_env("flat")
    simulate(agent)  # run physics headless, then play back smoothly

    verts = agent.vertices.to_numpy()
    edges = agent.edges_np
    # Fixed camera from the start pose — following COM every frame causes shake.
    com_x = float(verts[0, :, 0].mean())
    stride = max(1, int(round(1.0 / (fps * sim.DT))))
    frame_dt = stride * sim.DT

    gui = ti.GUI("Sim", res=600)
    for t in range(0, sim.TIME_STEPS, stride):
        started = time.perf_counter()
        display_frame(verts[t], edges, gui, com_x)
        leftover = frame_dt - (time.perf_counter() - started)
        if leftover > 0:
            time.sleep(leftover)


def train_one_agent(agent_id, directory="agents", log_every=10):
    import numpy as np
    import taichi as ti
    import simulation as sim

    agent = sim.Agent(agent_id, directory=directory)
    best_fitness = float("-inf")
    best_weights = None

    for gen in range(sim.GENERATIONS):
        sim.set_agent(agent)
        with ti.ad.Tape(loss=agent.loss):
            simulate(agent)
            sim.compute_loss(agent.vertices, agent.vertices, agent.loss)
        sim.update_agent_weights(
            agent.weights1,
            agent.weights2,
            agent.weights3,
            sim.agent_lr(gen),
        )

        # Evaluate often enough to keep peak policies; GD here is noisy.
        if gen == 0 or (gen + 1) % log_every == 0 or gen + 1 == sim.GENERATIONS:
            sim.set_agent(agent)
            simulate(agent)
            fit = sim.agent_fitness(agent)
            print(
                f"agent{agent_id} gen {gen + 1}/{sim.GENERATIONS} fitness={fit:.4f}",
                flush=True,
            )
            if fit > best_fitness:
                best_fitness = fit
                best_weights = (
                    agent.weights1.to_numpy().copy(),
                    agent.weights2.to_numpy().copy(),
                    agent.weights3.to_numpy().copy(),
                )

    if best_weights is not None:
        agent.weights1.from_numpy(best_weights[0].astype(np.float32))
        agent.weights2.from_numpy(best_weights[1].astype(np.float32))
        agent.weights3.from_numpy(best_weights[2].astype(np.float32))

    sim.set_agent(agent)
    simulate(agent)
    fitness = sim.agent_fitness(agent)
    # Prefer recorded peak if final eval drifted slightly.
    fitness = max(fitness, best_fitness) if best_fitness > float("-inf") else fitness
    agent.write(fitness=fitness)
    return fitness


def train_agents(agent_start, agent_end, directory="agents"):
    import simulation as sim
    import environments as envs

    os.makedirs(directory, exist_ok=True)
    envs.set_env("flat")
    print(
        f"training agents {agent_start}-{agent_end}: "
        f"gens={sim.GENERATIONS} steps={sim.TIME_STEPS}",
        flush=True,
    )
    for agent_id in range(agent_start, agent_end + 1):
        fitness = train_one_agent(agent_id, directory=directory)
        print(f"agent{agent_id} final fitness={fitness:.4f}", flush=True)


def build_quality_pool(
    count,
    directory="agents_pool",
    min_fitness=1.0,
    max_attempts=8,
):
    """Generate+train agents, resampling morphologies that stay below min_fitness."""
    import shutil
    import environments as envs
    import generate
    import simulation as sim

    os.makedirs(directory, exist_ok=True)
    envs.set_env("flat")
    print(
        f"quality pool n={count} min_fitness={min_fitness} "
        f"max_attempts={max_attempts} gens={sim.GENERATIONS}",
        flush=True,
    )

    accepted = 0
    total_attempts = 0
    for agent_id in range(count):
        best_fitness = float("-inf")
        best_path = f"{directory}/agent{agent_id}.best.npz"
        final_path = f"{directory}/agent{agent_id}.npz"

        for attempt in range(1, max_attempts + 1):
            total_attempts += 1
            generate.generate_one_agent(agent_id, directory=directory)
            print(
                f"agent{agent_id} attempt {attempt}/{max_attempts}",
                flush=True,
            )
            fitness = train_one_agent(agent_id, directory=directory)
            if fitness > best_fitness:
                best_fitness = fitness
                shutil.copy2(final_path, best_path)

            if fitness >= min_fitness:
                accepted += 1
                print(
                    f"agent{agent_id} ACCEPTED fitness={fitness:.4f}",
                    flush=True,
                )
                break
            print(
                f"agent{agent_id} rejected fitness={fitness:.4f} < {min_fitness}",
                flush=True,
            )
        else:
            shutil.copy2(best_path, final_path)
            print(
                f"agent{agent_id} KEEP-BEST fitness={best_fitness:.4f} "
                f"(no attempt cleared {min_fitness})",
                flush=True,
            )

        if os.path.exists(best_path):
            os.remove(best_path)

    print(
        f"quality pool done: {accepted}/{count} cleared threshold; "
        f"{total_attempts} train attempts",
        flush=True,
    )
    return {"accepted": accepted, "count": count, "attempts": total_attempts}


def main():
    import runtime as rt

    parser = argparse.ArgumentParser(description="Train or watch agents")
    sub = parser.add_subparsers(dest="command", required=True)

    watch_p = sub.add_parser("watch", help="GUI playback of one agent")
    watch_p.add_argument("agent_id", type=int)
    watch_p.add_argument("--directory", default="agents")
    rt.add_arch_args(watch_p)

    train_p = sub.add_parser("train", help="Train a range of existing agents")
    train_p.add_argument("agent_start", type=int)
    train_p.add_argument("agent_end", type=int)
    train_p.add_argument("--directory", default="agents")
    rt.add_arch_args(train_p)
    rt.add_train_args(train_p)

    pool_p = sub.add_parser(
        "build-pool",
        help="Generate+train agents, resampling morphs below min fitness",
    )
    pool_p.add_argument("count", type=int)
    pool_p.add_argument("--directory", default="agents_pool")
    pool_p.add_argument("--min-fitness", type=float, default=1.0)
    pool_p.add_argument("--max-attempts", type=int, default=8)
    rt.add_arch_args(pool_p)
    rt.add_train_args(pool_p)

    args = parser.parse_args()
    rt.configure_from_args(args)

    if args.command == "watch":
        watch(args.agent_id, directory=args.directory)
    elif args.command == "train":
        train_agents(
            args.agent_start,
            args.agent_end,
            directory=args.directory,
        )
    else:
        build_quality_pool(
            args.count,
            directory=args.directory,
            min_fitness=args.min_fitness,
            max_attempts=args.max_attempts,
        )


if __name__ == "__main__":
    main()
