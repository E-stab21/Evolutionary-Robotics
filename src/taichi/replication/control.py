"""
Main experiment control file.

Default is a cool CPU light run. For the full paper-scale job:
  META_ROBOTS_ARCH=gpu python control.py --full
"""

import argparse
import os


def _parse_args():
    parser = argparse.ArgumentParser(description="Simplified meta-robots experiment")
    parser.add_argument(
        "--full",
        action="store_true",
        help="300 agents, 100 selected, 40 connectors, full gens/steps (heavy)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU (can spin fans). Default is CPU.",
    )
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--selected", type=int, default=None)
    parser.add_argument("--connectors-per-set", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    return parser.parse_args()


def _configure_runtime(args):
    if args.full:
        pool, selected, per_set, samples = 300, 100, 10, 20
        generations, time_steps = 30, 1000
    else:
        # Tiny defaults so a validation pass stays quiet.
        pool, selected, per_set, samples = 12, 8, 1, 1
        generations, time_steps = 2, 80

    if args.pool is not None:
        pool = args.pool
    if args.selected is not None:
        selected = args.selected
    if args.connectors_per_set is not None:
        per_set = args.connectors_per_set
    if args.samples is not None:
        samples = args.samples
    if args.generations is not None:
        generations = args.generations
    if args.time_steps is not None:
        time_steps = args.time_steps

    os.environ["META_ROBOTS_ARCH"] = "gpu" if args.gpu else "cpu"
    os.environ["META_ROBOTS_GENERATIONS"] = str(generations)
    os.environ["META_ROBOTS_TIME_STEPS"] = str(time_steps)
    return pool, selected, per_set, samples


def _clear_npz(directory, prefix):
    os.makedirs(directory, exist_ok=True)
    for name in os.listdir(directory):
        if name.startswith(prefix) and name.endswith(".npz"):
            os.remove(os.path.join(directory, name))


def main():
    args = _parse_args()
    pool, selected, per_set, samples = _configure_runtime(args)

    import generate
    import train_agents
    import train_connectors
    import select_agents
    import deploy
    import simulation as sim

    connectors = per_set * 4
    print(
        f"arch={os.environ['META_ROBOTS_ARCH']} gens={sim.GENERATIONS} "
        f"steps={sim.TIME_STEPS} pool={pool} selected={selected} "
        f"connectors={connectors} samples={samples}",
        flush=True,
    )

    print(f"Generating {pool} agents into agents_pool/", flush=True)
    _clear_npz("agents_pool", "agent")
    generate.generate_agents(pool, directory="agents_pool")

    print("Training agent pool on flat ground", flush=True)
    train_agents.train_agents(0, pool - 1, directory="agents_pool")

    print(f"Selecting {selected} agents with approx-normal fitness", flush=True)
    _clear_npz("agents", "agent")
    stats = select_agents.select_normal(
        pool_dir="agents_pool", selected_dir="agents", n=selected
    )
    print(
        f"pool mean={stats['pool_mean']:.4f} std={stats['pool_std']:.4f}; "
        f"selected mean={stats['selected_mean']:.4f} std={stats['selected_std']:.4f}",
        flush=True,
    )

    print(
        f"Generating {connectors} connectors (weak/strong × uniform/diverse)",
        flush=True,
    )
    _clear_npz("connectors", "connector")
    generate.generate_connectors(num_per_set=per_set)

    print("Training connectors", flush=True)
    train_connectors.train_connectors(0, connectors - 1, 0, selected - 1)

    print("Deploying collectives to novel environments", flush=True)
    deploy.run_deployment(
        agent_start=0,
        agent_end=selected - 1,
        samples=samples,
        out_path="results.csv",
    )
    print("Wrote results.csv", flush=True)


if __name__ == "__main__":
    main()
