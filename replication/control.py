"""
Main experiment control file.

Default is a cool CPU light run. For the full paper-scale job:
  python control.py --full --gpu
"""

import argparse
import os
import runtime as rt


def _parse_args():
    parser = argparse.ArgumentParser(description="Simplified meta-robots experiment")
    parser.add_argument(
        "--full",
        action="store_true",
        help="300 agents, 100 selected, 40 connectors, full gens/steps (heavy)",
    )
    rt.add_arch_args(parser)
    rt.add_train_args(parser)
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--selected", type=int, default=None)
    parser.add_argument("--connectors-per-set", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=None,
        help="Reject morphologies below this flat-ground fitness (default 1.0 full / 0.2 light)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max generate+train tries per agent slot (default 8)",
    )
    return parser.parse_args()


def _configure_runtime(args):
    if args.full:
        pool, selected, per_set, samples = 300, 100, 10, 20
        generations, time_steps = 100, 1000
        min_fitness, max_attempts = 1.0, 8
    else:
        pool, selected, per_set, samples = 12, 8, 1, 1
        generations, time_steps = 2, 80
        min_fitness, max_attempts = 0.2, 3

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
    if args.min_fitness is not None:
        min_fitness = args.min_fitness
    if args.max_attempts is not None:
        max_attempts = args.max_attempts

    pause = rt.configure_from_args(args)
    # configure_from_args already set arch; ensure gens/steps match resolved values
    rt.configure(generations=generations, time_steps=time_steps)
    return pool, selected, per_set, samples, min_fitness, max_attempts, pause


def _clear_npz(directory, prefix):
    directory = rt.resolve_path(directory)
    os.makedirs(directory, exist_ok=True)
    for name in os.listdir(directory):
        if name.startswith(prefix) and name.endswith(".npz"):
            os.remove(os.path.join(directory, name))


def main():
    args = _parse_args()
    (
        pool,
        selected,
        per_set,
        samples,
        min_fitness,
        max_attempts,
        pause,
    ) = _configure_runtime(args)

    import generate
    import train_agents
    import train_connectors
    import select_agents
    import deploy
    import simulation as sim

    connectors = per_set * 4
    print(
        f"arch={rt.ARCH} gens={sim.GENERATIONS} steps={sim.TIME_STEPS} "
        f"pool={pool} selected={selected} connectors={connectors} samples={samples} "
        f"min_fitness={min_fitness} max_attempts={max_attempts} pause={pause}",
        flush=True,
    )

    print("Building quality agent pool (generate + train + reject)", flush=True)
    _clear_npz("agents_pool", "agent")
    train_agents.build_quality_pool(
        pool,
        directory="agents_pool",
        min_fitness=min_fitness,
        max_attempts=max_attempts,
        pause=pause,
    )

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
