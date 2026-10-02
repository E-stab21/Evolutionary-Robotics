"""
Evaluate one already-trained agent at a different TIME_STEPS than it was
trained with, in its own process (Taichi bakes constants like TIME_STEPS into
compiled kernels per-process, so this can't happen in-process alongside
training at a different length). Prints "FITNESS:<value>" to stdout.
Invoked by build_quality_pool's eval_time_steps option; not meant to be run
by hand.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-id", type=int, required=True)
    p.add_argument("--directory", required=True)
    p.add_argument("--time-steps", type=int, required=True)
    args = p.parse_args()

    import runtime as rt
    rt.configure(time_steps=args.time_steps)

    import simulation as sim
    import environments as envs
    import train_agents

    envs.set_env("flat")
    agent = sim.Agent(args.agent_id, directory=args.directory)
    sim.set_agent(agent)
    train_agents.simulate(agent)
    fitness = sim.agent_fitness(agent)
    print(f"FITNESS:{fitness}", flush=True)


if __name__ == "__main__":
    main()
