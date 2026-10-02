"""
Trains a fixed set of seed agents under one physics configuration and prints
the result as a JSON line. Runs as its own process because Taichi bakes
plain-Python constants into a kernel the first time it's compiled -- a single
process can't switch SPRING_K/MOTOR_FORCE/etc. between configs mid-run.
Invoked by sweep_physics.py; not meant to be run by hand.
"""
import argparse
import json
import os
import shutil
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--agent-ids", required=True, help="comma-separated")
    p.add_argument("--generations", type=int, required=True)
    p.add_argument("--spring-damping", type=float, required=True)
    p.add_argument("--motor-force", type=float, required=True)
    p.add_argument("--friction-smoothing", type=float, required=True)
    p.add_argument("--ground-k", type=float, default=None)
    p.add_argument("--ground-damping", type=float, default=None)
    args = p.parse_args()

    import runtime as rt

    rt.configure(generations=args.generations)
    rt.configure_physics(
        spring_damping=args.spring_damping,
        motor_force=args.motor_force,
        friction_smoothing=args.friction_smoothing,
        ground_k=args.ground_k,
        ground_damping=args.ground_damping,
    )

    import environments as envs
    import train_agents

    envs.set_env("flat")
    os.makedirs(args.work_dir, exist_ok=True)

    results = []
    for agent_id in (int(x) for x in args.agent_ids.split(",")):
        src = os.path.join(args.seed_dir, f"agent{agent_id}.npz")
        dst = os.path.join(args.work_dir, f"agent{agent_id}.npz")
        shutil.copy2(src, dst)
        history = []
        fitness = train_agents.train_one_agent(
            agent_id,
            directory=args.work_dir,
            pause=0.0,
            log_every=10,
            history=history,
        )
        results.append({"agent_id": agent_id, "fitness": fitness, "history": history})

    print("RESULT_JSON:" + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
