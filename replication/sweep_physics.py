"""
Grid search over the body/actuation/contact physics constants.

Trains the SAME fixed seed agents (identical morphology + initial weights)
under each candidate constant combination, in its own subprocess, so morphology
luck doesn't confound the comparison -- only the physics changes between runs.

Usage:
    python sweep_physics.py --agents 3 --generations 40 \
        --spring-damping 15,25,40 \
        --motor-force 200,300,450 \
        --friction-smoothing 0.1,0.15,0.25
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import generate

SEED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_seeds")


def ensure_seeds(n):
    os.makedirs(SEED_DIR, exist_ok=True)
    for agent_id in range(n):
        path = os.path.join(SEED_DIR, f"agent{agent_id}.npz")
        if not os.path.exists(path):
            generate.generate_one_agent(agent_id, directory=SEED_DIR)
    print(f"[sweep] seed agents ready in {SEED_DIR}", flush=True)


def run_config(work_dir, agent_ids, generations, spring_damping, motor_force,
                friction_smoothing, ground_k=None, ground_damping=None):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sweep_worker.py"),
        "--seed-dir", SEED_DIR,
        "--work-dir", work_dir,
        "--agent-ids", ",".join(str(a) for a in agent_ids),
        "--generations", str(generations),
        "--spring-damping", str(spring_damping),
        "--motor-force", str(motor_force),
        "--friction-smoothing", str(friction_smoothing),
    ]
    if ground_k is not None:
        cmd += ["--ground-k", str(ground_k)]
    if ground_damping is not None:
        cmd += ["--ground-damping", str(ground_damping)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(
        f"worker produced no result (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
    )


def score(results):
    fitnesses = [r["fitness"] for r in results]
    worst_dip = min(min(r["history"], default=0.0) for r in results)
    return {
        "mean_fitness": sum(fitnesses) / len(fitnesses),
        "min_fitness": min(fitnesses),
        "worst_dip": worst_dip,
    }


def main():
    p = argparse.ArgumentParser(description="Sweep physics constants for gait learnability")
    p.add_argument("--agents", type=int, default=3, help="number of fixed seed agents to test each config on")
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--spring-damping", default="25")
    p.add_argument("--motor-force", default="300")
    p.add_argument("--friction-smoothing", default="0.15")
    p.add_argument("--ground-k", default=None)
    p.add_argument("--ground-damping", default=None)
    p.add_argument("--work-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_work"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.json"))
    args = p.parse_args()

    def parse_list(s):
        return [float(x) for x in s.split(",")]

    damping_vals = parse_list(args.spring_damping)
    motor_vals = parse_list(args.motor_force)
    friction_vals = parse_list(args.friction_smoothing)
    ground_k_vals = parse_list(args.ground_k) if args.ground_k else [None]
    ground_damping_vals = parse_list(args.ground_damping) if args.ground_damping else [None]

    agent_ids = list(range(args.agents))
    ensure_seeds(args.agents)

    combos = list(itertools.product(
        damping_vals, motor_vals, friction_vals, ground_k_vals, ground_damping_vals
    ))
    print(f"[sweep] {len(combos)} configs x {args.agents} agents x {args.generations} generations", flush=True)

    all_results = []
    started_all = time.perf_counter()
    for i, (sd, mf, fs, gk, gd) in enumerate(combos):
        config = {
            "spring_damping": sd, "motor_force": mf, "friction_smoothing": fs,
            "ground_k": gk, "ground_damping": gd,
        }
        started = time.perf_counter()
        results = run_config(args.work_dir, agent_ids, args.generations, sd, mf, fs, gk, gd)
        elapsed = time.perf_counter() - started
        s = score(results)
        print(
            f"[{i + 1}/{len(combos)}] {config} -> "
            f"mean={s['mean_fitness']:.3f} min={s['min_fitness']:.3f} "
            f"worst_dip={s['worst_dip']:.3f} ({elapsed:.1f}s)",
            flush=True,
        )
        all_results.append({"config": config, "results": results, "score": s})

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    ranked = sorted(all_results, key=lambda r: r["score"]["mean_fitness"], reverse=True)
    print(f"\n[sweep] done in {time.perf_counter() - started_all:.1f}s. Top 5 by mean fitness:", flush=True)
    for r in ranked[:5]:
        print(f"  {r['config']} -> {r['score']}", flush=True)
    print(f"[sweep] full results written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
