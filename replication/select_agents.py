"""
Subsample trained agents to an approximate normal fitness distribution.
"""

import os
import shutil
import numpy as np
import runtime as rt


POOL_DIR = "agents_pool"
SELECTED_DIR = "agents"
NUM_SELECTED = 100


def _normal_ppf(quantiles, mu, sigma):
    # Beasley-Springer-Moro style approximation via erfinv.
    return mu + sigma * np.sqrt(2.0) * erfinv(2.0 * quantiles - 1.0)


def erfinv(y):
    # Numerical Recipes approximation of inverse erf.
    y = np.asarray(y, dtype=float)
    a = 0.147
    sign = np.sign(y)
    ln = np.log(1.0 - y * y)
    first = 2.0 / (np.pi * a) + ln / 2.0
    return sign * np.sqrt(np.sqrt(first * first - ln / a) - first)


def load_pool_fitnesses(pool_dir=POOL_DIR):
    pool_dir = rt.resolve_path(pool_dir)
    rows = []
    for name in sorted(os.listdir(pool_dir)):
        if not name.startswith("agent") or not name.endswith(".npz"):
            continue
        path = os.path.join(pool_dir, name)
        with np.load(path) as loaded:
            fitness = float(loaded["fitness"])
        agent_id = int(name[len("agent") : -len(".npz")])
        rows.append((agent_id, fitness, path))
    rows.sort(key=lambda row: row[0])
    return rows


def select_normal(pool_dir=POOL_DIR, selected_dir=SELECTED_DIR, n=NUM_SELECTED):
    selected_dir = rt.resolve_path(selected_dir)
    rows = load_pool_fitnesses(pool_dir)
    if len(rows) < n:
        raise ValueError(f"Need at least {n} trained agents, found {len(rows)}")

    fitnesses = np.array([row[1] for row in rows], dtype=float)
    mu = float(fitnesses.mean())
    sigma = float(fitnesses.std())
    if sigma < 1e-8:
        sigma = 1.0

    lo, hi = float(fitnesses.min()), float(fitnesses.max())
    quantiles = (np.arange(n) + 0.5) / n
    targets = np.clip(_normal_ppf(quantiles, mu, sigma), lo, hi)

    remaining = list(range(len(rows)))
    chosen = []
    for target in targets:
        best_j = min(remaining, key=lambda j: abs(fitnesses[j] - target))
        chosen.append(best_j)
        remaining.remove(best_j)

    os.makedirs(selected_dir, exist_ok=True)
    for name in os.listdir(selected_dir):
        if name.startswith("agent") and name.endswith(".npz"):
            os.remove(os.path.join(selected_dir, name))

    selected_fitness = []
    for new_id, old_j in enumerate(chosen):
        src = rows[old_j][2]
        dst = os.path.join(selected_dir, f"agent{new_id}.npz")
        shutil.copy2(src, dst)
        selected_fitness.append(fitnesses[old_j])

    return {
        "pool_mean": mu,
        "pool_std": sigma,
        "selected_mean": float(np.mean(selected_fitness)),
        "selected_std": float(np.std(selected_fitness)),
        "selected_fitness": selected_fitness,
    }
