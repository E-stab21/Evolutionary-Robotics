"""
Main experiment control file
"""

import os
import generate
import train_agents
import train_connectors
import select_agents
import deploy


POOL_SIZE = 300
SELECTED = 100
CONNECTORS = 40


def _clear_npz(directory, prefix):
    os.makedirs(directory, exist_ok=True)
    for name in os.listdir(directory):
        if name.startswith(prefix) and name.endswith(".npz"):
            os.remove(os.path.join(directory, name))


def main():
    print(f"Generating {POOL_SIZE} agents into agents_pool/")
    _clear_npz("agents_pool", "agent")
    generate.generate_agents(POOL_SIZE, directory="agents_pool")

    print("Training agent pool on flat ground")
    train_agents.train_agents(0, POOL_SIZE - 1, directory="agents_pool")

    print(f"Selecting {SELECTED} agents with approx-normal fitness")
    _clear_npz("agents", "agent")
    stats = select_agents.select_normal(
        pool_dir="agents_pool", selected_dir="agents", n=SELECTED
    )
    print(
        f"pool mean={stats['pool_mean']:.4f} std={stats['pool_std']:.4f}; "
        f"selected mean={stats['selected_mean']:.4f} std={stats['selected_std']:.4f}"
    )

    print("Generating 40 connectors (weak/strong × uniform/diverse)")
    _clear_npz("connectors", "connector")
    generate.generate_connectors(num_per_set=10)

    print("Training connectors")
    train_connectors.train_connectors(0, CONNECTORS - 1, 0, SELECTED - 1)

    print("Deploying collectives to novel environments")
    deploy.run_deployment(
        agent_start=0, agent_end=SELECTED - 1, samples=20, out_path="results.csv"
    )
    print("Wrote results.csv")


if __name__ == "__main__":
    main()
