"""
Zero-shot deployment of collective types into novel environments.
"""

import csv
import os
import random
import numpy as np
import simulation as sim
import environments as envs
import train_agents
import train_connectors


COLLECTIVE_TYPES = [
    ("weak", "uniform", "uniform"),
    ("weak", "uniform", "diverse"),
    ("weak", "diverse", "uniform"),
    ("weak", "diverse", "diverse"),
    ("strong", "uniform", "uniform"),
    ("strong", "uniform", "diverse"),
    ("strong", "diverse", "uniform"),
    ("strong", "diverse", "diverse"),
]


def list_connectors(strength, train_pairing, max_id=200):
    ids = []
    for connector_id in range(max_id):
        path = f"connectors/connector{connector_id}.npz"
        if not os.path.exists(path):
            continue
        with np.load(path) as loaded:
            if (
                loaded["strength"].item() == strength
                and loaded["connector_type"].item() == train_pairing
            ):
                ids.append(connector_id)
    return ids


def collective_fitness(connector):
    f1 = sim.agent_fitness(connector.agent1)
    f2 = sim.agent_fitness(connector.agent2)
    return 0.5 * (f1 + f2)


def eval_independent(agent_ids, env_name, samples=20):
    envs.set_env(env_name)
    scores = []
    for agent_id in random.sample(agent_ids, min(samples, len(agent_ids))):
        agent = sim.Agent(agent_id)
        sim.set_agent(agent)
        train_agents.simulate(agent)
        scores.append(sim.agent_fitness(agent))
    return float(np.mean(scores)) if scores else 0.0


def eval_collective(strength, train_pairing, deploy_pairing, agent_ids, env_name, samples=20):
    connector_ids = list_connectors(strength, train_pairing)
    if not connector_ids:
        return 0.0

    envs.set_env(env_name)
    scores = []
    for _ in range(samples):
        agent_id = random.choice(agent_ids)
        partner_id = train_connectors.choose_partner_id(
            agent_id, agent_ids, deploy_pairing
        )
        connector_id = random.choice(connector_ids)
        connector = sim.Connector(
            connector_id, sim.Agent(agent_id), sim.Agent(partner_id)
        )
        sim.set_agent(connector.agent1)
        sim.set_agent(connector.agent2)
        train_connectors.place_pair(connector.agent1, connector.agent2)
        train_connectors.refresh_rest_lengths(connector)
        train_connectors.simulate(connector)
        scores.append(collective_fitness(connector))
    return float(np.mean(scores)) if scores else 0.0


def run_deployment(agent_start=0, agent_end=99, samples=20, out_path="results.csv"):
    agent_ids = list(range(agent_start, agent_end + 1))
    rows = []

    for env_name in ["flat"] + envs.ENVIRONMENTS:
        ind = eval_independent(agent_ids, env_name, samples=samples)
        rows.append(
            {
                "collective": "independent",
                "strength": "",
                "train_pairing": "",
                "deploy_pairing": "",
                "environment": env_name,
                "mean_fitness": ind,
            }
        )
        print(f"{env_name} independent={ind:.4f}")

        for strength, train_pairing, deploy_pairing in COLLECTIVE_TYPES:
            score = eval_collective(
                strength,
                train_pairing,
                deploy_pairing,
                agent_ids,
                env_name,
                samples=samples,
            )
            label = f"{strength}/{train_pairing}->{deploy_pairing}"
            rows.append(
                {
                    "collective": label,
                    "strength": strength,
                    "train_pairing": train_pairing,
                    "deploy_pairing": deploy_pairing,
                    "environment": env_name,
                    "mean_fitness": score,
                }
            )
            print(f"{env_name} {label}={score:.4f}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "collective",
                "strength",
                "train_pairing",
                "deploy_pairing",
                "environment",
                "mean_fitness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows
