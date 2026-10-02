"""
Connector training and simulation
"""

import argparse
import random
import numpy as np
import runtime as rt


def place_pair(agent1, agent2, offset=3.0):
    import taichi as ti
    import simulation as sim

    a1 = agent1.vertices.to_numpy()[0]
    a2 = agent2.vertices.to_numpy()[0]
    shift = float(a1[:, 0].max()) + offset - float(a2[:, 0].min())
    for i in range(sim.NUM_OF_AGENT_VERTICES):
        p = agent2.vertices[0, i]
        agent2.vertices[0, i] = ti.Vector([p[0] + shift, p[1]])


def refresh_rest_lengths(connector):
    import simulation as sim

    a1 = connector.agent1.vertices.to_numpy()[0]
    a2 = connector.agent2.vertices.to_numpy()[0]
    rests = []
    for i in range(sim.NUM_OF_CONNECTOR_EDGES):
        p1 = a1[connector.agent1.anchors_np[i]]
        p2 = a2[connector.agent2.anchors_np[i]]
        rests.append(float(np.linalg.norm(p2 - p1)))
    connector.resting_lengths.from_numpy(np.array(rests, dtype=np.float32))


def display(connector, t, video_manager=None, gui=None):
    import simulation as sim
    import train_agents

    agent1 = connector.agent1
    agent2 = connector.agent2
    pts1 = agent1.vertices.to_numpy()[t]
    pts2 = agent2.vertices.to_numpy()[t]
    edges1 = agent1.edges_np
    edges2 = agent2.edges_np
    com_x = 0.5 * (float(pts1[:, 0].mean()) + float(pts2[:, 0].mean()))
    view = 12.0

    gui.line(
        train_agents._world_to_gui(-view, 0.0, com_x, view),
        train_agents._world_to_gui(view, 0.0, com_x, view),
        radius=2,
        color=0xFFFFFF,
    )

    for pts, edges, color in (
        (pts1, edges1, 0x068587),
        (pts2, edges2, 0xED553B),
    ):
        for i in range(sim.NUM_OF_AGENT_EDGES):
            a, b = int(edges[i, 0]), int(edges[i, 1])
            gui.line(
                train_agents._world_to_gui(
                    float(pts[a, 0]), float(pts[a, 1]), com_x, view
                ),
                train_agents._world_to_gui(
                    float(pts[b, 0]), float(pts[b, 1]), com_x, view
                ),
                radius=3,
                color=color,
            )

    a1 = agent1.anchors_np
    a2 = agent2.anchors_np
    for i in range(sim.NUM_OF_CONNECTOR_EDGES):
        gui.line(
            train_agents._world_to_gui(
                float(pts1[a1[i], 0]), float(pts1[a1[i], 1]), com_x, view
            ),
            train_agents._world_to_gui(
                float(pts2[a2[i], 0]), float(pts2[a2[i], 1]), com_x, view
            ),
            radius=2,
            color=0xF6D55C,
        )

    if video_manager is not None:
        video_manager.write_frame(gui.get_image())
    gui.show()


def simulate(connector, with_display=False, video_manager=None, gui=None):
    import simulation as sim

    agent1 = connector.agent1
    agent2 = connector.agent2

    for t in range(sim.TIME_STEPS - 1):
        for agent in (agent1, agent2):
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

        sim.compute_connector_nn(
            t,
            agent1.vertices,
            agent1.velocities,
            agent2.vertices,
            agent2.velocities,
            agent1.anchors,
            agent2.anchors,
            connector.agent1_center,
            connector.agent2_center,
            connector.input_state,
            connector.hidden1_state,
            connector.hidden2_state,
            connector.output_state,
            connector.weights1,
            connector.weights2,
            connector.weights3,
        )
        sim.apply_connector_forces(
            t,
            agent1.vertices,
            agent1.velocities,
            agent1.forces,
            agent2.vertices,
            agent2.velocities,
            agent2.forces,
            agent1.anchors,
            agent2.anchors,
            connector.resting_lengths,
            connector.output_state,
            connector.k,
        )

        sim.apply_forces(t, agent1.vertices, agent1.velocities, agent1.forces)
        sim.apply_forces(t, agent2.vertices, agent2.velocities, agent2.forces)

        if with_display:
            display(connector, t, video_manager, gui=gui)


def watch(
    connector_id,
    agent1_id,
    agent2_id,
    connectors_directory="connectors",
    agents_directory="agents",
):
    import taichi as ti
    import simulation as sim
    import environments as envs

    connector = sim.Connector(
        connector_id,
        sim.Agent(agent1_id, directory=agents_directory),
        sim.Agent(agent2_id, directory=agents_directory),
        directory=connectors_directory,
    )
    sim.set_agent(connector.agent1)
    sim.set_agent(connector.agent2)
    place_pair(connector.agent1, connector.agent2)
    refresh_rest_lengths(connector)
    envs.set_env("flat")
    gui = ti.GUI("Sim", res=600)
    simulate(connector, with_display=True, gui=gui)


def load_connector_meta(connector_id, directory="connectors"):
    path = f"{rt.resolve_path(directory)}/connector{connector_id}.npz"
    with np.load(path) as loaded:
        return loaded["connector_type"].item(), loaded["strength"].item()


def choose_partner_id(agent_id, agent_ids, pairing):
    if pairing == "uniform":
        return agent_id
    candidates = [partner_id for partner_id in agent_ids if partner_id != agent_id]
    if not candidates:
        return agent_id
    return random.choice(candidates)


def train_connectors(
    connector_start,
    connector_end,
    agent_start,
    agent_end,
    connectors_directory="connectors",
    agents_directory="agents",
):
    import taichi as ti
    import simulation as sim
    import environments as envs

    agent_ids = list(range(agent_start, agent_end + 1))
    envs.set_env("flat")

    for connector_id in range(connector_start, connector_end + 1):
        pairing, strength = load_connector_meta(connector_id, directory=connectors_directory)
        print(f"training connector{connector_id} ({strength}/{pairing})", flush=True)

        for agent_id in agent_ids:
            partner_id = choose_partner_id(agent_id, agent_ids, pairing)
            connector = sim.Connector(
                connector_id,
                sim.Agent(agent_id, directory=agents_directory),
                sim.Agent(partner_id, directory=agents_directory),
                directory=connectors_directory,
            )

            for gen in range(sim.GENERATIONS):
                sim.set_agent(connector.agent1)
                sim.set_agent(connector.agent2)
                place_pair(connector.agent1, connector.agent2)
                refresh_rest_lengths(connector)
                with ti.ad.Tape(loss=connector.agent1.loss):
                    simulate(connector)
                    sim.compute_connector_loss(
                        connector.agent1.vertices,
                        connector.agent2.vertices,
                        connector.output_state,
                        connector.output_sum,
                        connector.output_sumsq,
                        connector.agent1.loss,
                    )
                sim.update_connector_weights(
                    connector.weights1, connector.weights2, connector.weights3, sim.agent_lr(gen)
                )

            connector.write()


def main():
    import runtime as rt

    parser = argparse.ArgumentParser(description="Train or watch connectors")
    sub = parser.add_subparsers(dest="command", required=True)

    watch_p = sub.add_parser("watch", help="GUI playback of a tethered pair")
    watch_p.add_argument("connector_id", type=int)
    watch_p.add_argument("agent1_id", type=int)
    watch_p.add_argument("agent2_id", type=int)
    watch_p.add_argument("--connectors-directory", default="connectors")
    watch_p.add_argument("--agents-directory", default="agents")
    rt.add_arch_args(watch_p)

    train_p = sub.add_parser("train", help="Train a range of connectors")
    train_p.add_argument("connector_start", type=int)
    train_p.add_argument("connector_end", type=int)
    train_p.add_argument("agent_start", type=int)
    train_p.add_argument("agent_end", type=int)
    train_p.add_argument("--connectors-directory", default="connectors")
    train_p.add_argument("--agents-directory", default="agents")
    rt.add_arch_args(train_p)

    args = parser.parse_args()
    rt.configure(arch=rt.arch_from_args(args))

    if args.command == "watch":
        watch(
            args.connector_id,
            args.agent1_id,
            args.agent2_id,
            connectors_directory=args.connectors_directory,
            agents_directory=args.agents_directory,
        )
    else:
        train_connectors(
            args.connector_start,
            args.connector_end,
            args.agent_start,
            args.agent_end,
            connectors_directory=args.connectors_directory,
            agents_directory=args.agents_directory,
        )


if __name__ == "__main__":
    main()
