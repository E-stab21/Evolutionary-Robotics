"""
Connector training and simulation
"""

# imports
import random
import taichi as ti
import numpy as np
import sys
import simulation as sim


# globals
gui = ti.GUI("Sim", res=600)


def display(connector, t, video_manager=None):
    agent1 = connector.agent1
    agent2 = connector.agent2

    gui.line([0.0, 0.0], [1.0, 0.0], radius=2, color=0xFFFFFF)

    for agent, color in ((agent1, 0x068587), (agent2, 0xED553B)):
        for i in range(sim.NUM_OF_AGENT_EDGES):
            a, b = agent.edges[i]
            gui.line(
                [
                    agent.vertices[t, a][0] / sim.SCALE,
                    agent.vertices[t, a][1] / sim.SCALE,
                ],
                [
                    agent.vertices[t, b][0] / sim.SCALE,
                    agent.vertices[t, b][1] / sim.SCALE,
                ],
                radius=3,
                color=color,
            )

    for i in range(sim.NUM_OF_CONNECTOR_EDGES):
        gui.line(
            [
                agent1.vertices[t, i][0] / sim.SCALE,
                agent1.vertices[t, i][1] / sim.SCALE,
            ],
            [
                agent2.vertices[t, i][0] / sim.SCALE,
                agent2.vertices[t, i][1] / sim.SCALE,
            ],
            radius=2,
            color=0xF6D55C,
        )

    if video_manager is not None:
        video_manager.write_frame(gui.get_image())
    gui.show()


def simulate(connector, with_display=False, video_manager=None):
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

        sim.compute_connector_motor_forces(
            t,
            agent1.vertices,
            agent1.velocities,
            agent1.forces,
            agent2.vertices,
            agent2.velocities,
            agent2.forces,
            connector.weights1,
            connector.weights2,
        )

        sim.apply_forces(t, agent1.vertices, agent1.velocities, agent1.forces)
        sim.apply_forces(t, agent2.vertices, agent2.velocities, agent2.forces)

        if with_display:
            display(connector, t, video_manager)


def save_video(connector, with_display=False):
    video_manager = ti.tools.VideoManager(
        output_dir="/home/ethan/Projects/Evolutionary-Robotics/vids",
        framerate=60,
        automatic_build=True,
    )
    simulate(connector, with_display=with_display, video_manager=video_manager)
    print("Exporting video...")
    video_manager.make_video(gif=False, mp4=True)


def watch(connector_id, agent1_id, agent2_id):
    connector = sim.Connector(connector_id, sim.Agent(agent1_id), sim.Agent(agent2_id))
    sim.set_agent(connector.agent1)
    sim.set_agent(connector.agent2)
    simulate(connector, with_display=True)


def load_connector_type(connector_id):
    with np.load(f"connectors/connector{connector_id}.npz") as loaded:
        return loaded["connector_type"].item()


def choose_partner_id(agent_id, agent_ids, connector_type):
    if connector_type == "uniform":
        return agent_id

    candidates = [partner_id for partner_id in agent_ids if partner_id != agent_id]
    if not candidates:
        return agent_id

    return random.choice(candidates)


def train_connectors(connector_start, connector_end, agent_start, agent_end):
    agent_ids = list(range(agent_start, agent_end + 1))

    for connector_id in range(connector_start, connector_end + 1):
        connector_type = load_connector_type(connector_id)

        for agent_id in agent_ids:
            partner_id = choose_partner_id(agent_id, agent_ids, connector_type)
            connector = sim.Connector(
                connector_id, sim.Agent(agent_id), sim.Agent(partner_id)
            )

            for _ in range(sim.GENERATIONS):
                sim.set_agent(connector.agent1)
                sim.set_agent(connector.agent2)
                with ti.ad.Tape(loss=connector.agent1.loss):
                    simulate(connector)
                    sim.compute_loss(
                        connector.agent1.vertices,
                        connector.agent2.vertices,
                        connector.agent1.loss,
                    )
                sim.update_connector_weights(connector.weights1, connector.weights2)

            connector.write()


def main():
    connector_start = int(sys.argv[2])
    connector_end = int(sys.argv[3])
    agent_start = int(sys.argv[4])
    agent_end = int(sys.argv[5])
    train_connectors(connector_start, connector_end, agent_start, agent_end)


if __name__ == "__main__":
    main()
