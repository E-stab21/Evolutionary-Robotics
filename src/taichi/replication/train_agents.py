"""
Agent training and simulation
"""

# imports
import taichi as ti
import sys
import simulation as sim


gui = ti.GUI("Sim", res=600)


def display(agent, t, video_manager=None):
    gui.line([0.0, 0.0], [1.0, 0.0], radius=2, color=0xFFFFFF)
    for i in range(sim.NUM_OF_AGENT_EDGES):
        a, b = agent.edges[i]
        gui.line(
            [agent.vertices[t, a][0] / sim.SCALE, agent.vertices[t, a][1] / sim.SCALE],
            [agent.vertices[t, b][0] / sim.SCALE, agent.vertices[t, b][1] / sim.SCALE],
            radius=3,
            color=0x068587,
        )
    if video_manager is not None:
        video_manager.write_frame(gui.get_image())
    gui.show()


def simulate(agent, with_display=False, video_manager=None):
    for t in range(sim.TIME_STEPS - 1):
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
        sim.apply_forces(t, agent.vertices, agent.velocities, agent.forces)
        if with_display:
            display(agent, t, video_manager)


def save_video(agent, with_display=False):
    video_manager = ti.tools.VideoManager(
        output_dir="/home/ethan/Projects/Evolutionary-Robotics/vids",
        framerate=60,
        automatic_build=True,
    )
    simulate(agent, with_display=with_display, video_manager=video_manager)
    print("Exporting video...")
    video_manager.make_video(gif=False, mp4=True)


def watch(agent_id):
    agent = sim.Agent(agent_id)
    sim.set_agent(agent)
    simulate(agent, with_display=True)


def train_agents(agent_start, agent_end):
    for agent_id in range(agent_start, agent_end + 1):
        agent = sim.Agent(agent_id)

        for _ in range(sim.GENERATIONS):
            sim.set_agent(agent)
            with ti.ad.Tape(loss=agent.loss):
                simulate(agent)
                sim.compute_loss(agent.vertices, agent.vertices, agent.loss)
            sim.update_agent_weights(agent.weights1, agent.weights2, agent.weights3)

        agent.write()


def main():
    agent_start = int(sys.argv[2])
    agent_end = int(sys.argv[3])
    train_agents(agent_start, agent_end)


if __name__ == "__main__":
    main()
