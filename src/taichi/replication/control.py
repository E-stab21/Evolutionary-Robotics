"""
Main expirement control file
"""

# imports
import generate
import simulation
import train_agents
import train_connectors


def main():
    num_of_agents = 1
    agent_start = 0
    agent_end = num_of_agents - 1

    generate.generate_agents(num_of_agents)
    train_agents.train_agents(agent_start, agent_end)

    for agent_id in range(agent_start, agent_end + 1):
        train_agents.watch(agent_id)


if __name__ == "__main__":
    main()
