import numpy as np

class Agent(object):

    # steps agent forward in time and outputs information that we want to log through time.
    def perform_timestep(self, other_agents):
        raise NotImplemented

    def safe_copy(self):
        raise NotImplemented

    def get_id(self):
        raise NotImplemented



class AgentSimulation(object):

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.agent_list = []

    # updates all agents and returns the data obtained at that time
    def timestep(self):
        old_agents = [agent.safe_copy() for agent in self.agent_list]
        all_agent_data = {}
        for agent in self.agent_list:
            data = agent.perform_timestep(old_agents)
            all_agent_data[agent.get_id()] = data
        return all_agent_data

    # stop_condition takes in (timesteps, all_agents)
    def run_simulation(self, stop_condition):
        t = 0
        all_time_data = []
        while not stop_condition(t, self.agent_list):
            agent_data = self.timestep()
            all_time_data.append(agent_data)
            t += 1
        return all_time_data


