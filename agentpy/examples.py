import agentpy as ap
import numpy as np


def gini(x):

    """ Calculate Gini Coefficient """
    # By Warren Weckesser https://stackoverflow.com/a/39513799

    x = np.array(x)
    mad = np.abs(np.subtract.outer(x, x)).mean()  # Mean absolute difference
    rmad = mad / np.mean(x)  # Relative mean absolute difference
    return 0.5 * rmad


class WealthAgent(ap.Agent):

    """ An agent with wealth """

    def setup(self):

        self.wealth = 1

    def wealth_transfer(self):

        if self.wealth > 0:

            partner = self.model.agents.random()
            partner.wealth += 1
            self.wealth -= 1


class WealthModel(ap.Model):

    """
    Demonstration model of random wealth transfers.

    See Also:
        Notebook in the model library: :doc:`agentpy_wealth_transfer`

    Arguments:
        parameters (dict):

            - agents (int): Number of agents.
            - steps (int, optional): Number of time-steps.
    """

    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, WealthAgent)

    def step(self):
        self.agents.wealth_transfer()

    def update(self):
        self.gini = gini(self.agents.wealth)
        self.record('gini')

    def end(self):
        self.report('gini')


class SegregationAgent(ap.Agent):

    def setup(self):
        """ Initiate agent attributes. """
        self.grid = self.model.grid
        self.random = self.model.random
        self.group = self.random.choice(range(self.p.n_groups))
        self.share_similar = 0
        self.happy = False

    def update_happiness(self):
        """ Be happy if rate of similar neighbors is high enough. """
        neighbors = self.grid.neighbors(self)
        similar = len([n for n in neighbors if n.group == self.group])
        ln = len(neighbors)
        self.share_similar = similar / ln if ln > 0 else 0
        self.happy = self.share_similar >= self.p.want_similar

    def find_new_home(self):
        """ Move to random free spot and update free spots. """
        new_spot = self.random.choice(self.model.grid.empty)
        self.grid.move_to(self, new_spot)


class SegregationModel(ap.Model):
    """
    Demonstration model of segregation dynamics.

    See Also:
        Notebook in the model library: :doc:`agentpy_segregation`

    Arguments:
        parameters (dict):

            - want_similar (float):
              Percentage of similar neighbors
              for agents to be happy
            - n_groups (int): Number of groups
            - density (float): Density of population
            - size (int): Height and length of the grid
            - steps (int, optional): Maximum number of steps
    """

    def setup(self):

        # Parameters
        s = self.p.size
        n = self.n = int(self.p.density * (s ** 2))

        # Create grid and agents
        self.grid = ap.Grid(self, (s, s), track_empty=True)
        self.agents = ap.AgentList(self, n, SegregationAgent)
        self.grid.add_agents(self.agents, random=True, empty=True)

    def update(self):
        # Update list of unhappy people
        self.agents.update_happiness()
        self.unhappy = self.agents.select(self.agents.happy == False)

        # Stop simulation if all are happy
        if len(self.unhappy) == 0:
            self.stop()

    def step(self):
        # Move unhappy people to new location
        self.unhappy.find_new_home()

    def get_segregation(self):
        # Calculate average percentage of similar neighbors
        return round(sum(self.agents.share_similar) / self.n, 2)

    def end(self):
        # Measure segregation at the end of the simulation
        self.report('segregation', self.get_segregation())