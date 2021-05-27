# Model design
import agentpy as ap
import numpy as np

# Visualization
import seaborn as sns


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
    A simple model of random wealth transfers.

    Parameters:
        agents (int): Number of agents.

    Recorded variables:
        gini: Gini coefficient during each time-step.

    Reporters:
        gini: Gini coefficient at the end of the simulation.
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

