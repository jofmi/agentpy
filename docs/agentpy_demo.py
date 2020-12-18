import agentpy as ap





class MoneyAgent(ap.Agent):

    def setup(self):
        self.wealth = 1

    def wealth_transfer(self):
        if self.wealth == 0:
            return
        a = self.model.agents.random()
        a.wealth += 1
        self.wealth -= 1



class MoneyModel(ap.Model):

    def setup(self):
        self.add_agents(
            self.p.agents, MoneyAgent)

    def step(self):
        self.agents.record('wealth')
        self.agents.wealth_transfer()










# Perform single run
parameters = {'agents': 10, 'steps': 10}
model = MoneyModel(parameters)
results = model.run()

# Perform multiple runs
parameters['agents'] = (10, 500, int)
sample = ap.sample(parameters, n=49)

exp = ap.Experiment(
    MoneyModel,
    sample,
    iterations=5,
    record=True
)


results = exp.run()