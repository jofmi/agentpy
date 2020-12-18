from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.batchrunner import BatchRunner
from mesa.datacollection \
    import DataCollector

class MoneyAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1

    def step(self):
        if self.wealth == 0:
            return
        other_agent = self.random.choice(
            self.model.schedule.agents)
        other_agent.wealth += 1
        self.wealth -= 1

class MoneyModel(Model):

    def __init__(self, N):
        self.running = True
        self.num_agents = N
        self.schedule = \
            RandomActivation(self)
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

        self.collector = DataCollector(
            agent_reporters={
                "Wealth": "wealth"})

    def step(self):
        self.collector.collect(self)
        self.schedule.step()

# Perform single run
model = MoneyModel(10)
for i in range(10):
    model.step()

# Perform multiple runs
variable_params = {
    "N": range(10, 500, 10)}

batch_run = BatchRunner(
    MoneyModel,
    variable_params,
    iterations=5,
    max_steps=10,
    agent_reporters={"Wealth": "wealth"}
)

batch_run.run_all()