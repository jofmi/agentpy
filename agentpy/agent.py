"""
Agentpy Agent Module
Content: Agent Classes
"""

from .objects import Object
from .sequences import AgentList
from .tools import AgentpyError, make_list


class Agent(Object):
    """ Template for an individual agent.

    Arguments:
        model (Model): The model instance.
        **kwargs: Will be forwarded to :func:`Agent.setup`.

    Attributes:
        id (int): Unique identifier of the agent.
        log (dict): Recorded variables of the agent.
        type (str): Class name of the agent.
        model (Model): The model instance.
        p (AttrDict): The model parameters.
        vars (list of str): Names of the agent's custom variables.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        self.setup(*args, **kwargs)
