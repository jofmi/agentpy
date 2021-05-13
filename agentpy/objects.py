"""
Agentpy Objects Module
Content: Base classes for agents and environment
"""

from .sequences import AgentList
from .tools import AgentpyError, make_list


class Object:
    """ Base class for all objects of an agent-based models. """

    def __init__(self, model):
        self._var_ignore = []

        self.id = model._new_id()  # Assign id to new object
        self.type = type(self).__name__
        self.log = {}

        self.model = model
        self.p = model.p

    def __repr__(self):
        return f"{self.type} (Obj {self.id})"

    def __getattr__(self, key):
        raise AttributeError(f"{self} has no attribute '{key}'.")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def _set_var_ignore(self):
        """Store current attributes to separate them from custom variables"""
        self._var_ignore = [k for k in self.__dict__.keys() if k[0] != '_']

    @property
    def vars(self):
        return [k for k in self.__dict__.keys()
                if k[0] != '_'
                and k not in self._var_ignore]

    def record(self, var_keys, value=None):
        """ Records an object's variables at the current time-step.
        Recorded variables can be accessed via the object's `log` attribute
        and will be saved to the model's output at the end of a simulation.

        Arguments:
            var_keys (str or list of str):
                Names of the variables to be recorded.
            value (optional): Value to be recorded.
                The same value will be used for all `var_keys`.
                If none is given, the values of object attributes
                with the same name as each var_key will be used.

        Examples:

            Record the existing attributes `x` and `y` of an object `a`::

                a.record(['x', 'y'])

            Record a variable `z` with the value `1` for an object `a`::

                a.record('z', 1)

            Record all variables of an object::

                a.record(a.vars)
        """

        # Initial record call

        # Connect log to the model's dict of logs
        if self.type not in self.model._logs:
            self.model._logs[self.type] = {}
        self.model._logs[self.type][self.id] = self.log
        self.log['t'] = [self.model.t]  # Initiate time dimension

        # Perform initial recording
        for var_key in make_list(var_keys):
            v = getattr(self, var_key) if value is None else value
            self.log[var_key] = [v]

        # Set default recording function from now on
        self.record = self._record  # noqa

    def _record(self, var_keys, value=None):

        for var_key in make_list(var_keys):

            # Create empty lists
            if var_key not in self.log:
                self.log[var_key] = [None] * len(self.log['t'])

            if self.model.t != self.log['t'][-1]:

                # Create empty slot for new documented time step
                for v in self.log.values():
                    v.append(None)

                # Store time step
                self.log['t'][-1] = self.model.t

            if value is None:
                v = getattr(self, var_key)
            else:
                v = value

            self.log[var_key][-1] = v

    def setup(self, **kwargs):
        """This empty method is called automatically at the objects' creation.
        Can be overwritten in custom sub-classes
        to define initial attributes and actions.

        Arguments:
            **kwargs: Keyword arguments that have been passed to
                :class:`Agent` or :func:`Model.add_agents`.
                If the original setup method is used,
                they will be set as attributes of the object.

        Examples:
            The following setup initializes an object with three variables::

                def setup(self, y):
                    self.x = 0  # Value defined locally
                    self.y = y  # Value defined in kwargs
                    self.z = self.p.z  # Value defined in parameters
        """

        for k, v in kwargs.items():
            setattr(self, k, v)
