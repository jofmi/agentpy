"""
Agentpy Tools Module
Content: Errors, generators, and base classes
"""

from numpy import ndarray


class AgentpyError(Exception):
    pass


def make_matrix(shape, class_):
    """ Returns a nested list with given shape and class instance. """

    # H/T Thierry Lathuille https://stackoverflow.com/a/64467230/

    if len(shape) == 1:
        return [class_() for _ in range(shape[0])]
    return [make_matrix(shape[1:], class_) for _ in range(shape[0])]


def make_list(element, keep_none=False):
    """ Turns element into a list of itself
    if it is not of type list or tuple. """

    if element is None and not keep_none:
        element = []  # Convert none to empty list
    if not isinstance(element, (list, tuple, ndarray)):
        element = [element]
    elif isinstance(element, tuple):
        element = list(element)

    return element


def param_tuples_to_salib(param_ranges_tuples):
    """ Convert param_ranges to SALib Format """

    param_ranges_salib = {
        'num_vars': len(param_ranges_tuples),
        'names': list(param_ranges_tuples.keys()),
        'bounds': []
    }

    for var_key, var_range in param_ranges_tuples.items():
        param_ranges_salib['bounds'].append([var_range[0], var_range[1]])

    return param_ranges_salib


class AttrDict(dict):
    """ Dictionary where attribute calls are handled like item calls.

    Examples:

        >>> ad = ap.AttrDict()
        >>> ad['a'] = 1
        >>> ad.a
        1

        >>> ad.b = 2
        >>> ad['b']
        2
    """

    def __init__(self, *args, **kwargs):
        if args == (None, ):
            args = ()  # Empty tuple
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError:
            # Important for pickle to work
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self.__setitem__(name, value)

    def __delattr__(self, item):
        del self[item]

    def __repr__(self):
        return f"AttrDict {super().__repr__()}"
