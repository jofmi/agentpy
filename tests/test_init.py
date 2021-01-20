import agentpy as ap
import pkg_resources


def test_version():

    assert ap.__version__ == pkg_resources.get_distribution('agentpy').version