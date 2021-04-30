try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python
    import importlib_metadata as metadata  # noqa

__version__ = metadata.version('agentpy')
