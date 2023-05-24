"""Utility functions and classes."""


class DataStorage(dict):
    """A dictionary that can be accessed with dot notation."""

    def __new__(cls, *args, **kwargs):
        """Create a new instance of DataStorage."""
        obj = super().__new__(cls, *args, **kwargs)
        obj.__dict__ = obj
        return obj
