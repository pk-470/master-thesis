"""
BaseArgs: Base dataclass for arguments used to initialize objects.
Can also be viewed as a dictionary.
"""

from dataclasses import dataclass
from typing import Any, ItemsView, KeysView, ValuesView


@dataclass
class BaseArgs:
    """
    Base dataclass for arguments used to initialize objects.
    Can also be viewed as a dictionary.
    """

    def __getitem__(self, item: str) -> Any:
        return self.__dict__[item]

    def keys(self) -> KeysView:
        """Return self.__dict__ keys."""
        return self.__dict__.keys()

    def values(self) -> ValuesView:
        """Return self.__dict__ values."""
        return self.__dict__.values()

    def items(self) -> ItemsView:
        """Return self.__dict__ items."""
        return self.__dict__.items()

    def as_dict(self) -> dict[str, Any]:
        """View as dictionary."""
        return self.__dict__
