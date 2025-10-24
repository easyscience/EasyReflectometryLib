from abc import abstractmethod

from easyscience import ObjBase as BaseObj

from easyreflectometry.utils import yaml_dump


class BaseCore(BaseObj):
    def __init__(
        self,
        name: str,
        interface,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Updates interface using property in base object
        self.interface = interface

    @abstractmethod
    def _dict_repr(self) -> dict[str, str]: ...

    def __repr__(self) -> str:
        """
        String representation of the layer.

        :return: a string representation of the layer
        :rtype: str
        """
        return yaml_dump(self._dict_repr)

    # For classes with special serialization needs one must adopt the dict produced by super
    # def as_dict(self, skip: list = None) -> dict:
    #    """Should produce a cleaned dict that matches the parameters in __init__
    #
    #    :param skip: List of keys to skip, defaults to `None`.
    #    """
    #    if skip is None:
    #        skip = []
    #    this_dict = super().as_dict(skip=skip)
    #    ...
    #    Correct the dict here
    #    ...
    #    return this_dict
