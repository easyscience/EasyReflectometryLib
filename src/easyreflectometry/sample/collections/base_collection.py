from typing import List
from typing import Optional

from easyscience import global_object
from easyscience.base_classes import CollectionBase as EasyBaseCollection

from easyreflectometry.utils import yaml_dump


class BaseCollection(EasyBaseCollection):
    def __init__(
        self,
        name: str,
        interface,
        *args,
        unique_name: Optional[str] = None,
        **kwargs,
    ):
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        super().__init__(name, unique_name=unique_name, *args, **kwargs)
        self.interface = interface

        # Needed to ensure an empty list is created when saving and instatiating the object as_dict -> from_dict
        # Else collisions might occur in global_object.map
        self.populate_if_none = False

    def __repr__(self) -> str:
        """
        String representation of the collection.

        :return: a string representation of the collection
        """
        return yaml_dump(self._dict_repr)

    @property
    def names(self) -> list:
        """
        :returns: list of names for the elements in the collection.
        """
        return [i.name for i in self]

    def move_up(self, index: int):
        """Move the element at the given index up in the collection.

        :param index: Index of the element to move up.
        """
        if index == 0:
            return
        self.insert(index - 1, self.pop(index))

    def move_down(self, index: int):
        """Move the element at the given index down in the collection.

        :param index: Index of the element to move down.
        """
        if index == len(self) - 1:
            return
        self.insert(index + 1, self.pop(index))

    def remove(self, index: int):
        """
        Remove an element from the elements.

        :param index: Index of the element to remove
        """
        self.pop(index)

    @property
    def _dict_repr(self) -> dict:
        """
        A simplified dict representation.

        :return: Simple dictionary
        """
        return {self.name: [i._dict_repr for i in self]}

    def as_dict(self, skip: Optional[List[str]] = None) -> dict:
        """
        Create a dictionary representation of the collection.

        :return: A dictionary representation of the collection
        """
        if skip is None:
            skip = []
        this_dict = super().as_dict(skip=skip)
        this_dict['data'] = []
        for collection_element in self:
            this_dict['data'].append(collection_element.as_dict(skip=skip))
        this_dict['populate_if_none'] = self.populate_if_none
        return this_dict

    def __deepcopy__(self, memo):
        return self.from_dict(self.as_dict(skip=['unique_name']))
