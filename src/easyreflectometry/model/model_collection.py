from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple

from easyreflectometry.model.model import COLORS
from easyreflectometry.sample.collections.base_collection import BaseCollection

from .model import Model


# Needs to be a function, elements are added to the global_object.map
def DEFAULT_ELEMENTS(interface):
    return (Model(interface),)


class ModelCollection(BaseCollection):
    def __init__(
        self,
        *models: Tuple[Model],
        name: str = 'Models',
        interface=None,
        unique_name: Optional[str] = None,
        populate_if_none: bool = True,
        next_color_index: Optional[int] = None,
        **kwargs,
    ):
        if not models:
            if populate_if_none:
                models = DEFAULT_ELEMENTS(interface)
            else:
                models = []
        # Needed to ensure an empty list is created when saving and instatiating the object as_dict -> from_dict
        # Else collisions might occur in global_object.map
        self.populate_if_none = False
        self._next_color_index = next_color_index

        super().__init__(name, interface, *models, unique_name=unique_name, **kwargs)

        color_count = len(COLORS)
        if color_count == 0:
            self._next_color_index = 0
        elif self._next_color_index is None:
            self._next_color_index = len(self) % color_count
        else:
            self._next_color_index %= color_count

    def add_model(self, model: Optional[Model] = None):
        """Add a model to the collection.

        :param model: Model to add.
        """
        if model is None:
            model = Model(name='Model', interface=self.interface, color=self._current_color())
        self.append(model)

    def duplicate_model(self, index: int):
        """Duplicate a model in the collection.

        :param index: Model to duplicate.
        """
        to_be_duplicated = self[index]
        duplicate = Model.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        duplicate.name = duplicate.name + ' duplicate'
        self.append(duplicate)

    def as_dict(self, skip: List[str] | None = None) -> dict:
        this_dict = super().as_dict(skip=skip)
        this_dict['populate_if_none'] = self.populate_if_none
        this_dict['next_color_index'] = self._next_color_index
        return this_dict

    @classmethod
    def from_dict(cls, this_dict: dict) -> ModelCollection:
        """
        Create an instance of a collection from a dictionary.

        :param data: The dictionary for the collection
        """
        collection_dict = this_dict.copy()
        # We need to call from_dict on the base class to get the models
        dict_data = collection_dict.pop('data')
        next_color_index = collection_dict.pop('next_color_index', None)

        collection = super().from_dict(collection_dict)  # type: ModelCollection

        for model_data in dict_data:
            collection._append_internal(Model.from_dict(model_data), advance=False)

        if len(collection) != len(this_dict['data']):
            raise ValueError(f'Expected {len(collection)} models, got {len(this_dict["data"])}')

        color_count = len(COLORS)
        if color_count == 0:
            collection._next_color_index = 0
        elif next_color_index is None:
            collection._next_color_index = len(collection) % color_count
        else:
            collection._next_color_index = next_color_index % color_count

        return collection

    def append(self, model: Model) -> None:  # type: ignore[override]
        self._append_internal(model, advance=True)

    def _append_internal(self, model: Model, advance: bool) -> None:
        super().append(model)
        if advance:
            self._advance_color_index()

    def _advance_color_index(self) -> None:
        if not COLORS:
            self._next_color_index = 0
            return
        if self._next_color_index is None:
            self._next_color_index = len(self) % len(COLORS)
            return
        self._next_color_index = (self._next_color_index + 1) % len(COLORS)

    def _current_color(self) -> str:
        if not COLORS:
            raise ValueError('No colors defined for models.')
        if self._next_color_index is None:
            self._next_color_index = 0
        return COLORS[self._next_color_index]
