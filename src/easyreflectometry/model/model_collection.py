from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple

from easyreflectometry.model.resolution_functions import PercentageFwhm
from easyreflectometry.sample.collections.base_collection import BaseCollection

from .model import Model


# Needs to be a function, elements are added to the global_object.map
def DEFAULT_ELEMENTS(interface):
    return (Model(interface),)


class ModelCollection(BaseCollection):
    def __init__(
        self,
        *models: Tuple[Model],
        name: str = 'EasyModels',
        interface=None,
        unique_name: Optional[str] = None,
        populate_if_none: bool = True,
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

        super().__init__(name, interface, unique_name=unique_name, *models, **kwargs)

    def add_model(self, model: Optional[Model] = None):
        """Add a model to the collection.

        :param model: Model to add.
        """
        if model is None:
            model = Model(
            sample=None,
            scale=1,
            background=0,
            resolution_function=PercentageFwhm,
            interface=self.interface,
            name='New EasyModel',
            )
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
        return this_dict

    @classmethod
    def from_dict(cls, this_dict: dict) -> ModelCollection:
        """
        Create an instance of a collection from a dictionary.

        :param data: The dictionary for the collection
        """
        collection_dict = this_dict.copy()
        # We neeed to call from_dict on the base class to get the models
        dict_data = collection_dict['data']
        del collection_dict['data']

        collection = super().from_dict(collection_dict)  # type: ModelCollection

        for model_data in dict_data:
            collection.add_model(Model.from_dict(model_data))

        if len(collection) != len(this_dict['data']):
            raise ValueError(f"Expected {len(collection)} models, got {len(this_dict['data'])}")

        return collection
