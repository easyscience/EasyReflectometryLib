# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers
from easyscience.variable import Parameter
from easyscience.variable.parameter_dependency_resolver import resolve_all_parameter_dependencies
from scipp import DataGroup

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import load_as_dataset
from easyreflectometry.data.measurement import extract_orso_title
from easyreflectometry.data.measurement import load_data_from_orso_file
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.limits import apply_default_limits
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialCollection
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample.collections.base_collection import BaseCollection

logger = logging.getLogger(__name__)

Q_MIN = 0.001
Q_MAX = 0.3
Q_RESOLUTION = 500

DEFAULT_MINIMIZER = AvailableMinimizers.LMFit_leastsq


class Project:
    def __init__(self):
        """Init function."""
        self._info = self._default_info()
        self._path_project_parent = Path(os.path.expanduser('~'))
        self._models = ModelCollection(populate_if_none=False, unique_name='project_models')
        self._materials = MaterialCollection(populate_if_none=False, unique_name='project_materials')
        self._calculator = CalculatorFactory()
        self._experiments: Dict[DataGroup] = {}
        self._fitter: MultiFitter = None
        self._minimizer_selection: AvailableMinimizers = DEFAULT_MINIMIZER
        self._colors: list[str] = None
        self._report = None
        self._q_min: float = None
        self._q_max: float = None
        self._q_resolution: int = None
        self._current_material_index = 0
        self._current_model_index = 0
        self._current_assembly_index = 0
        self._current_layer_index = 0
        self._fitter_model_index = None
        self._current_experiment_index = 0

        # Project flags
        self._created = False
        self._with_experiments = False

    def reset(self):
        """Reset function."""
        del self._models
        del self._materials
        global_object.map._clear()

        self.__init__()

    @property
    def parameters(self) -> List[Parameter]:
        """Get all unique parameters from all models in the project.

        Parameters shared across multiple models (e.g. material SLD) are
        only included once to avoid double-counting.
        """
        parameters = []
        seen_ids: set[int] = set()
        if self._models is not None:
            for model in self._models:
                for param in model.get_all_parameters():
                    pid = id(param)
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        parameters.append(param)
        return parameters

    def _sync_parameter_states(self) -> None:
        """Apply project-level parameter enablement and default limits.

        Superphase thickness/roughness and subphase thickness are physically
        meaningless and are marked as disabled. Thickness and roughness default
        ranges are then applied only to enabled parameters created from project
        defaults, leaving explicit user-provided bounds untouched.
        """
        if self._models is None:
            return

        disabled_ids: set[int] = set()
        for model in self._models:
            sample = model.sample
            if sample is None or len(sample) == 0:
                continue
            superphase = sample.superphase
            if superphase is not None:
                disabled_ids.add(id(superphase.thickness))
                disabled_ids.add(id(superphase.roughness))
            subphase = sample.subphase
            if subphase is not None:
                disabled_ids.add(id(subphase.thickness))

        for model in self._models:
            sample = model.sample
            if sample is None or len(sample) == 0:
                continue
            for assembly in sample:
                for layer in assembly.layers:
                    self._sync_layer_parameter_state(layer.thickness, 'thickness', disabled_ids)
                    self._sync_layer_parameter_state(layer.roughness, 'roughness', disabled_ids)

    def _sync_layer_parameter_state(self, parameter: Parameter, kind: str, disabled_ids: set[int]) -> None:
        """Update a layer parameter's enabled state and pending default limits."""
        if id(parameter) in disabled_ids:
            parameter.enabled = False
            return

        if getattr(parameter, 'default_limits_pending', False):
            delattr(parameter, 'default_limits_pending')
            if getattr(parameter, 'enabled', True):
                parameter.min = -np.inf
                parameter.max = np.inf
                apply_default_limits(parameter, kind)

    @property
    def q_min(self):
        """Q min."""
        if self._q_min is None:
            return Q_MIN
        return self._q_min

    @q_min.setter
    def q_min(self, value: float) -> None:
        """Q min."""
        self._q_min = value

    @property
    def q_max(self):
        """Q max."""
        if self._q_max is None:
            return Q_MAX
        return self._q_max

    @q_max.setter
    def q_max(self, value: float) -> None:
        """Q max."""
        self._q_max = value

    @property
    def q_resolution(self):
        """Q resolution."""
        if self._q_resolution is None:
            return Q_RESOLUTION
        return self._q_resolution

    @q_resolution.setter
    def q_resolution(self, value: int) -> None:
        """Q resolution."""
        self._q_resolution = value

    @property
    def current_material_index(self) -> Optional[int]:
        """Current material index."""
        return self._current_material_index

    @current_material_index.setter
    def current_material_index(self, value: int) -> None:
        """Current material index."""
        if value < 0 or value >= len(self._materials):
            raise ValueError(f'Index {value} out of range')
        if self._current_material_index != value:
            self._current_material_index = value

    @property
    def current_model_index(self) -> Optional[int]:
        """Current model index."""
        return self._current_model_index

    @current_model_index.setter
    def current_model_index(self, value: int) -> None:
        """Current model index."""
        if value < 0 or value >= len(self._models):
            raise ValueError(f'Index {value} out of range')
        if self._current_model_index != value:
            self._current_model_index = value
            self._current_assembly_index = 0
            self._current_layer_index = 0

    @property
    def current_assembly_index(self) -> Optional[int]:
        """Current assembly index."""
        return self._current_assembly_index

    @current_assembly_index.setter
    def current_assembly_index(self, value: int) -> None:
        """Current assembly index."""
        if value < 0 or value >= len(self._models[self._current_model_index].sample):
            raise ValueError(f'Index {value} out of range')
        if self._current_assembly_index != value:
            self._current_assembly_index = value
            self._current_layer_index = 0

    @property
    def current_layer_index(self) -> Optional[int]:
        """Current layer index."""
        return self._current_layer_index

    @current_layer_index.setter
    def current_layer_index(self, value: int) -> None:
        """Current layer index."""
        if value < 0 or value >= len(self._models[self._current_model_index].sample[self._current_assembly_index].layers):
            raise ValueError(f'Index {value} out of range')
        if self._current_layer_index != value:
            self._current_layer_index = value

    @property
    def current_experiment_index(self) -> Optional[int]:
        """Current experiment index."""
        return self._current_experiment_index

    @current_experiment_index.setter
    def current_experiment_index(self, value: int) -> None:
        """Current experiment index."""
        if value < 0 or value >= len(self._experiments):
            raise ValueError(f'Index {value} out of range')
        if self._current_experiment_index != value:
            self._current_experiment_index = value
            # Resetting the model index to 0 when changing the experiment
            # self.current_model_index = 0

    @property
    def created(self) -> bool:
        """Created function."""
        return self._created

    @property
    def path(self):
        """Path function."""
        return self._path_project_parent / self._info['name']

    def set_path_project_parent(self, path: Union[Path, str]):
        """Set path project parent."""
        self._path_project_parent = Path(path)

    @property
    def models(self) -> ModelCollection:
        """Models function."""
        return self._models

    @models.setter
    def models(self, models: ModelCollection) -> None:
        """Models function."""
        self._replace_collection(models, self._models)
        # Use setter to update indicies for current model, assembly and layer
        self.current_model_index = 0
        self._materials.extend(self._get_materials_in_models())
        for model in self._models:
            model.interface = self._calculator
        self._sync_parameter_states()

    @property
    def fitter(self) -> MultiFitter:
        """Fitter function."""
        if len(self._models):
            if (self._fitter is None) or (self._fitter_model_index != self._current_model_index):
                self._fitter = MultiFitter(self._models[self._current_model_index])
                self._fitter.easy_science_multi_fitter.switch_minimizer(self._minimizer_selection)
                self._fitter_model_index = self._current_model_index
        return self._fitter

    @property
    def calculator(self) -> str:
        """Calculator function."""
        return self._calculator.current_interface_name

    @calculator.setter
    def calculator(self, calculator: str) -> None:
        """Calculator function."""
        if calculator == self._calculator.current_interface_name:
            return

        self._calculator.switch(calculator)
        self._calculator.reset_storage()

        for model in self._models:
            model.generate_bindings()

        self._fitter = None
        self._fitter_model_index = None

    @property
    def minimizer(self) -> AvailableMinimizers:
        """Minimizer function."""
        if self._fitter is not None:
            return self._fitter.easy_science_multi_fitter.minimizer.enum
        return self._minimizer_selection

    @minimizer.setter
    def minimizer(self, minimizer: AvailableMinimizers) -> None:
        """Minimizer function."""
        old_name = getattr(self._minimizer_selection, 'name', str(self._minimizer_selection))
        new_name = getattr(minimizer, 'name', str(minimizer))
        logger.info(
            'Minimizer changed from %s to %s (fitter active: %s)',
            old_name,
            new_name,
            self._fitter is not None,
        )
        self._minimizer_selection = minimizer
        if self._fitter is not None:
            self._fitter.easy_science_multi_fitter.switch_minimizer(minimizer)

    @property
    def experiments(self) -> Dict[int, DataSet1D]:
        """Experiments function."""
        return self._experiments

    @experiments.setter
    def experiments(self, experiments: Dict[int, DataSet1D]) -> None:
        """Experiments function."""
        self._experiments = experiments

    @property
    def path_json(self):
        """Path json."""
        return self.path / 'project.json'

    def get_index_air(self) -> int:
        """Get index air."""
        if 'Air' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='Air', sld=0.0, isld=0.0))
        return [material.name for material in self._materials].index('Air')

    def get_index_si(self) -> int:
        """Get index si."""
        if 'Si' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='Si', sld=2.07, isld=0.0))
        return [material.name for material in self._materials].index('Si')

    def get_index_sio2(self) -> int:
        """Get index sio2."""
        if 'SiO2' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='SiO2', sld=3.47, isld=0.0))
        return [material.name for material in self._materials].index('SiO2')

    def get_index_d2o(self) -> int:
        """Get index d2o."""
        if 'D2O' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='D2O', sld=6.36, isld=0.0))
        return [material.name for material in self._materials].index('D2O')

    def load_orso_file(self, path: Union[Path, str]) -> None:
        """Load an ORSO file and optionally create a model and a data from it."""
        from easyreflectometry.orso_utils import LoadOrso

        model, data = LoadOrso(path)
        if model is not None:
            if isinstance(model, Sample):
                model = Model(sample=model, name=model.name)
            self.models = ModelCollection([model])
        else:
            self.default_model()
        if data is not None:
            self._experiments[0] = data
            self._experiments[0].name = 'Experiment from ORSO'
            self._experiments[0].model = self.models[0]
            self._with_experiments = True
        pass

    def set_sample_from_orso(self, sample: Sample) -> None:
        """Replace the current project model collection with a single model built from an ORSO-parsed sample.

        This is a convenience helper for the ORSO import pipeline where a complete
        :class:`~easyreflectometry.sample.Sample` is constructed elsewhere.

        Parameters
        ----------
        sample : Sample
            Sample to set as the project's (single) model.

        Returns
        -------
        None
            ``None``.
        """
        model = Model(sample=sample)
        self.models = ModelCollection([model])

    def add_sample_from_orso(self, sample: Sample) -> None:
        """Add a new model with the given sample to the existing model collection.

        The created model is appended to :attr:`models`, its calculator interface is
        set to the project's current calculator, and any materials referenced in the
        sample are added to the project's material collection.

        After adding the model, :attr:`current_model_index` is updated to point to
        the newly added model.

        Parameters
        ----------
        sample : Sample
            Sample to add as a new model.

        Returns
        -------
        None
            ``None``.
        """
        if sample is None:
            raise ValueError('The ORSO file does not contain a valid sample model definition.')
        model = Model(sample=sample)
        self.models.add_model(model)
        # Set interface after adding to collection
        model.interface = self._calculator
        # Extract materials from the new model and add to project materials
        self._materials.extend(self._get_materials_from_model(model))
        self._sync_parameter_states()
        # Switch to the newly added model so its data is visible in the UI
        self.current_model_index = len(self._models) - 1

    def replace_models_from_orso(self, sample: Sample) -> None:
        """Replace all models and materials with a single model from an ORSO sample.

        All existing models and their associated materials are removed. A new
        model is created from *sample*, assigned to the project's calculator,
        and the material collection is rebuilt from the new model only.

        Parameters
        ----------
        sample : Sample
            Sample to set as the project's only model.

        Returns
        -------
        None
            ``None``.
        """
        if sample is None:
            raise ValueError('The ORSO file does not contain a valid sample model definition.')
        model = Model(sample=sample)
        if sample.name:
            model.user_data['original_name'] = sample.name  # Store original name for reference
        self.models = ModelCollection([model])
        model.interface = self._calculator
        self._materials = self._get_materials_from_model(model)
        self.current_model_index = 0

    def _get_materials_from_model(self, model: Model) -> 'MaterialCollection':
        """Get all materials from a single model's sample."""
        materials_in_model = MaterialCollection(populate_if_none=False)
        for assembly in model.sample:
            for layer in assembly.layers:
                if layer.material not in materials_in_model:
                    materials_in_model.append(layer.material)
        return materials_in_model

    def _apply_experiment_metadata(
        self,
        path: Union[Path, str],
        experiment: DataSet1D,
        fallback_name: str,
        data_group=None,
        data_key: Optional[str] = None,
    ) -> None:
        """Set experiment name from ORSO title and configure the resolution function.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the experiment data file.
        experiment : DataSet1D
            The loaded experiment dataset to configure.
        fallback_name : str
            Name to use when no ORSO title is available.
        data_group :
            Pre-loaded scipp DataGroup (avoids reloading the file). By default, None.
        data_key : Optional[str], optional
            Specific dataset key to use for title extraction (e.g. ``'R_1'``). By default, None.
        """
        # Prefer ORSO title when available (keeps UI descriptive)
        title = None
        try:
            if data_group is None:
                data_group = load_data_from_orso_file(str(path))
            if data_key is None:
                data_key = list(data_group['data'].keys())[0]
            title = extract_orso_title(data_group, data_key)
        except (KeyError, AttributeError, ValueError, IndexError):
            title = None

        if title:
            experiment.name = title
        elif not experiment.name or experiment.name == 'Series':
            experiment.name = fallback_name

    def _apply_resolution_function(
        self,
        experiment: DataSet1D,
        model: Model,
    ) -> None:
        """Set the resolution function on *model* based on variance data in *experiment*.

        Uses the measured per-point q-resolution (``Pointwise``) when the
        experiment carries q-variance data (``xe``, i.e. sQz²); otherwise
        falls back to the default 5% FWHM percentage resolution.

        Parameters
        ----------
        experiment : DataSet1D
            The experiment whose variance data drives the choice.
        model : Model
            The model whose resolution function is set.
        """
        if experiment.xe is not None and np.any(experiment.xe):
            model.resolution_function = Pointwise(q_data_points=[experiment.x, experiment.y, experiment.xe])
        else:
            model.resolution_function = PercentageFwhm(5.0)

    @staticmethod
    def _auto_set_background(experiment: DataSet1D) -> None:
        """Set the model background to the minimum y-value of the experiment data."""
        if experiment.model is not None and len(experiment.y) > 0:
            experiment.model.background = max(np.min(experiment.y), 1e-10)

    def load_new_experiment(self, path: Union[Path, str]) -> None:
        """Load new experiment."""
        new_experiment = load_as_dataset(str(path))
        new_index = len(self._experiments)

        model_index = 0
        if new_index < len(self.models):
            model_index = new_index

        self._apply_experiment_metadata(path, new_experiment, f'Experiment {new_index}')
        new_experiment.model = self.models[model_index]
        self._auto_set_background(new_experiment)
        self._experiments[new_index] = new_experiment
        self._with_experiments = True
        self._apply_resolution_function(new_experiment, self.models[model_index])

    def count_datasets_in_file(self, path: Union[Path, str]) -> int:
        """Return the number of datasets contained in the file at *path*.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the data file.

        Returns
        -------
        int
            Number of datasets found; 1 if the file cannot be introspected.
        """
        try:
            data_group = load_data_from_orso_file(str(path))
            return len(data_group['data'])
        except Exception:
            return 1

    def load_all_experiments_from_file(self, path: Union[Path, str]) -> int:
        """Load all datasets from a file as separate experiments sharing the current model.

        For a multi-dataset ORSO file (e.g. a multi-angle measurement), each dataset is
        registered as an independent experiment.  All experiments share the model that is
        currently selected.  Falls back to :meth:`load_new_experiment` for single-dataset
        files or on any loading error.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the data file.

        Returns
        -------
        int
            Number of experiments that were added.
        """
        try:
            data_group = load_data_from_orso_file(str(path))
        except Exception:
            self.load_new_experiment(path)
            return 1

        data_keys = sorted(data_group['data'].keys())
        if len(data_keys) <= 1:
            self.load_new_experiment(path)
            return 1

        model_index = self._current_model_index
        for data_key in data_keys:
            coord_key = data_key.replace('R_', 'Qz_')
            new_index = len(self._experiments)

            d = data_group['data'][data_key]
            c = data_group['coords'][coord_key]

            new_experiment = DataSet1D(
                name=f'Experiment {new_index}',
                x=c.values,
                y=d.values,
                ye=d.variances,
                xe=c.variances if c.variances is not None else None,
            )
            self._apply_experiment_metadata(
                path,
                new_experiment,
                f'Experiment {new_index}',
                data_group=data_group,
                data_key=data_key,
            )
            new_experiment.model = self.models[model_index]
            self._auto_set_background(new_experiment)
            self._experiments[new_index] = new_experiment
            self._apply_resolution_function(new_experiment, self.models[model_index])

        self._with_experiments = True
        return len(data_keys)

    def load_experiment_for_model_at_index(self, path: Union[Path, str], index: Optional[int] = 0) -> None:
        """Load experiment for model at index."""
        experiment = load_as_dataset(str(path))

        self._apply_experiment_metadata(path, experiment, f'Experiment {index}')
        experiment.model = self.models[index]
        self._auto_set_background(experiment)
        self._experiments[index] = experiment
        self._with_experiments = True
        self._apply_resolution_function(experiment, self._models[index])

    def sld_data_for_model_at_index(self, index: int = 0) -> DataSet1D:
        """Sld data for model at index."""
        self.models[index].interface = self._calculator
        sld = self.models[index].interface().sld_profile(self._models[index].unique_name)
        return DataSet1D(
            name=f'SLD for Model {index}',
            x=sld[0],
            y=sld[1],
        )

    def sample_data_for_model_at_index(self, index: int = 0, q_range: Optional[np.array] = None) -> DataSet1D:
        """Sample data for model at index."""
        original_resolution_function = self.models[index].resolution_function
        self.models[index].resolution_function = PercentageFwhm(0)
        reflectivity_data = self.model_data_for_model_at_index(index, q_range)
        self.models[index].resolution_function = original_resolution_function

        return reflectivity_data

    def model_data_for_model_at_index(self, index: int = 0, q_range: Optional[np.array] = None) -> DataSet1D:
        """Model data for model at index."""
        if q_range is None:
            q_range = np.linspace(self.q_min, self.q_max, self.q_resolution)
        self.models[index].interface = self._calculator
        reflectivity = self.models[index].interface().reflectity_profile(q_range, self._models[index].unique_name)
        return DataSet1D(
            name=f'Reflectivity for Model {index}',
            x=q_range,
            y=reflectivity,
        )

    def experimental_data_for_model_at_index(self, index: int = 0) -> DataSet1D:
        """Experimental data for model at index."""
        if index in self._experiments.keys():
            return self._experiments[index]
        else:
            raise IndexError(f'No experiment data for model at index {index}')

    def default_model(self):
        """Default model."""
        self._replace_collection(MaterialCollection(interface=self._calculator), self._materials)

        layers = [
            Layer(
                material=self._materials[0],
                thickness=0.0,
                roughness=0.0,
                name='Vacuum Layer',
                interface=self._calculator,
            ),
            Layer(
                material=self._materials[1],
                thickness=100.0,
                roughness=3.0,
                name='D2O Layer',
                interface=self._calculator,
            ),
            Layer(
                material=self._materials[2],
                thickness=0.0,
                roughness=1.2,
                name='Si Layer',
                interface=self._calculator,
            ),
        ]
        assemblies = [
            Multilayer(layers[0], name='Superphase', interface=self._calculator),
            Multilayer(layers[1], name='D2O', interface=self._calculator),
            Multilayer(layers[2], name='Subphase', interface=self._calculator),
        ]
        sample = Sample(*assemblies, interface=self._calculator)
        model = Model(sample=sample, interface=self._calculator)
        model.is_default = True
        self.models = ModelCollection([model])

    def is_default_model(self, index: int) -> bool:
        """Check if the model at the given index is a default model.

        Parameters
        ----------
        index : int
            Index of the model to check.

        Returns
        -------
        bool
            True if the model was created as a default placeholder.
        """
        if index < 0 or index >= len(self._models):
            return False

        return self._models[index].is_default

    def remove_model_at_index(self, index: int) -> None:
        """Remove the model at the given index.

        Removes the model from the model collection, removes the experiment at the
        same index (if any), and reindexes experiments above the removed index so
        model/experiment indices stay aligned.

        Adjusts the current model index if necessary.

        Parameters
        ----------
        index : int
            Index of the model to remove.

        Raises
        ------
        IndexError :
            If the index is out of range.
        ValueError :
            If trying to remove the last remaining model.
        """
        if index < 0 or index >= len(self._models):
            raise IndexError(f'Model index {index} out of range')

        if len(self._models) <= 1:
            raise ValueError('Cannot remove the last model from the project')

        # Remove the model from the collection
        self._models.pop(index)

        # Remove experiment mapped to the removed model index.
        if index in self._experiments:
            self._experiments.pop(index)

        # Reindex experiments above the removed model index to keep mapping aligned.
        reindexed_experiments: dict[int, DataSet1D] = {}
        for exp_index, experiment in sorted(self._experiments.items()):
            if exp_index > index:
                reindexed_experiments[exp_index - 1] = experiment
            else:
                reindexed_experiments[exp_index] = experiment
        self._experiments = reindexed_experiments

        # Adjust current model index if necessary
        if self._current_model_index >= len(self._models):
            self._current_model_index = len(self._models) - 1
        elif self._current_model_index > index:
            self._current_model_index -= 1

        # Reset assembly and layer indices for the new current model
        self._current_assembly_index = 0
        self._current_layer_index = 0

    def add_material(self, material: MaterialCollection) -> None:
        """Add material."""
        if material in self._materials:
            print(f'WARNING: Material {material} is already in material collection')
        else:
            self._materials.append(material)

    def remove_material(self, index: int) -> None:
        """Remove material."""
        if self._materials[index] in self._get_materials_in_models():
            print(f'ERROR: Material {self._materials[index]} is used in models')
        else:
            self._materials.pop(index)

    def _default_info(self):
        """Default info."""
        return dict(
            name='DefaultEasyReflectometryProject',
            short_description='Reflectometry, 1D',
            modified=datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
        )

    def create(self):
        """Create function."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(self.path / 'experiments')
            self._created = True
            self._timestamp_modification()
        else:
            print(f'ERROR: Directory {self.path} already exists')

    def save_as_json(self, overwrite=False):
        """Save as json."""
        if self.path_json.exists() and overwrite:
            print(f'File already exists {self.path_json}. Overwriting...')
            self.path_json.unlink()
        try:
            project_json = json.dumps(self.as_dict(include_materials_not_in_model=True), indent=4)
            self.path_json.parent.mkdir(exist_ok=True, parents=True)
            with open(self.path_json, mode='x') as file:
                file.write(project_json)
        except Exception as exception:
            print(exception)

    def load_from_json(self, path: Optional[Union[Path, str]] = None):
        """Load from json."""
        if path is None:
            path = self.path_json
        path = Path(path)
        if path.exists():
            with open(path, 'r') as file:
                project_dict = json.load(file)
                self.reset()
                self.from_dict(project_dict)
            self._path_project_parent = path.parents[1]
            self._created = True
        else:
            print(f'ERROR: File {path} does not exist')

    #: Schema version embedded in every serialized project. Bumped from 1 → 2
    #: when the sample/model classes migrated from the legacy
    #: ``easyscience.ObjBase``/``CollectionBase`` pipeline to
    #: ``ModelBase``/``EasyList``. The on-disk shape of nested objects (Layer,
    #: Material, MaterialMixture, MaterialSolvated, LayerAreaPerMolecule, etc.)
    #: changed in a way that is not backward-compatible with v1 files.
    FILE_FORMAT = 2

    def as_dict(self, include_materials_not_in_model=False):
        """As dict."""
        project_dict = {}
        project_dict['file_format'] = self.FILE_FORMAT
        project_dict['info'] = self._info
        project_dict['with_experiments'] = self._with_experiments
        if self._models is not None:
            project_dict['models'] = self._models.as_dict()
            project_dict['models']['unique_name'] = self._models.unique_name + '_to_prevent_collisions_on_load'
        if include_materials_not_in_model:
            self._as_dict_add_materials_not_in_model_dict(project_dict)
        if self._with_experiments:
            self._as_dict_add_experiments(project_dict)
        if self.fitter is not None:
            project_dict['fitter_minimizer'] = self.fitter.easy_science_multi_fitter.minimizer.name
        elif self._minimizer_selection is not None:
            project_dict['fitter_minimizer'] = self._minimizer_selection.name
        if self._calculator is not None:
            project_dict['calculator'] = self._calculator.current_interface_name
        if self._colors is not None:
            project_dict['colors'] = self._colors
        return project_dict

    def _as_dict_add_materials_not_in_model_dict(self, project_dict: dict):
        """As dict add materials not in model dict."""
        materials_not_in_model = []
        for material in self._materials:
            if material not in self._get_materials_in_models():
                materials_not_in_model.append(material)
        if len(materials_not_in_model) > 0:
            project_dict['materials_not_in_model'] = MaterialCollection(materials_not_in_model).as_dict(skip=['interface'])

    def _as_dict_add_experiments(self, project_dict: dict):
        """As dict add experiments."""
        project_dict['experiments'] = {}
        project_dict['experiments_models'] = {}
        project_dict['experiments_names'] = {}

        for key, experiment in self._experiments.items():
            project_dict['experiments'][key] = [
                list(experiment.x),
                list(experiment.y),
                list(experiment.ye),
            ]
            if experiment.xe is not None:
                project_dict['experiments'][key].append(list(experiment.xe))
                project_dict['experiments_models'][key] = experiment.model.name
                project_dict['experiments_names'][key] = experiment.name

    def from_dict(self, project_dict: dict):
        """From dict."""
        keys = list(project_dict.keys())
        # Validate file format. v1 files were written by the legacy
        # `ObjBase`/`CollectionBase` pipeline; their inner shapes (Layer,
        # Material, MaterialMixture, …) are not compatible with the v2
        # `ModelBase`/`EasyList` deserializer. Older files must be re-created.
        file_format = project_dict.get('file_format')
        if file_format is None:
            raise ValueError(
                'This project file predates file_format=2 and cannot be loaded by '
                'this version of easyreflectometry. The serialization format changed '
                'when the sample/model classes migrated from the legacy ObjBase / '
                'CollectionBase pipeline. Please re-create the project from its '
                'underlying data using the current API.'
            )
        if file_format != self.FILE_FORMAT:
            raise ValueError(
                f'Unsupported project file_format={file_format!r}; this version of '
                f'easyreflectometry only reads file_format={self.FILE_FORMAT}. Please '
                'either update easyreflectometry or re-create the project.'
            )
        self._info = project_dict['info']
        self._with_experiments = project_dict['with_experiments']
        if 'calculator' in keys:
            self._calculator.switch(project_dict['calculator'])
        if 'models' in keys:
            self.models = ModelCollection.from_dict(project_dict['models'])
        self._replace_collection(self._get_materials_in_models(), self._materials)
        if 'materials_not_in_model' in keys:
            self._materials.extend(MaterialCollection.from_dict(project_dict['materials_not_in_model']))
        if 'fitter_minimizer' in keys:
            self.minimizer = AvailableMinimizers[project_dict['fitter_minimizer']]
        else:
            self._fitter = None
        if 'experiments' in keys:
            self._experiments = self._from_dict_extract_experiments(project_dict)
        else:
            self._experiments = {}

        # Resolve any pending parameter dependencies (constraints) after all objects are loaded
        resolve_all_parameter_dependencies(self)

    def _from_dict_extract_experiments(self, project_dict: dict) -> Dict[int, DataSet1D]:
        """From dict extract experiments."""
        experiments = {}
        for key in project_dict['experiments'].keys():
            experiments[int(key)] = DataSet1D(
                name=project_dict['experiments_names'][key],
                x=project_dict['experiments'][key][0],
                y=project_dict['experiments'][key][1],
                ye=project_dict['experiments'][key][2],
                xe=project_dict['experiments'][key][3],
                model=self._models[project_dict['experiments_models'][key]],
                auto_background=False,
            )
        return experiments

    def _get_materials_in_models(self) -> MaterialCollection:
        """Get materials in models."""
        materials_in_model = MaterialCollection(populate_if_none=False)
        for model in self._models:
            for assembly in model.sample:
                for layer in assembly.layers:
                    materials_in_model.append(layer.material)
        return materials_in_model

    def _replace_collection(self, src_collection: BaseCollection, dst_collection: BaseCollection) -> None:
        """Replace collection."""
        # Clear the destination collection
        for i in range(len(dst_collection)):
            dst_collection.pop(0)

        for element in src_collection:
            dst_collection.append(element)

    def _timestamp_modification(self):
        """Timestamp modification."""
        self._info['modified'] = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
