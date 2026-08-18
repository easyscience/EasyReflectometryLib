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
from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.calculators.calculator_base import CalculatorBase
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.data import detect_polarization_channel
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

# Guide-field angle A in degrees, used to project the moment onto the neutron
# quantisation axis for the spin-up/spin-down potentials and nothing else.
# refl1d's default (`Aguide = 270`), which is what every model the library can
# currently build uses — `Aguide` is not exposed as a parameter yet. Keep this
# the single source of the value: exposing it later is a change here only.
GUIDE_FIELD_ANGLE = 270.0

# Points whose spin-asymmetry denominator (R⁺⁺ + R⁻⁻) is smaller than this
# multiple of its own uncertainty are dropped: there SA is noise divided by
# noise and would swamp the axis with ±values of no physical meaning.
SPIN_ASYMMETRY_SIGNIFICANCE = 3.0

# Points whose spin-asymmetry denominator (R⁺⁺ + R⁻⁻) is smaller than this
# fraction of |R⁺⁺| + |R⁻⁻| are dropped too: there the sum is what is left after
# cancellation between the two channels, so SA is a ratio of rounding noise.
# Unlike the significance rule above this needs no uncertainties, which is what
# keeps a two-column file from putting ±10³ values on the axis.
SPIN_ASYMMETRY_CANCELLATION_FRACTION = 1e-3

# A depth whose magnetic SLD is below this fraction of the largest one in the
# profile carries no moment worth speaking of, and its moment *angle* is
# meaningless — the direction of a (nearly) zero-length vector. Used to restrict
# the reported theta_m profile; a relative floor because it has to work both for
# a weak 0.1 and a strong 5 (1e-6 A^-2) moment, and because the interface
# roughness leaves a small erf tail everywhere.
MAGNETIC_MOMENT_FLOOR_FRACTION = 0.01

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
                    magnetism = getattr(layer, 'magnetism', None)
                    if magnetism is not None:
                        # Magnetic parameters exist only on magnetic layers, so
                        # they are never in `disabled_ids`; theta_m carries
                        # explicit 0-360 bounds and needs no default window.
                        self._sync_layer_parameter_state(magnetism.rho_m, 'rho_m', disabled_ids)

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

        # The cached fitter holds fit functions bound to the previous backend.
        self._fitter = None
        self._fitter_model_index = None

    @property
    def calculator_supports_magnetism(self) -> bool:
        """Whether the active calculator can model magnetic samples.

        The GUI uses this to gate magnetism-related controls (e.g. when the
        refnx or bornagain backend is selected).
        """
        return self._calculator().supports_magnetism

    @property
    def calculators_supporting_magnetism(self) -> List[str]:
        """Names of the available calculators that can model magnetic samples.

        Lets an application offer the switch a magnetic sample needs ("this
        requires refl1d — change to it?") instead of only reporting that the
        current calculator cannot do it. The active calculator is not touched.
        """
        supporting = []
        for name in self._calculator.available_interfaces:
            calculator = next(
                (candidate for candidate in CalculatorBase._calculators if candidate.name == name),
                None,
            )
            if calculator is not None and calculator().supports_magnetism:
                supporting.append(name)
        return supporting

    @property
    def models_have_magnetism(self) -> bool:
        """Whether any model in the project carries a magnetic layer.

        A calculator without magnetism cannot be selected while this holds — the
        binding it would have to build does not exist.
        """
        return any(model.has_magnetism for model in self._models)

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
    def experiments(self) -> Dict[int, Union[DataSet1D, PolarizedDataSet]]:
        """Experiments function."""
        return self._experiments

    @experiments.setter
    def experiments(self, experiments: Dict[int, Union[DataSet1D, PolarizedDataSet]]) -> None:
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

    def suggest_polarized_channel_assignment(self, paths: List[Union[Path, str]]) -> Dict[str, Optional[PolarizationChannel]]:
        """Suggest a spin-channel assignment for a set of data files.

        For each file, the ORSO header polarization is used when present, falling
        back to filename heuristics ('_uu'/'_up' → pp, '_dd'/'_down' → mm, ...).
        Files that cannot be identified map to None; a GUI should present the
        result as an editable file → channel table.

        Parameters
        ----------
        paths : List[Union[Path, str]]
            Paths of the per-channel data files.

        Returns
        -------
        Dict[str, Optional[PolarizationChannel]]
            Detected channel (or None) per path.
        """
        return {str(path): detect_polarization_channel(str(path)) for path in paths}

    def load_polarized_experiment(
        self,
        paths: Dict[Union[PolarizationChannel, str], Union[Path, str]],
        model_index: Optional[int] = None,
    ) -> int:
        """Load a polarized experiment from one data file per spin channel.

        The channel datasets form a single :class:`PolarizedDataSet` experiment
        sharing one model. Two-channel NSF-only experiments simply pass 'pp' and
        'mm' entries; spin-flip channels are optional.

        Parameters
        ----------
        paths : Dict[Union[PolarizationChannel, str], Union[Path, str]]
            Explicit channel → file assignment (e.g. from the GUI import dialog,
            pre-filled via :meth:`suggest_polarized_channel_assignment`).
        model_index : Optional[int], optional
            Index of the model the experiment belongs to. By default, the
            current model.

        Returns
        -------
        int
            Index of the newly loaded experiment, so a caller can make it
            current.
        """
        paths = {PolarizationChannel(channel): path for channel, path in paths.items()}
        channels = {}
        for channel, path in paths.items():
            # One file per channel means one dataset per file; a multi-dataset ORSO
            # file has no defined channel-to-dataset assignment here.
            if self.count_datasets_in_file(path) > 1:
                raise ValueError(
                    f"File '{path}' contains multiple datasets; polarized loading requires one dataset "
                    f'per channel file. Export the {channel.value} channel to its own file.'
                )
            dataset = load_as_dataset(str(path))
            # Keep the source file visible per channel (file → channel provenance).
            dataset.name = f'{channel.value}: {Path(path).name}'
            channels[channel] = dataset

        new_index = len(self._experiments)
        if model_index is None:
            model_index = self._current_model_index
        model = self.models[model_index]

        experiment = PolarizedDataSet(
            name=f'Polarized experiment {new_index}',
            channels=channels,
            model=model,
        )
        # Name from the ORSO title of the first file, when available.
        first_channel = experiment.available_channels[0]
        first_path = paths[first_channel]
        self._apply_experiment_metadata(first_path, experiment, f'Polarized experiment {new_index}')

        # Background and resolution follow the first (in canonical order) channel;
        # per-channel resolution functions are not supported (one per experiment).
        first_dataset = experiment[first_channel]
        self._auto_set_background(first_dataset)
        self._apply_resolution_function(first_dataset, model)

        self._experiments[new_index] = experiment
        self._with_experiments = True
        return new_index

    def load_experiment_for_model_at_index(self, path: Union[Path, str], index: Optional[int] = 0) -> None:
        """Load experiment for model at index."""
        experiment = load_as_dataset(str(path))

        self._apply_experiment_metadata(path, experiment, f'Experiment {index}')
        experiment.model = self.models[index]
        self._auto_set_background(experiment)
        self._experiments[index] = experiment
        self._with_experiments = True
        self._apply_resolution_function(experiment, self._models[index])

    def _bind_calculator(self, model) -> None:
        """Bind the project's calculator to a model unless it already is.

        Reassigning ``model.interface`` re-propagates the interface over the
        whole sample tree — an expensive rebuild. The plot getters run on every
        chart refresh, so an already-bound model must be left alone; engine
        switches go through the ``calculator`` setter, which regenerates the
        bindings itself.
        """
        if model.interface is not self._calculator:
            model.interface = self._calculator

    def sld_data_for_model_at_index(self, index: int = 0) -> DataSet1D:
        """Sld data for model at index."""
        self._bind_calculator(self.models[index])
        sld = self.models[index].interface().sld_profile(self._models[index].unique_name)
        return DataSet1D(
            name=f'SLD for Model {index}',
            x=sld[0],
            y=sld[1],
        )

    def model_has_magnetism_at_index(self, index: int = 0) -> bool:
        """Whether the model at index carries magnetism (False when there is no such model)."""
        try:
            return bool(self.models[index].has_magnetism)
        except (IndexError, AttributeError):
            return False

    def magnetic_sld_data_for_model_at_index(self, index: int = 0) -> Dict[str, DataSet1D]:
        """Nuclear and magnetic depth profiles of a magnetic model.

        Parameters
        ----------
        index : int
            Index of the model.

        Returns
        -------
        Dict[str, DataSet1D]
            Profiles versus depth z, keyed:

            - ``'sld'``: nuclear ρ(z), the same curve as
              :meth:`sld_data_for_model_at_index`;
            - ``'rho_m'``: magnetic SLD ρM(z);
            - ``'theta_m'``: in-plane moment angle θM(z) in degrees, restricted
              to the depths that carry a moment — the angle of a zero-length
              vector is meaningless, so points below
              :data:`MAGNETIC_MOMENT_FLOOR_FRACTION` of the largest ρM are left
              out instead of drawing an arbitrary angle through vacuum;
            - ``'spin_up'`` / ``'spin_down'``: the potentials each spin state
              sees, ρ(z) ± ρM(z)·cos(θM(z) − A), where A is the guide-field
              angle :data:`GUIDE_FIELD_ANGLE`.

        Raises
        ------
        ValueError
            The model has no magnetic layer, so there is no magnetic profile.
        NotImplementedError
            The active calculator cannot model magnetism.
        """
        model = self.models[index]
        if not model.has_magnetism:
            raise ValueError(
                f'Model {index} has no magnetic layer; there is no magnetic SLD profile to show. '
                'Attach magnetism to a layer first.'
            )
        self._bind_calculator(model)
        z, sld, rho_m, theta_m = model.interface().magnetic_sld_profile(model.unique_name)
        z = np.asarray(z, dtype=float)
        sld = np.asarray(sld, dtype=float)
        rho_m = np.asarray(rho_m, dtype=float)
        theta_m = np.asarray(theta_m, dtype=float)
        # Component of the moment along the guide field: what the neutron spin
        # states add to / subtract from the nuclear potential.
        projection = rho_m * np.cos(np.radians(theta_m - GUIDE_FIELD_ANGLE))
        # Only report the angle where there is a moment to have an angle, and
        # make it continuous along z: 359° followed by 1° is a 2° turn, but a
        # plotted line through the wrapped values sweeps the whole circle.
        magnitude = np.abs(rho_m)
        has_moment = magnitude > MAGNETIC_MOMENT_FLOOR_FRACTION * magnitude.max(initial=0.0)
        theta_display = _unwrapped_angle(theta_m, has_moment)
        return {
            'sld': DataSet1D(name=f'SLD for Model {index}', x=z, y=sld),
            'rho_m': DataSet1D(name=f'Magnetic SLD for Model {index}', x=z, y=rho_m),
            'theta_m': DataSet1D(name=f'Moment angle for Model {index}', x=z[has_moment], y=theta_display[has_moment]),
            'spin_up': DataSet1D(name=f'Spin-up potential for Model {index}', x=z, y=sld + projection),
            'spin_down': DataSet1D(name=f'Spin-down potential for Model {index}', x=z, y=sld - projection),
        }

    def sample_data_for_model_at_index(self, index: int = 0, q_range: Optional[np.array] = None) -> DataSet1D:
        """Sample data for model at index."""
        original_resolution_function = self.models[index].resolution_function
        self.models[index].resolution_function = PercentageFwhm(0)
        reflectivity_data = self.model_data_for_model_at_index(index, q_range)
        self.models[index].resolution_function = original_resolution_function

        return reflectivity_data

    def model_data_for_model_at_index(
        self,
        index: int = 0,
        q_range: Optional[np.array] = None,
        channel: Optional[Union[PolarizationChannel, str]] = None,
    ) -> DataSet1D:
        """Model data for model at index.

        Parameters
        ----------
        index : int
            Index of the model.
        q_range : Optional[np.array]
            Points to calculate at; the project q range by default.
        channel : Optional[Union[PolarizationChannel, str]]
            Spin cross-section to calculate. `None` (the default) gives the
            ordinary, channel-agnostic reflectivity. A channel requires a
            magnetic model (except 'pp', which falls back to the unpolarized
            calculation) and a calculator supporting magnetism, otherwise the
            calculator raises.
        """
        if q_range is None:
            q_range = np.linspace(self.q_min, self.q_max, self.q_resolution)
        self._bind_calculator(self.models[index])
        if channel is None:
            reflectivity = self.models[index].interface().reflectity_profile(q_range, self._models[index].unique_name)
            name = f'Reflectivity for Model {index}'
        else:
            channel = PolarizationChannel(channel)
            reflectivity = (
                self.models[index].interface().reflectivity_profile_channel(q_range, self._models[index].unique_name, channel)
            )
            name = f'Reflectivity ({channel.value}) for Model {index}'
        return DataSet1D(
            name=name,
            x=q_range,
            y=reflectivity,
        )

    def experimental_data_for_model_at_index(
        self,
        index: int = 0,
        channel: Optional[Union[PolarizationChannel, str]] = None,
    ) -> Union[DataSet1D, PolarizedDataSet]:
        """Experimental data for model at index.

        Parameters
        ----------
        index : int
            Index of the experiment.
        channel : Optional[Union[PolarizationChannel, str]]
            Spin channel of a polarized experiment. `None` (the default) keeps
            the historical behavior and returns the stored experiment as is: a
            `DataSet1D` for an unpolarized experiment, the whole
            `PolarizedDataSet` for a polarized one.

        Returns
        -------
        Union[DataSet1D, PolarizedDataSet]
            The experiment, or the `DataSet1D` of the requested channel.

        Raises
        ------
        IndexError
            No experiment is loaded at `index`.
        KeyError
            `channel` is a valid spin channel but was not measured.
        ValueError
            `channel` was given for an unpolarized experiment, or is not one of
            'pp', 'pm', 'mp', 'mm'.
        """
        if index not in self._experiments.keys():
            raise IndexError(f'No experiment data for model at index {index}')

        experiment = self._experiments[index]
        if channel is None:
            return experiment

        if not isinstance(experiment, PolarizedDataSet):
            raise ValueError(
                f"Experiment at index {index} is not polarized; it has no '{channel}' channel. "
                'Call without a channel argument to get its data.'
            )
        try:
            channel = PolarizationChannel(channel)
        except ValueError as exception:
            known = ', '.join(member.value for member in PolarizationChannel)
            raise ValueError(f"Unknown spin channel '{channel}'; expected one of {known}.") from exception
        if channel not in experiment:
            measured = ', '.join(member.value for member in experiment.available_channels)
            raise KeyError(f"Channel '{channel.value}' was not measured in experiment {index} (measured: {measured}).")
        return experiment[channel]

    def experiment_is_polarized_at_index(self, index: int = 0) -> bool:
        """Whether the experiment at index holds per-channel (polarized) data.

        Returns False when no experiment is loaded at `index`, so consumers can
        use it as a plain predicate.
        """
        return isinstance(self._experiments.get(index), PolarizedDataSet)

    def experiment_channels_at_index(self, index: int = 0) -> List[PolarizationChannel]:
        """Measured spin channels of the experiment at index ([] when unpolarized)."""
        experiment = self._experiments.get(index)
        if not isinstance(experiment, PolarizedDataSet):
            return []
        return experiment.available_channels

    def experiment_supports_spin_asymmetry_at_index(self, index: int = 0) -> bool:
        """Whether SA can be formed for the experiment at index.

        Spin asymmetry needs both non-spin-flip channels; an NSF-incomplete or
        spin-flip-only experiment has no SA. The two channel datasets must also
        be structurally usable (non-empty, matching lengths, a strictly ordered
        q grid) — an application asks this before offering a spin-asymmetry
        view, so a dataset that cannot be turned into one must not be advertised
        and then fail.
        """
        channels = self.experiment_channels_at_index(index)
        if not _has_both_nsf_channels(channels):
            return False
        experiment = self._experiments[index]
        try:
            for channel in (PolarizationChannel.PP, PolarizationChannel.MM):
                _ordered_channel_arrays(experiment[channel])
        except ValueError as exception:
            logger.warning('Experiment %s cannot produce a spin asymmetry: %s', index, exception)
            return False
        return True

    def spin_asymmetry_for_experiment_at_index(self, index: int = 0) -> Dict[str, object]:
        """Spin asymmetry SA = (R⁺⁺ − R⁻⁻) / (R⁺⁺ + R⁻⁻) of a polarized experiment.

        The measured SA is formed on the q grid of the pp channel. A mm channel
        measured on a different grid is linearly interpolated onto it — values
        with the usual weights, variances with the *squared* weights, which is
        the propagation rule for independent endpoints (the covariance this
        introduces between neighbouring SA points is not representable in
        `DataSet1D` and is discarded). Interpolation happens **only inside the q
        range both channels cover**: `np.interp` would otherwise clamp to the
        edge value and turn extrapolated points into fabricated measurements.

        Points are dropped when the denominator cannot carry a meaningful
        asymmetry:

        - it is not positive, or is lost to cancellation between the two
          channels (see :data:`SPIN_ASYMMETRY_CANCELLATION_FRACTION`) — this
          guard needs no uncertainties and is what keeps a background-subtracted
          tail from throwing ±10³ values onto the axis;
        - it is not above :data:`SPIN_ASYMMETRY_SIGNIFICANCE` times its own
          uncertainty, when the channels carry usable uncertainties;
        - the point itself is not usable (non-finite reflectivity, or a negative
          or non-finite variance).

        Parameters
        ----------
        index : int
            Index of the experiment.

        Returns
        -------
        Dict[str, object]
            - ``'measured'``: `DataSet1D` of SA versus q. As everywhere in this
              library, ``ye`` holds **variances**, not standard deviations.
            - ``'calculated'``: `DataSet1D` of the model SA on the same q
              points, or None when the model is not magnetic (an unpolarized
              model has SA ≡ 0, which is not worth drawing).
            - ``'masked_points'``: how many measured points were dropped for any
              of the reasons above.
            - ``'low_significance_points'``: of those, how many had a
              denominator below the significance threshold.
            - ``'small_denominator_points'``: of those, how many had a
              denominator that was non-positive or lost to cancellation.
            - ``'invalid_points'``: of those, how many carried a non-finite
              reflectivity or an unusable variance.
            - ``'out_of_overlap_points'``: number of pp points dropped because
              the mm channel does not cover their q.

        Raises
        ------
        IndexError
            No experiment is loaded at `index`.
        ValueError
            The experiment does not carry both pp and mm channels, or their
            datasets are not structurally usable.
        """
        if index not in self._experiments.keys():
            raise IndexError(f'No experiment data for model at index {index}')
        channels = self.experiment_channels_at_index(index)
        if not _has_both_nsf_channels(channels):
            raise ValueError(f'Experiment {index} does not have both non-spin-flip channels; spin asymmetry needs pp and mm.')

        experiment = self._experiments[index]
        q, r_pp, var_pp = _ordered_channel_arrays(experiment[PolarizationChannel.PP])
        q_mm, r_mm_source, var_mm_source = _ordered_channel_arrays(experiment[PolarizationChannel.MM])

        out_of_overlap = 0
        if q.shape == q_mm.shape and np.allclose(q, q_mm, rtol=1e-9, atol=0.0):
            # The usual case: both channels come from one instrument scan.
            # A tolerance keeps grids that only differ by float round-trips
            # (text files) on this path.
            r_mm, var_mm = r_mm_source, var_mm_source
        else:
            # Different grids: put mm on the pp grid, but only where mm has data
            # — np.interp clamps outside its range, which would silently invent
            # measurements at the edges.
            inside = (q >= q_mm.min()) & (q <= q_mm.max())
            out_of_overlap = int(np.count_nonzero(~inside))
            if out_of_overlap:
                logger.warning(
                    'Spin asymmetry of experiment %s: %s of %s pp points lie outside the mm q range '
                    '[%.5g, %.5g] and are dropped.',
                    index,
                    out_of_overlap,
                    q.size,
                    q_mm.min(),
                    q_mm.max(),
                )
            q, r_pp, var_pp = q[inside], r_pp[inside], var_pp[inside]
            r_mm, var_mm = _interpolate_with_variance(q, q_mm, r_mm_source, var_mm_source)

        asymmetry, variance, keep, reasons = _spin_asymmetry(r_pp, r_mm, var_pp, var_mm)
        measured = DataSet1D(
            name=f'Spin asymmetry for Experiment {index}',
            x=q[keep],
            y=asymmetry[keep],
            ye=variance[keep],
            x_label='q (1/angstrom)',
            y_label='Spin asymmetry',
        )

        calculated = None
        model_index = self._model_index_for_experiment(experiment)
        if model_index is not None and self.models[model_index].has_magnetism and measured.x.size > 0:
            calculated_pp = self.model_data_for_model_at_index(model_index, q_range=measured.x, channel='pp').y
            calculated_mm = self.model_data_for_model_at_index(model_index, q_range=measured.x, channel='mm').y
            calculated_asymmetry, _, _, _ = _spin_asymmetry(calculated_pp, calculated_mm)
            calculated = DataSet1D(
                name=f'Calculated spin asymmetry for Experiment {index}',
                x=measured.x,
                y=calculated_asymmetry,
                x_label='q (1/angstrom)',
                y_label='Spin asymmetry',
            )

        return {
            'measured': measured,
            'calculated': calculated,
            'masked_points': int(np.count_nonzero(~keep)),
            'out_of_overlap_points': out_of_overlap,
            **reasons,
        }

    def _model_index_for_experiment(self, experiment) -> Optional[int]:
        """Index of the model an experiment is bound to, or None."""
        model = getattr(experiment, 'model', None)
        if model is None:
            return None
        for model_index, candidate in enumerate(self._models):
            if candidate is model:
                return model_index
        return None

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
            if isinstance(experiment, PolarizedDataSet):
                self._as_dict_add_polarized_experiment(project_dict, key, experiment)
                continue
            project_dict['experiments'][key] = [
                list(experiment.x),
                list(experiment.y),
                list(experiment.ye),
            ]
            if experiment.xe is not None:
                project_dict['experiments'][key].append(list(experiment.xe))
                project_dict['experiments_models'][key] = experiment.model.name
                project_dict['experiments_names'][key] = experiment.name

    @staticmethod
    def _as_dict_add_polarized_experiment(project_dict: dict, key: int, experiment: PolarizedDataSet) -> None:
        """Serialize a `PolarizedDataSet`: one (name, x, y, ye, xe) array set per measured channel.

        `experiments[key]` is a plain list for an ordinary `DataSet1D` (see
        `_as_dict_add_experiments`); a dict here — tagged `'polarized': True` —
        is how `_from_dict_extract_experiments` tells the two apart on load.
        """
        project_dict['experiments'][key] = {
            'polarized': True,
            'channels': {
                channel.value: [
                    experiment[channel].name,
                    list(experiment[channel].x),
                    list(experiment[channel].y),
                    list(experiment[channel].ye),
                    list(experiment[channel].xe),
                ]
                for channel in experiment.available_channels
            },
        }
        if experiment.model is not None:
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

    def _from_dict_extract_experiments(self, project_dict: dict) -> Dict[int, Union[DataSet1D, PolarizedDataSet]]:
        """From dict extract experiments."""
        experiments = {}
        for key, raw in project_dict['experiments'].items():
            if isinstance(raw, dict) and raw.get('polarized'):
                experiments[int(key)] = self._polarized_experiment_from_dict(key, raw, project_dict)
                continue
            experiments[int(key)] = DataSet1D(
                name=project_dict['experiments_names'][key],
                x=raw[0],
                y=raw[1],
                ye=raw[2],
                xe=raw[3],
                model=self._models[project_dict['experiments_models'][key]],
                auto_background=False,
            )
        return experiments

    def _polarized_experiment_from_dict(self, key: str, raw: dict, project_dict: dict) -> PolarizedDataSet:
        """Reconstruct a `PolarizedDataSet` serialized by `_as_dict_add_polarized_experiment`."""
        model = self._models[project_dict['experiments_models'][key]]
        channels = {
            channel_value: DataSet1D(
                name=arrays[0],
                x=arrays[1],
                y=arrays[2],
                ye=arrays[3],
                xe=arrays[4],
                model=model,
                auto_background=False,
            )
            for channel_value, arrays in raw['channels'].items()
        }
        return PolarizedDataSet(
            name=project_dict['experiments_names'][key],
            channels=channels,
            model=model,
        )

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


def _unwrapped_angle(angle: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Make an angle profile continuous along z within each magnetic region.

    The moment angle is periodic, so a profile that turns smoothly from 359° to
    1° comes back from `atan2` as a jump of nearly 360°. Drawn as a line that is
    a full sweep across the chart where the moment barely moves. Each contiguous
    run of `mask` (a magnetic region) is therefore unwrapped on its own and then
    shifted by whole turns so it sits as close to the conventional [0, 360)
    range as possible — neighbouring regions stay independent, since the angle
    between them is not defined.

    Parameters
    ----------
    angle : np.ndarray
        Wrapped angles in degrees.
    mask : np.ndarray
        Which points carry a moment.

    Returns
    -------
    np.ndarray
        Angles in degrees, continuous within each masked region. Points outside
        the mask are returned unchanged (callers drop them).
    """
    unwrapped = np.array(angle, dtype=float, copy=True)
    if mask.size == 0 or not np.any(mask):
        return unwrapped

    # Each contiguous run of masked points is one magnetic region.
    masked_indices = np.flatnonzero(mask)
    region_breaks = np.flatnonzero(np.diff(masked_indices) > 1) + 1
    for region in np.split(masked_indices, region_breaks):
        segment = np.unwrap(unwrapped[region], period=360.0)
        # Keep the drawn values near the usual range rather than at 720 deg.
        turns = np.round(np.median(segment) / 360.0 - 0.5)
        unwrapped[region] = segment - turns * 360.0
    return unwrapped


def _has_both_nsf_channels(channels: List[PolarizationChannel]) -> bool:
    """Whether both non-spin-flip channels (pp, mm) — the pair spin asymmetry needs — are present."""
    return PolarizationChannel.PP in channels and PolarizationChannel.MM in channels


def _ordered_channel_arrays(dataset: DataSet1D) -> tuple:
    """A channel's (q, reflectivity, variance) arrays, ordered and checked.

    Spin asymmetry pairs two channels point by point and interpolates one onto
    the other, both of which assume well-formed, strictly increasing q. Rather
    than trusting that, the arrays are checked here and sorted when needed —
    `np.interp` silently returns nonsense for a descending grid.

    Raises
    ------
    ValueError
        The dataset is empty, its arrays disagree in length or shape, its q
        values are not finite, or it visits the same q twice.
    """
    name = getattr(dataset, 'name', '?')
    q = np.asarray(getattr(dataset, 'x', np.empty(0)), dtype=float).ravel()
    y = np.asarray(getattr(dataset, 'y', np.empty(0)), dtype=float).ravel()
    if q.size == 0:
        raise ValueError(f"Channel '{name}' has no data points.")
    if q.size != y.size:
        raise ValueError(f"Channel '{name}' has {q.size} q values for {y.size} reflectivities.")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"Channel '{name}' has non-finite q values.")

    variance = np.asarray(getattr(dataset, 'ye', None) if getattr(dataset, 'ye', None) is not None else [], dtype=float)
    variance = variance.ravel()
    if variance.size == 0:
        variance = np.zeros_like(y)
    elif variance.size != y.size:
        logger.warning(
            "Channel '%s' has %s uncertainties for %s points; treating it as having none.",
            name,
            variance.size,
            y.size,
        )
        variance = np.zeros_like(y)

    order = np.argsort(q, kind='stable')
    if not np.array_equal(order, np.arange(q.size)):
        logger.warning("Channel '%s' is not ordered in q; sorting it before pairing.", name)
        q, y, variance = q[order], y[order], variance[order]
    # A single-point channel has an empty `np.diff`, so the duplicate-q check
    # below is vacuously satisfied and it passes through here unrejected. That
    # is intentional: a one-point channel is a legitimate (if degenerate) SA
    # pair when it lines up exactly with the other channel's grid, and
    # `_interpolate_with_variance` handles a size-1 `q_source` correctly (its
    # clipped `searchsorted` result always resolves to that single point).
    if np.any(np.diff(q) <= 0):
        raise ValueError(f"Channel '{name}' visits the same q more than once; the pairing would be ambiguous.")
    return q, y, variance


def _interpolate_with_variance(q: np.ndarray, q_source: np.ndarray, values: np.ndarray, variances: np.ndarray) -> tuple:
    """Linear interpolation of values and their variances onto `q`.

    Values use the linear weights (1−t, t); variances use their **squares**,
    which is the propagation rule for independent endpoints — interpolating a
    variance linearly (as `np.interp` would) overestimates it, by a factor 2 at
    the midpoint of two equal variances.

    `q` must lie inside `q_source`; the caller restricts it to the overlap.
    """
    upper = np.clip(np.searchsorted(q_source, q, side='left'), 1, q_source.size - 1)
    lower = upper - 1
    span = q_source[upper] - q_source[lower]
    # span is > 0 for a strictly increasing source grid; guard anyway.
    weight = np.where(span > 0, (q - q_source[lower]) / np.where(span > 0, span, 1.0), 0.0)
    interpolated = (1.0 - weight) * values[lower] + weight * values[upper]
    interpolated_variance = (1.0 - weight) ** 2 * variances[lower] + weight**2 * variances[upper]
    return interpolated, interpolated_variance


def _spin_asymmetry(
    r_pp: np.ndarray,
    r_mm: np.ndarray,
    var_pp: Optional[np.ndarray] = None,
    var_mm: Optional[np.ndarray] = None,
) -> tuple:
    """Spin asymmetry, its variance, which points to keep, and why not.

    Parameters
    ----------
    r_pp, r_mm : np.ndarray
        Non-spin-flip reflectivities on a common q grid.
    var_pp, var_mm : Optional[np.ndarray]
        Their variances (`DataSet1D.ye`), or None for a calculated curve.

    Returns
    -------
    tuple
        SA, its variance (zeros without input variances), a boolean mask of the
        points to keep, and a dict counting the dropped ones by reason.
    """
    r_pp = np.asarray(r_pp, dtype=float)
    r_mm = np.asarray(r_mm, dtype=float)
    denominator = r_pp + r_mm
    var_pp = np.zeros_like(r_pp) if var_pp is None else np.asarray(var_pp, dtype=float)
    var_mm = np.zeros_like(r_mm) if var_mm is None else np.asarray(var_mm, dtype=float)

    # A denominator of exactly zero would divide by zero; those points are
    # dropped by the masks below anyway.
    safe_denominator = np.where(denominator == 0, np.nan, denominator)
    with np.errstate(invalid='ignore', divide='ignore'):
        asymmetry = (r_pp - r_mm) / safe_denominator
        # sigma_SA = 2 sqrt(R--^2 sigma_++^2 + R++^2 sigma_--^2) / (R++ + R--)^2,
        # so the variance is its square. ye holds variances, hence no squaring
        # of var_pp / var_mm here.
        variance = 4.0 * (r_mm**2 * var_pp + r_pp**2 * var_mm) / safe_denominator**4

    asymmetry = np.nan_to_num(asymmetry, nan=0.0, posinf=0.0, neginf=0.0)
    variance = np.nan_to_num(variance, nan=0.0, posinf=0.0, neginf=0.0)

    # A point with a non-finite reflectivity, or an uncertainty that is not a
    # usable variance, cannot produce a meaningful SA — and must not be silently
    # demoted to "no uncertainty", which would also skip the significance test.
    invalid = ~np.isfinite(r_pp) | ~np.isfinite(r_mm)
    invalid |= ~np.isfinite(var_pp) | ~np.isfinite(var_mm) | (var_pp < 0) | (var_mm < 0)

    # Uncertainty-independent guard: the denominator must be positive and must
    # not be the small remainder of two much larger numbers. Reflectivities are
    # positive, so this only bites on background-subtracted data — which is
    # exactly where SA otherwise explodes to +/-1e3 and destroys the axis.
    magnitude = np.abs(r_pp) + np.abs(r_mm)
    with np.errstate(invalid='ignore'):
        degenerate = ~np.isfinite(denominator) | (denominator <= 0)
        degenerate |= np.abs(denominator) <= SPIN_ASYMMETRY_CANCELLATION_FRACTION * magnitude
    degenerate &= ~invalid

    # Uncertainty-based guard: is the denominator above the noise?
    denominator_sigma = np.sqrt(np.clip(var_pp + var_mm, 0.0, None))
    with_uncertainty = denominator_sigma > 0
    insignificant = with_uncertainty & (denominator <= SPIN_ASYMMETRY_SIGNIFICANCE * denominator_sigma)
    insignificant &= ~invalid & ~degenerate

    keep = ~invalid & ~degenerate & ~insignificant
    reasons = {
        'invalid_points': int(np.count_nonzero(invalid)),
        'small_denominator_points': int(np.count_nonzero(degenerate)),
        'low_significance_points': int(np.count_nonzero(insignificant)),
    }
    return asymmetry, variance, keep, reasons
