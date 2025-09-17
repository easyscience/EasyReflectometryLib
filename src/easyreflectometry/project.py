import datetime
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting.fitter import DEFAULT_MINIMIZER
from easyscience.variable import Parameter
from scipp import DataGroup

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import load_as_dataset
from easyreflectometry.fitting import MultiFitter
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
from easyreflectometry.utils import collect_unique_names_from_dict

Q_MIN = 0.001
Q_MAX = 0.3
Q_RESOLUTION = 500

DEFAULT_MINIZER = AvailableMinimizers.LMFit_leastsq


# Multiprocessing worker must be a top-level function for Windows compatibility
def _calculate_for_model_mpi(args):
    """Worker function for multiprocessing - must be pickleable."""
    try:
        idx, model_dict, q_range, calculator_name = args

        # Import required modules
        from easyreflectometry.model import Model  # noqa: I001
        from easyreflectometry.calculators import CalculatorFactory
        from easyreflectometry.data import DataSet1D

        # Instead of clearing the global registry (expensive!), we'll use process isolation
        # Each worker process starts with a clean registry anyway due to multiprocessing spawn

        # Reconstruct model and set calculator
        model = Model.from_dict(model_dict)
        calculator = CalculatorFactory()
        calculator.switch(calculator_name)
        model.interface = calculator

        # Calculate reflectivity using the original unique_name
        reflectivity = model.interface().reflectivity_profile(q_range, model.unique_name)

        return idx, DataSet1D(
            name=f'Reflectivity for Model {idx}',
            x=q_range,
            y=reflectivity,
        )

        return idx, DataSet1D(
            name=f'Reflectivity for Model {idx}',
            x=q_range,
            y=reflectivity,
        )
    except Exception as e:
        # Return error info for debugging
        import traceback
        return idx, f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"

class Project:

    def model_data_for_models_mpi(self, q_range: Optional[np.array] = None) -> Dict[int, DataSet1D]:
        """
        Calculate reflectivity for all models using multiprocessing (CPU-bound parallelism).

        Parameters
        ----------
        q_range : Optional[np.array], optional
            Q range for reflectivity calculation. If None, uses project q_min, q_max, q_resolution.

        Returns
        -------
        Dict[int, DataSet1D]
            Dictionary with model indices as keys and corresponding DataSet1D objects as values.
        """
        cpu_count = multiprocessing.cpu_count()
        print(f"Using {cpu_count} CPU cores")

        if q_range is None:
            q_range = np.linspace(self.q_min, self.q_max, self.q_resolution)

        # Set interface for all models
        for model in self.models:
            model.interface = self._calculator

        # Prepare serializable arguments (no live objects, just data)
        args = []
        for i in range(len(self.models)):
            model_dict = self.models[i].as_dict(skip=['interface'])
            args.append((i, model_dict, q_range, self._calculator.current_interface_name))

        if len(args) > 1:
            try:
                # Set spawn method for Windows compatibility
                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool() as pool:  # Use all available CPU cores
                    results = pool.map(_calculate_for_model_mpi, args)
            except Exception as e:
                # Fallback to sequential processing if multiprocessing fails
                print(f"Multiprocessing failed ({e}), falling back to sequential processing")
                results = [_calculate_for_model_mpi(arg) for arg in args]
        else:
            results = [_calculate_for_model_mpi(arg) for arg in args]

        # Check for errors and convert to dict
        final_results = {}
        for result in results:
            if isinstance(result[1], str) and result[1].startswith("Error:"):
                raise RuntimeError(f"Model {result[0]}: {result[1]}")
            final_results[result[0]] = result[1]

        return final_results

    def __init__(self):
        self._info = self._default_info()
        self._path_project_parent = Path(os.path.expanduser('~'))
        self._models = ModelCollection(populate_if_none=False, unique_name='project_models')
        self._materials = MaterialCollection(populate_if_none=False, unique_name='project_materials')
        self._calculator = CalculatorFactory()
        self._experiments: Dict[DataGroup] = {}
        self._fitter: MultiFitter = None
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
        del self._models
        del self._materials
        global_object.map._clear()

        self.__init__()

    @property
    def parameters(self) -> List[Parameter]:
        unique_names_in_project = collect_unique_names_from_dict(self.as_dict())
        parameters = []
        for vertice_str in global_object.map.vertices():
            vertice_obj = global_object.map.get_item_by_key(vertice_str)
            if isinstance(vertice_obj, Parameter) and vertice_str in unique_names_in_project:
                parameters.append(vertice_obj)
        return parameters

    @property
    def enabled_parameters(self) -> List[Parameter]:
        parameters = self.parameters
        # Only include enabled parameters
        return [param for param in parameters if param.enabled]

    @property
    def q_min(self):
        if self._q_min is None:
            return Q_MIN
        return self._q_min

    @q_min.setter
    def q_min(self, value: float) -> None:
        self._q_min = value

    @property
    def q_max(self):
        if self._q_max is None:
            return Q_MAX
        return self._q_max

    @q_max.setter
    def q_max(self, value: float) -> None:
        self._q_max = value

    @property
    def q_resolution(self):
        if self._q_resolution is None:
            return Q_RESOLUTION
        return self._q_resolution

    @q_resolution.setter
    def q_resolution(self, value: int) -> None:
        self._q_resolution = value

    @property
    def current_material_index(self) -> Optional[int]:
        return self._current_material_index

    @current_material_index.setter
    def current_material_index(self, value: int) -> None:
        if value < 0 or value >= len(self._materials):
            raise ValueError(f'Index {value} out of range')
        if self._current_material_index != value:
            self._current_material_index = value

    @property
    def current_model_index(self) -> Optional[int]:
        return self._current_model_index

    @current_model_index.setter
    def current_model_index(self, value: int) -> None:
        if value < 0 or value >= len(self._models):
            raise ValueError(f'Index {value} out of range')
        if self._current_model_index != value:
            self._current_model_index = value
            self._current_assembly_index = 0
            self._current_layer_index = 0

    @property
    def current_assembly_index(self) -> Optional[int]:
        return self._current_assembly_index

    @current_assembly_index.setter
    def current_assembly_index(self, value: int) -> None:
        if value < 0 or value >= len(self._models[self._current_model_index].sample):
            raise ValueError(f'Index {value} out of range')
        if self._current_assembly_index != value:
            self._current_assembly_index = value
            self._current_layer_index = 0

    @property
    def current_layer_index(self) -> Optional[int]:
        return self._current_layer_index

    @current_layer_index.setter
    def current_layer_index(self, value: int) -> None:
        if value < 0 or value >= len(self._models[self._current_model_index].sample[self._current_assembly_index].layers):
            raise ValueError(f'Index {value} out of range')
        if self._current_layer_index != value:
            self._current_layer_index = value

    @property
    def current_experiment_index(self) -> Optional[int]:
        return self._current_experiment_index

    @current_experiment_index.setter
    def current_experiment_index(self, value: int) -> None:
        if value < 0 or value >= len(self._experiments):
            raise ValueError(f'Index {value} out of range')
        if self._current_experiment_index != value:
            self._current_experiment_index = value
            # Resetting the model index to 0 when changing the experiment
            # self.current_model_index = 0

    @property
    def created(self) -> bool:
        return self._created

    @property
    def path(self):
        return self._path_project_parent / self._info['name']

    def set_path_project_parent(self, path: Union[Path, str]):
        self._path_project_parent = Path(path)

    @property
    def models(self) -> ModelCollection:
        return self._models

    @models.setter
    def models(self, models: ModelCollection) -> None:
        self._replace_collection(models, self._models)
        # Use setter to update indicies for current model, assembly and layer
        self.current_model_index = 0
        self._materials.extend(self._get_materials_in_models())
        for model in self._models:
            model.interface = self._calculator

    @property
    def fitter(self) -> MultiFitter:
        if len(self._models):
            if (self._fitter is None) or (self._fitter_model_index != self._current_model_index):
                minimizer = self.minimizer
                self._fitter = MultiFitter(self._models[self._current_model_index])
                self.minimizer = minimizer
                self._fitter_model_index = self._current_model_index
        return self._fitter

    @property
    def calculator(self) -> str:
        return self._calculator.current_interface_name

    @calculator.setter
    def calculator(self, calculator: str) -> None:
        self._calculator.switch(calculator)

    @property
    def minimizer(self) -> AvailableMinimizers:
        if self._fitter is not None:
            return self._fitter.easy_science_multi_fitter.minimizer.enum
        return DEFAULT_MINIMIZER

    @minimizer.setter
    def minimizer(self, minimizer: AvailableMinimizers) -> None:
        if self._fitter is not None:
            self._fitter.easy_science_multi_fitter.switch_minimizer(minimizer)

    @property
    def experiments(self) -> Dict[int, DataSet1D]:
        return self._experiments

    @experiments.setter
    def experiments(self, experiments: Dict[int, DataSet1D]) -> None:
        self._experiments = experiments

    @property
    def path_json(self):
        return self.path / 'project.json'

    def get_index_air(self) -> int:
        if 'Air' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='Air', sld=0.0, isld=0.0))
        return [material.name for material in self._materials].index('Air')

    def get_index_si(self) -> int:
        if 'Si' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='Si', sld=2.07, isld=0.0))
        return [material.name for material in self._materials].index('Si')

    def get_index_sio2(self) -> int:
        if 'SiO2' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='SiO2', sld=3.47, isld=0.0))
        return [material.name for material in self._materials].index('SiO2')

    def get_index_d2o(self) -> int:
        if 'D2O' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='D2O', sld=6.36, isld=0.0))
        return [material.name for material in self._materials].index('D2O')

    def load_new_experiment(self, path: Union[Path, str]) -> None:
        new_experiment = load_as_dataset(str(path))
        new_index = len(self._experiments)
        new_experiment.name = f'Experiment {new_index}'
        model_index = 0
        if new_index < len(self.models):
            model_index = new_index
        new_experiment.model = self.models[model_index]
        self._experiments[new_index] = new_experiment
        # self._current_model_index = new_index

    def load_experiment_for_model_at_index(self, path: Union[Path, str], index: Optional[int] = 0) -> None:
        self._experiments[index] = load_as_dataset(str(path))
        self._experiments[index].name = f'Experiment {index}'
        self._experiments[index].model = self.models[index]

        self._with_experiments = True

        # Set the resolution function if variance data is present
        if sum(self._experiments[index].ye) != 0:
            q = self._experiments[index].x
            reflectivity = self._experiments[index].y
            q_error = self._experiments[index].xe
            resolution_function = Pointwise(q_data_points=[q, reflectivity, q_error])
            # resolution_function = LinearSpline(
            #     q_data_points=self._experiments[index].y,
            #     fwhm_values=np.sqrt(self._experiments[index].ye),
            # )
            self._models[index].resolution_function = resolution_function

    def sld_data_for_model_at_index(self, index: int = 0) -> DataSet1D:
        self.models[index].interface = self._calculator
        sld = self.models[index].interface().sld_profile(self._models[index].unique_name)
        return DataSet1D(
            name=f'SLD for Model {index}',
            x=sld[0],
            y=sld[1],
        )

    def sample_data_for_model_at_index(self, index: int = 0, q_range: Optional[np.array] = None) -> DataSet1D:
        original_resolution_function = self.models[index].resolution_function
        self.models[index].resolution_function = PercentageFwhm(0)
        reflectivity_data = self.model_data_for_model_at_index(index, q_range)
        self.models[index].resolution_function = original_resolution_function

        return reflectivity_data

    def model_data_for_model_at_index(self, index: int = 0, q_range: Optional[np.array] = None) -> DataSet1D:
        if q_range is None:
            q_range = np.linspace(self.q_min, self.q_max, self.q_resolution)
        self.models[index].interface = self._calculator
        reflectivity = self.models[index].interface().reflectivity_profile(q_range, self._models[index].unique_name)
        return DataSet1D(
            name=f'Reflectivity for Model {index}',
            x=q_range,
            y=reflectivity,
        )

    def model_data_for_models(self, q_range: Optional[np.array] = None) -> Dict[int, DataSet1D]:
        """
        Calculate reflectivity for all models using multithreading.

        Parameters
        ----------
        q_range : Optional[np.array], optional
            Q range for reflectivity calculation. If None, uses project q_min, q_max, q_resolution.

        Returns
        -------
        Dict[int, DataSet1D]
            Dictionary with model indices as keys and corresponding DataSet1D objects as values.
        """
        if q_range is None:
            q_range = np.linspace(self.q_min, self.q_max, self.q_resolution)

        # Set interface for all models
        for model in self.models:
            model.interface = self._calculator

        results = {}

        # Function to calculate reflectivity for a single model
        def calculate_for_model(idx):
            model = self.models[idx]
            reflectivity = model.interface().reflectivity_profile(q_range, model.unique_name)
            return idx, DataSet1D(
                name=f'Reflectivity for Model {idx}',
                x=q_range,
                y=reflectivity,
            )

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and collect futures
            futures = [executor.submit(calculate_for_model, i) for i in range(len(self.models))]

            # Collect results as they complete
            for future in futures:
                idx, dataset = future.result()
                results[idx] = dataset

        return results

    def experimental_data_for_model_at_index(self, index: int = 0) -> DataSet1D:
        if index in self._experiments.keys():
            return self._experiments[index]
        else:
            raise IndexError(f'No experiment data for model at index {index}')

    def default_model(self):
        self._replace_collection(MaterialCollection(), self._materials)

        layers = [
            Layer(material=self._materials[0], thickness=0.0, roughness=0.0, name='Vacuum Layer'),
            Layer(material=self._materials[1], thickness=100.0, roughness=3.0, name='D2O Layer'),
            Layer(material=self._materials[2], thickness=0.0, roughness=1.2, name='Si Layer'),
        ]
        assemblies = [
            Multilayer(layers[0], name='Superphase'),
            Multilayer(layers[1], name='D2O'),
            Multilayer(layers[2], name='Subphase'),
        ]
        sample = Sample(*assemblies)
        model = Model(sample=sample)
        self.models = ModelCollection([model])

    def add_material(self, material: MaterialCollection) -> None:
        if material in self._materials:
            print(f'WARNING: Material {material} is already in material collection')
        else:
            self._materials.append(material)

    def remove_material(self, index: int) -> None:
        if self._materials[index] in self._get_materials_in_models():
            print(f'ERROR: Material {self._materials[index]} is used in models')
        else:
            self._materials.pop(index)

    def _default_info(self):
        return dict(
            name='DefaultEasyReflectometryProject',
            short_description='Reflectometry, 1D',
            modified=datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
        )

    def create(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(self.path / 'experiments')
            self._created = True
            self._timestamp_modification()
        else:
            print(f'ERROR: Directory {self.path} already exists')

    def save_as_json(self, overwrite=False):
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

    def as_dict(self, include_materials_not_in_model=False):
        project_dict = {}
        project_dict['info'] = self._info
        project_dict['with_experiments'] = self._with_experiments
        if self._models is not None:
            project_dict['models'] = self._models.as_dict(skip=['interface'])
            project_dict['models']['unique_name'] = project_dict['models']['unique_name'] + '_to_prevent_collisions_on_load'
        if include_materials_not_in_model:
            self._as_dict_add_materials_not_in_model_dict(project_dict)
        if self._with_experiments:
            self._as_dict_add_experiments(project_dict)
        if self.fitter is not None:
            project_dict['fitter_minimizer'] = self.fitter.easy_science_multi_fitter.minimizer.name
        if self._calculator is not None:
            project_dict['calculator'] = self._calculator.current_interface_name
        if self._colors is not None:
            project_dict['colors'] = self._colors
        return project_dict

    def _as_dict_add_materials_not_in_model_dict(self, project_dict: dict):
        materials_not_in_model = []
        for material in self._materials:
            if material not in self._get_materials_in_models():
                materials_not_in_model.append(material)
        if len(materials_not_in_model) > 0:
            project_dict['materials_not_in_model'] = MaterialCollection(materials_not_in_model).as_dict(skip=['interface'])

    def _as_dict_add_experiments(self, project_dict: dict):
        project_dict['experiments'] = {}
        project_dict['experiments_models'] = {}
        project_dict['experiments_names'] = {}

        for key, experiment in self._experiments.items():
            project_dict['experiments'][key] = [list(experiment.x), list(experiment.y), list(experiment.ye)]
            if experiment.xe is not None:
                project_dict['experiments'][key].append(list(experiment.xe))
                project_dict['experiments_models'][key] = experiment.model.name
                project_dict['experiments_names'][key] = experiment.name

    def from_dict(self, project_dict: dict):
        keys = list(project_dict.keys())
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
            self.fitter.easy_science_multi_fitter.switch_minimizer(AvailableMinimizers[project_dict['fitter_minimizer']])
        else:
            self._fitter = None
        if 'experiments' in keys:
            self._experiments = self._from_dict_extract_experiments(project_dict)
        else:
            self._experiments = {}

    def _from_dict_extract_experiments(self, project_dict: dict) -> Dict[int, DataSet1D]:
        experiments = {}
        for key in project_dict['experiments'].keys():
            experiments[int(key)] = DataSet1D(
                name=project_dict['experiments_names'][key],
                x=project_dict['experiments'][key][0],
                y=project_dict['experiments'][key][1],
                ye=project_dict['experiments'][key][2],
                xe=project_dict['experiments'][key][3],
                model=self._models[project_dict['experiments_models'][key]],
            )
        return experiments

    def _get_materials_in_models(self) -> MaterialCollection:
        materials_in_model = MaterialCollection(populate_if_none=False)
        for model in self._models:
            for assembly in model.sample:
                for layer in assembly.layers:
                    materials_in_model.append(layer.material)
        return materials_in_model

    def _replace_collection(self, src_collection: BaseCollection, dst_collection: BaseCollection) -> None:
        # Clear the destination collection
        for i in range(len(dst_collection)):
            dst_collection.pop(0)

        for element in src_collection:
            dst_collection.append(element)

    def _timestamp_modification(self):
        self._info['modified'] = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
