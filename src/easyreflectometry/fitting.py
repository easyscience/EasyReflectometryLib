__author__ = 'github.com/arm61'

import numpy as np
import scipp as sc
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import FitResults
from easyscience.fitting.multi_fitter import MultiFitter as EasyScienceMultiFitter

from easyreflectometry.data import DataSet1D
from easyreflectometry.model import Model


class MultiFitter:
    def __init__(self, *args: Model):
        r"""A convinence class for the :py:class:`easyscience.Fitting.Fitting`
        which will populate the :py:class:`sc.DataGroup` appropriately
        after the fitting is performed.

        :param args: Reflectometry model
        """

        # This lets the unique_name be passed with the fit_func.
        def func_wrapper(func, unique_name):
            def wrapped(*args, **kwargs):
                return func(*args, unique_name, **kwargs)

            return wrapped

        self._fit_func = [func_wrapper(m.interface.fit_func, m.unique_name) for m in args]
        self._models = args
        self.easy_science_multi_fitter = EasyScienceMultiFitter(args, self._fit_func)

    def fit(self, data: sc.DataGroup, id: int = 0) -> sc.DataGroup:
        """
        Perform the fitting and populate the DataGroups with the result.

        :param data: DataGroup to be fitted to and populated
        :param method: Optimisation method
        """
        refl_nums = [k[3:] for k in data['coords'].keys() if 'Qz' == k[:2]]
        x = [data['coords'][f'Qz_{i}'].values for i in refl_nums]
        y = [data['data'][f'R_{i}'].values for i in refl_nums]
        dy = [1 / np.sqrt(data['data'][f'R_{i}'].variances) for i in refl_nums]
        result = self.easy_science_multi_fitter.fit(x, y, weights=dy)
        new_data = data.copy()
        for i, _ in enumerate(result):
            id = refl_nums[i]
            new_data[f'R_{id}_model'] = sc.array(
                dims=[f'Qz_{id}'], values=self._fit_func[i](data['coords'][f'Qz_{id}'].values)
            )
            sld_profile = self.easy_science_multi_fitter._fit_objects[i].interface.sld_profile(self._models[i].unique_name)
            new_data[f'SLD_{id}'] = sc.array(dims=[f'z_{id}'], values=sld_profile[1] * 1e-6, unit=sc.Unit('1/angstrom') ** 2)
            new_data['attrs'][f'R_{id}_model'] = {'model': sc.scalar(self._models[i].as_dict())}
            new_data['coords'][f'z_{id}'] = sc.array(
                dims=[f'z_{id}'], values=sld_profile[0], unit=(1 / new_data['coords'][f'Qz_{id}'].unit).unit
            )
            new_data['reduced_chi'] = float(result[i].reduced_chi)
            new_data['success'] = result[i].success
        return new_data

    def fit_single_data_set_1d(self, data: DataSet1D) -> FitResults:
        """
        Perform the fitting and populate the DataGroups with the result.

        :param data: DataGroup to be fitted to and populated
        :param method: Optimisation method
        """
        return self.easy_science_multi_fitter.fit(x=[data.x], y=[data.y], weights=[data.ye])[0]

    def switch_minimizer(self, minimizer: AvailableMinimizers) -> None:
        """
        Switch the minimizer for the fitting.

        :param minimizer: Minimizer to be switched to
        """
        self.easy_science_multi_fitter.switch_minimizer(minimizer)


def _flatten_list(this_list: list) -> list:
    """
    Flatten nested lists.

    :param this_list: List to be flattened

    :return: Flattened list
    """
    return np.array([item for sublist in this_list for item in sublist])
