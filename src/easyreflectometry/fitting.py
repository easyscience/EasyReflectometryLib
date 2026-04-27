__author__ = 'github.com/arm61'

import warnings

import numpy as np
import scipp as sc
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import FitResults
from easyscience.fitting.multi_fitter import MultiFitter as EasyScienceMultiFitter

from easyreflectometry.data import DataSet1D
from easyreflectometry.model import Model

_VALID_OBJECTIVES = ('legacy_mask', 'mighell', 'hybrid', 'auto')
_EPS = 1e-30


def _validate_objective(objective: str) -> str:
    """Validate and resolve the objective string.

    :param objective: The objective mode string.
    :type objective: str
    :return: Resolved objective string ('auto' becomes 'hybrid').
    :rtype: str
    :raises ValueError: If the objective is not one of the valid options.
    """
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(f'Unknown objective {objective!r}. Valid options: {_VALID_OBJECTIVES}')
    if objective == 'auto':
        return 'hybrid'
    return objective


def _prepare_fit_arrays(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    variances: np.ndarray,
    objective: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Prepare x, y_eff, and weights arrays for fitting based on the objective mode.

    For ``legacy_mask``, zero-variance points are removed from all arrays.
    For ``hybrid``, valid-variance points use standard WLS while zero-variance
    points use Mighell-transformed y and weights.
    For ``mighell``, all points use the Mighell transform.

    Note: ``variances`` here means σ² (the scipp convention), not σ.

    :param x_vals: Independent variable values.
    :type x_vals: np.ndarray
    :param y_vals: Observed dependent variable values.
    :type y_vals: np.ndarray
    :param variances: Variance (σ²) of each observed point.
    :type variances: np.ndarray
    :param objective: One of 'legacy_mask', 'hybrid', 'mighell'.
    :type objective: str
    :return: Tuple of (x_out, y_eff, weights, stats) where stats is a dict
             with keys 'valid', 'mighell_substituted', 'masked'.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, dict]
    """
    n = len(y_vals)
    zero_mask = variances <= 0.0
    n_zero = int(np.sum(zero_mask))
    n_valid = n - n_zero

    if objective == 'legacy_mask':
        valid = ~zero_mask
        x_out = x_vals[valid]
        y_eff = y_vals[valid]
        if n_valid > 0:
            weights = 1.0 / np.sqrt(variances[valid])
        else:
            weights = np.array([])
        stats = {'valid': n_valid, 'mighell_substituted': 0, 'masked': n_zero, 'transformed_all_points': False}
        return x_out, y_eff, weights, stats

    # hybrid or mighell
    y_eff = np.copy(y_vals)
    sigma = np.empty(n)

    if objective == 'mighell':
        apply_mighell = np.ones(n, dtype=bool)
    else:
        # hybrid: apply Mighell only to zero-variance points
        apply_mighell = zero_mask

    # Standard WLS for non-Mighell points
    standard = ~apply_mighell
    if np.any(standard):
        sigma[standard] = np.sqrt(variances[standard])

    # Mighell transform for selected points
    if np.any(apply_mighell):
        y_m = y_vals[apply_mighell]
        delta = np.minimum(y_m, 1.0)
        y_eff[apply_mighell] = y_m + delta
        sigma[apply_mighell] = np.sqrt(np.maximum(y_m + 1.0, _EPS))

    weights = 1.0 / sigma
    n_mighell = int(np.sum(apply_mighell))
    stats = {
        'valid': n - n_mighell,
        'mighell_substituted': n_mighell,
        'masked': 0,
        'transformed_all_points': bool(objective == 'mighell'),
    }
    return x_vals, y_eff, weights, stats


def _compute_weighted_chi2(y_obs: np.ndarray, y_calc: np.ndarray, sigma: np.ndarray) -> float:
    """Return weighted chi-square for finite, strictly positive uncertainties."""
    valid = np.isfinite(y_obs) & np.isfinite(y_calc) & np.isfinite(sigma) & (sigma > 0.0)
    if not np.any(valid):
        return 0.0
    residual = (y_obs[valid] - y_calc[valid]) / sigma[valid]
    return float(np.sum(residual**2))


def _compute_reduced_chi2(chi2: float, n_points: int, n_params: int) -> float | None:
    """Return reduced chi-square or None when degrees of freedom are not positive."""
    dof = int(n_points) - int(n_params)
    if dof <= 0:
        return None
    return float(chi2 / dof)


class MultiFitter:
    def __init__(self, *args: Model, objective: str = 'hybrid'):
        r"""A convenience class for the :py:class:`easyscience.Fitting.Fitting`
        which will populate the :py:class:`sc.DataGroup` appropriately
        after the fitting is performed.

        :param args: Reflectometry model(s).
        :param objective: Zero-variance handling strategy. One of
            ``'hybrid'`` (default, Mighell for zero-variance, WLS otherwise),
            ``'mighell'`` (Mighell transform for all points),
            ``'legacy_mask'`` (drop zero-variance points),
            ``'auto'`` (alias for ``'hybrid'``).
        :type objective: str
        """

        # This lets the unique_name be passed with the fit_func.
        def func_wrapper(func, unique_name):
            def wrapped(*args, **kwargs):
                return func(*args, unique_name, **kwargs)

            return wrapped

        self._fit_func = [func_wrapper(m.interface.fit_func, m.unique_name) for m in args]
        self._models = args
        self.easy_science_multi_fitter = EasyScienceMultiFitter(args, self._fit_func)
        self._fit_results: list[FitResults] | None = None
        self._classical_fit_metrics: list[dict] | None = None
        self._objective = _validate_objective(objective)

    def fit(self, data: sc.DataGroup, id: int = 0, objective: str | None = None) -> sc.DataGroup:
        """Perform the fitting and populate the DataGroups with the result.

        :param data: DataGroup to be fitted to and populated.
        :type data: sc.DataGroup
        :param id: Unused parameter kept for backward compatibility.
        :type id: int
        :param objective: Per-call override for the zero-variance objective.
            If ``None``, uses the instance default set at construction.
        :type objective: str or None
        :return: A new DataGroup with fitted model curves, SLD profiles, and fit statistics.
        :rtype: sc.DataGroup

        :note: Under the ``mighell`` objective all points are transformed,
               so ``reduced_chi`` is not a classical chi-square statistic.
               Under ``hybrid``, only zero-variance points are transformed;
               when they are a small fraction of the data the chi-square
               remains approximately classical.
        """
        obj = _validate_objective(objective) if objective is not None else self._objective

        refl_nums = [k[3:] for k in data['coords'].keys() if 'Qz' == k[:2]]
        x = []
        y = []
        dy = []
        original_arrays = []

        # Process each reflectivity dataset
        for i in refl_nums:
            x_vals = data['coords'][f'Qz_{i}'].values
            y_vals = data['data'][f'R_{i}'].values
            variances = data['data'][f'R_{i}'].variances

            x_out, y_eff, weights, stats = _prepare_fit_arrays(x_vals, y_vals, variances, obj)

            if stats['masked'] > 0:
                warnings.warn(
                    f'Masked {stats["masked"]} data point(s) in reflectivity {i} due to zero variance during fitting.',
                    UserWarning,
                )
            if stats.get('transformed_all_points'):
                warnings.warn(
                    f'Applied Mighell transform to all {len(y_vals)} point(s) in reflectivity {i} during fitting.',
                    UserWarning,
                )
            elif stats['mighell_substituted'] > 0:
                warnings.warn(
                    f'Applied Mighell substitution to {stats["mighell_substituted"]} '
                    f'zero-variance point(s) in reflectivity {i} during fitting.',
                    UserWarning,
                )

            x.append(x_out)
            y.append(y_eff)
            dy.append(weights)
            original_arrays.append({'x': x_vals, 'y': y_vals, 'variances': variances})

        result = self.easy_science_multi_fitter.fit(x, y, weights=dy)
        self._fit_results = result
        self._classical_fit_metrics = []
        new_data = data.copy()
        for i, _ in enumerate(result):
            id = refl_nums[i]
            model_curve = self._fit_func[i](data['coords'][f'Qz_{id}'].values)
            new_data[f'R_{id}_model'] = sc.array(dims=[f'Qz_{id}'], values=model_curve)
            sld_profile = self.easy_science_multi_fitter._fit_objects[i].interface.sld_profile(self._models[i].unique_name)
            new_data[f'SLD_{id}'] = sc.array(dims=[f'z_{id}'], values=sld_profile[1] * 1e-6, unit=sc.Unit('1/angstrom') ** 2)
            if 'attrs' in new_data:
                new_data['attrs'][f'R_{id}_model'] = {'model': sc.scalar(self._models[i].as_dict())}
            new_data['coords'][f'z_{id}'] = sc.array(
                dims=[f'z_{id}'], values=sld_profile[0], unit=(1 / new_data['coords'][f'Qz_{id}'].unit).unit
            )
            original = original_arrays[i]
            sigma_classical = np.sqrt(np.clip(original['variances'], 0.0, None))
            n_classical_points = int(np.sum(original['variances'] > 0.0))
            classical_chi2 = _compute_weighted_chi2(original['y'], model_curve, sigma_classical)
            classical_reduced_chi = _compute_reduced_chi2(classical_chi2, n_classical_points, result[i].n_pars)
            objective_chi2 = float(result[i].chi2)
            objective_reduced_chi = float(result[i].reduced_chi)

            self._classical_fit_metrics.append(
                {
                    'classical_chi2': classical_chi2,
                    'classical_reduced_chi': classical_reduced_chi,
                    'objective_chi2': objective_chi2,
                    'objective_reduced_chi': objective_reduced_chi,
                    'n_classical_points': n_classical_points,
                }
            )

            new_data['objective_chi2'] = objective_chi2
            new_data['objective_reduced_chi'] = objective_reduced_chi
            new_data['classical_chi2'] = classical_chi2
            new_data['classical_reduced_chi'] = classical_reduced_chi
            new_data['reduced_chi'] = float(result[i].reduced_chi)
            new_data['success'] = result[i].success
        return new_data

    def fit_single_data_set_1d(self, data: DataSet1D, objective: str | None = None) -> FitResults:
        """Perform fitting on a single 1D dataset.

        :param data: The 1D dataset to fit. Note that ``data.ye`` stores
            variances (σ²), not standard deviations.
        :type data: DataSet1D
        :param objective: Per-call override for the zero-variance objective.
            If ``None``, uses the instance default set at construction.
        :type objective: str or None
        :return: Fit results from the minimizer.
        :rtype: FitResults
        """
        obj = _validate_objective(objective) if objective is not None else self._objective

        x_vals = np.asarray(data.x)
        y_vals = np.asarray(data.y)
        variances = np.asarray(data.ye)

        x_out, y_eff, weights, stats = _prepare_fit_arrays(x_vals, y_vals, variances, obj)

        if stats['masked'] > 0:
            warnings.warn(
                f'Masked {stats["masked"]} data point(s) in single-dataset fit due to zero variance during fitting.',
                UserWarning,
            )
        if stats.get('transformed_all_points'):
            warnings.warn(
                f'Applied Mighell transform to all {len(y_vals)} point(s) in single-dataset fit during fitting.',
                UserWarning,
            )
        elif stats['mighell_substituted'] > 0:
            warnings.warn(
                f'Applied Mighell substitution to {stats["mighell_substituted"]} '
                'zero-variance point(s) in single-dataset fit during fitting.',
                UserWarning,
            )

        if obj == 'legacy_mask' and len(x_out) == 0:
            raise ValueError('Cannot fit single dataset: all points have zero variance.')

        result = self.easy_science_multi_fitter.fit(x=[x_out], y=[y_eff], weights=[weights])[0]
        self._fit_results = [result]
        sigma_classical = np.sqrt(np.clip(variances, 0.0, None))
        model_curve = self._fit_func[0](x_vals)
        n_classical_points = int(np.sum(variances > 0.0))
        classical_chi2 = _compute_weighted_chi2(y_vals, model_curve, sigma_classical)
        classical_reduced_chi = _compute_reduced_chi2(classical_chi2, n_classical_points, result.n_pars)
        self._classical_fit_metrics = [
            {
                'classical_chi2': classical_chi2,
                'classical_reduced_chi': classical_reduced_chi,
                'objective_chi2': float(result.chi2),
                'objective_reduced_chi': float(result.reduced_chi),
                'n_classical_points': n_classical_points,
            }
        ]
        return result

    @property
    def chi2(self) -> float | None:
        """Total chi-squared across all fitted datasets, or None if no fit has been performed."""
        if self._fit_results is None:
            return None
        return sum(r.chi2 for r in self._fit_results)

    @property
    def reduced_chi(self) -> float | None:
        """Reduced chi-squared from the most recent fit, or None if no fit has been performed."""
        if self._fit_results is None:
            return None
        total_chi2 = sum(r.chi2 for r in self._fit_results)
        total_points = sum(np.size(r.x) for r in self._fit_results)
        n_params = self._fit_results[0].n_pars
        total_dof = total_points - n_params

        if total_dof <= 0:
            return None

        return total_chi2 / total_dof

    @property
    def classical_chi2(self) -> float | None:
        """Classical chi-squared using only points with positive variances."""
        if self._classical_fit_metrics is None:
            return None
        return float(sum(metric['classical_chi2'] for metric in self._classical_fit_metrics))

    @property
    def classical_reduced_chi(self) -> float | None:
        """Reduced classical chi-squared using only points with positive variances."""
        if self._classical_fit_metrics is None or self._fit_results is None:
            return None
        total_chi2 = self.classical_chi2
        total_points = sum(metric['n_classical_points'] for metric in self._classical_fit_metrics)
        n_params = self._fit_results[0].n_pars
        return _compute_reduced_chi2(total_chi2, total_points, n_params)

    @property
    def objective_chi2(self) -> float | None:
        """Objective-space chi-squared returned by the minimizer."""
        return self.chi2

    @property
    def objective_reduced_chi(self) -> float | None:
        """Objective-space reduced chi-squared returned by the minimizer."""
        return self.reduced_chi

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
