"""
Example: Zero-variance handling with the three objective modes.

This script demonstrates how MultiFitter handles data points that have
zero variance under each of the three objective strategies:

  - legacy_mask : drops zero-variance points (old behaviour)
  - hybrid      : keeps all points; applies Mighell substitution only to
                   zero-variance entries (new default)
  - mighell     : applies the Mighell transform to every point

The example builds a simple film-on-substrate reflectometry model,
creates synthetic data with a few zero-variance points injected, and
fits with each mode so you can compare results side-by-side.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample


def build_sample_and_model():
    """Return a fresh (sample, model) pair with fittable parameters."""
    si = Material(2.07, 0, 'Si')
    sio2 = Material(3.47, 0, 'SiO2')
    film = Material(2.0, 0, 'Film')
    d2o = Material(6.36, 0, 'D2O')

    si_layer = Layer(si, 0, 0, 'Si layer')
    sio2_layer = Layer(sio2, 30, 3, 'SiO2 layer')
    film_layer = Layer(film, 250, 3, 'Film Layer')
    superphase = Layer(d2o, 0, 3, 'D2O Subphase')

    sample = Sample(
        Multilayer(si_layer),
        Multilayer(sio2_layer),
        Multilayer(film_layer),
        Multilayer(superphase),
        name='Film Structure',
    )

    resolution_function = PercentageFwhm(0.02)
    model = Model(sample, 1, 1e-6, resolution_function, 'Film Model')

    # Free the parameters we want to fit
    sio2_layer.thickness.fixed = False
    sio2_layer.thickness.bounds = (15, 50)

    film_layer.thickness.fixed = False
    film_layer.thickness.bounds = (200, 300)

    film.sld.fixed = False
    film.sld.bounds = (0.1, 3)

    model.background.fixed = False
    model.background.bounds = (1e-7, 1e-5)

    model.scale.fixed = False
    model.scale.bounds = (0.5, 1.5)

    model.interface = CalculatorFactory()

    return sample, model


def make_synthetic_data_with_zero_variances(model):
    """Generate a DataGroup from the model, then inject zero variances."""
    qz = np.linspace(0.01, 0.30, 80)
    r_true = model.interface.fit_func(qz, model.unique_name)

    # Add a little Gaussian noise for realism
    rng = np.random.default_rng(42)
    noise_sigma = 0.02 * r_true + 1e-4
    r_noisy = r_true + rng.normal(0, noise_sigma)

    # Variance = sigma^2
    variances = noise_sigma**2

    # Inject 6 points with zero variance (simulating detector dead-spots, etc.)
    zero_indices = [5, 15, 30, 45, 60, 75]
    variances[zero_indices] = 0.0

    data = sc.DataGroup(
        {
            'coords': {
                'Qz_0': sc.array(dims=['Qz_0'], values=qz, unit=sc.Unit('1/angstrom')),
            },
            'data': {
                'R_0': sc.array(dims=['Qz_0'], values=r_noisy, variances=variances),
            },
        }
    )
    return data, zero_indices


def run_fit(objective, data):
    """Build a fresh model and fit *data* using the given objective."""
    _, model = build_sample_and_model()
    fitter = MultiFitter(model, objective=objective)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fitter.fit(data)

    return result, fitter, caught


def plot_fit_comparison(data, fit_summaries, zero_indices):
    """Plot the experiment and all fitted curves on a single matplotlib chart."""
    qz = data['coords']['Qz_0'].values
    reflectivity = data['data']['R_0'].values
    variances = data['data']['R_0'].variances
    yerr = np.sqrt(np.clip(variances, 0.0, None)) if variances is not None else None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        qz,
        reflectivity,
        yerr=yerr,
        fmt='o',
        ms=4,
        color='black',
        alpha=0.65,
        capsize=2,
        label='Experiment',
    )
    ax.scatter(
        qz[zero_indices],
        reflectivity[zero_indices],
        s=80,
        facecolors='none',
        edgecolors='crimson',
        linewidths=1.5,
        label='Zero-variance points',
        zorder=4,
    )

    for summary in fit_summaries:
        ax.plot(
            qz,
            summary['model_curve'],
            linewidth=2,
            label=(
                f"{summary['objective']} fit "
                f"(obj={summary['objective_reduced_chi']:.3g}, class={summary['classical_reduced_chi']:.3g})"
            ),
        )

    ax.set_title('Zero-Variance Objective Comparison')
    ax.set_xlabel('Qz (1/angstrom)')
    ax.set_ylabel('Reflectivity')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    # --- setup ---------------------------------------------------------------
    _, seed_model = build_sample_and_model()
    data, zero_idx = make_synthetic_data_with_zero_variances(seed_model)

    total_points = len(data['data']['R_0'].values)
    print(f'Dataset: {total_points} points, {len(zero_idx)} with zero variance\n')

    # --- fit with each objective --------------------------------------------
    objectives = ['legacy_mask', 'hybrid', 'mighell']
    fit_summaries = []

    for obj in objectives:
        print(f'{"=" * 60}')
        print(f'Objective: {obj}')
        print(f'{"=" * 60}')

        result, fitter, caught = run_fit(obj, data)

        # Show warnings emitted during fitting
        for w in caught:
            print(f'  [WARNING] {w.message}')

        print(f'  Success      : {result["success"]}')
        print(f'  Objective reduced chi2 : {result["objective_reduced_chi"]:.6f}')
        print(f'  Objective total chi2   : {fitter.objective_chi2:.6f}')
        print(f'  Classical reduced chi2 : {result["classical_reduced_chi"]:.6f}')
        print(f'  Classical total chi2   : {fitter.classical_chi2:.6f}')
        print()

        fit_summaries.append(
            {
                'objective': obj,
                'model_curve': result['R_0_model'].values,
                'objective_reduced_chi': float(result['objective_reduced_chi']),
                'classical_reduced_chi': float(result['classical_reduced_chi']),
            }
        )

    # --- interpretation note -------------------------------------------------
    print('-' * 60)
    print('NOTE: Under the mighell objective ALL points are transformed,')
    print('so objective chi2 is not a classical chi-square statistic.')
    print('Under hybrid, only zero-variance points are transformed;')
    print('the classical chi2 is still computed from the original')
    print('positive-variance points for comparison.')
    print('-' * 60)

    plot_fit_comparison(data, fit_summaries, zero_idx)


if __name__ == '__main__':
    main()
