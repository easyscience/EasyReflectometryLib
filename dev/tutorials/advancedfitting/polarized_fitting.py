# %% [markdown]
# # Polarized Neutron Reflectometry: Channels, Depth Profiles & Simultaneous Fitting
#
# `magnetism.ipynb` (in the *Simulation* section) introduces magnetic layers and
# how to select a single polarization channel. This tutorial picks up from
# there and focuses on what is new for **polarization analysis**:
#
# 1. Computing all four spin cross-sections (`pp`, `pm`, `mp`, `mm`) in a single,
#    stateless call.
# 2. Reading off the spin-resolved depth profile — the potential each neutron
#    spin state actually sees.
# 3. Loading a polarized experiment from per-channel data files and forming the
#    **spin asymmetry**, with proper error propagation.
# 4. **Fitting multiple polarization channels simultaneously** against one
#    shared model with `MultiFitter.fit_polarized()` — the headline new
#    capability — first recovering just the moment's magnitude from the two
#    non-spin-flip channels, then recovering the full magnetization vector
#    (magnitude *and* direction) from all four channels.
#
# Only the `refl1d` calculator supports magnetism; `refnx` and `bornagain` do
# not. All magnetism handling — enabling it on the calculator, computing
# channels, fitting — goes through `refl1d`.
#
# The reflectometry convention used throughout: with the default guide field,
# a moment at ``theta_m = 270`` degrees is aligned with it (no spin-flip
# scattering); ``theta_m = 90`` is anti-aligned. A **canted** moment away from
# 270/90 produces spin-flip scattering (`pm`, `mp`) alongside the
# non-spin-flip channels (`pp`, `mm`) — which is why the sample below uses
# ``theta_m = 45`` degrees rather than a value aligned with the guide field:
# it is the only choice that makes all four channels visually distinct.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

print('All libraries imported successfully.')

# %% [markdown]
# ## 1. Build a magnetic sample
#
# A single magnetic layer between two non-magnetic media: a thin Fe film
# (nuclear SLD 4, in units of $10^{-6}$ Å$^{-2}$) with an in-plane magnetic
# moment, sitting on a Si substrate below a vacuum superphase.
#
# Magnetism is attached with `Layer.magnetism = LayerMagnetism(rho_m=..., theta_m=...)`
# (or passed directly to `Layer(..., magnetism=...)`, as below). This
# automatically enables magnetism on the calculator once the layer has an
# interface — there is no separate "turn magnetism on" step required.
#
# `LayerMagnetism` exposes two ordinary, fittable `Parameter`s:
#
# | Parameter | Meaning                                | Unit                       | Default |
# |-----------|-----------------------------------------|-----------------------------|---------|
# | `rho_m`   | Magnetic scattering length density      | $10^{-6}$ Å$^{-2}$          | 0.0     |
# | `theta_m` | In-plane moment angle vs. the beam      | degree                      | 270.0   |
#
# We wrap sample construction in a function so the same recipe can be reused
# below to build a "truth" model and, later, independent "fit starting point"
# models.


# %%
def build_magnetic_model(rho_m: float, theta_m: float, name: str) -> tuple[Model, Layer]:
    """Build a Vacuum / Fe(magnetic) / Si model and return it with the magnetic layer.

    :param rho_m: Magnetic SLD of the Fe film, in 1e-6/angstrom^2.
    :param theta_m: In-plane moment angle of the Fe film, in degrees.
    :param name: Name for the model.
    :return: The model (interface already switched to refl1d) and the Fe layer,
        so its ``.magnetism`` parameters can be reached directly for fitting.
    """
    vacuum = Material(sld=0, isld=0, name='Vacuum')
    iron = Material(sld=4.0, isld=0, name='Fe')
    silicon = Material(sld=2.047, isld=0, name='Si')

    superphase = Layer(material=vacuum, thickness=0, roughness=0, name='Vacuum Superphase')
    film = Layer(
        material=iron,
        thickness=100,
        roughness=0,
        magnetism=LayerMagnetism(rho_m=rho_m, theta_m=theta_m, name='Fe film moment'),
        name='Fe Film',
    )
    subphase = Layer(material=silicon, thickness=0, roughness=0, name='Si Subphase')

    sample = Sample(Multilayer(superphase), Multilayer(film), Multilayer(subphase), name='Vacuum / Fe(magnetic) / Si')
    model = Model(sample=sample, scale=1, background=0, name=name)
    model.resolution_function = PercentageFwhm(0)  # 0% resolution keeps this simulation clean

    interface = CalculatorFactory()
    interface.switch('refl1d')  # the only calculator that supports magnetism
    model.interface = interface

    return model, film


RHO_M_TRUE = 2.5  # 1e-6 / angstrom^2
THETA_M_TRUE = 45.0  # degrees -- canted, so all four channels differ

truth_model, truth_film = build_magnetic_model(RHO_M_TRUE, THETA_M_TRUE, name='Truth model')

print(f'Fe film magnetism: rho_m = {truth_film.magnetism.rho_m.value}, theta_m = {truth_film.magnetism.theta_m.value}')

# %% [markdown]
# ## 2. All four polarization channels, one call
#
# `model.interface.polarized_reflectivity_profiles(q, model_name)` returns a
# dict with all four spin cross-sections at once:
#
# - `'pp'` — non-spin-flip, up-up
# - `'mm'` — non-spin-flip, down-down
# - `'pm'` — spin-flip, up-down
# - `'mp'` — spin-flip, down-up
#
# Internally this is a single `refl1d` kernel evaluation shared by all four
# channels (and cached per model state), so this costs about the same as
# computing one channel. For a single explicit channel without touching any
# calculator state, use `reflectivity_profile_channel(q, model_name, channel)`
# instead — both are stateless, unlike setting `interface.polarization_channel`.

# %%
Q_PLOT = np.linspace(0.001, 0.3, 500)

channels_truth = truth_model.interface.polarized_reflectivity_profiles(Q_PLOT, truth_model.unique_name)

plt.figure(figsize=(8, 5))
plt.semilogy(Q_PLOT, channels_truth['pp'], '-k', label='pp (non-spin-flip)', linewidth=2)
plt.semilogy(Q_PLOT, channels_truth['mm'], '-r', label='mm (non-spin-flip)', linewidth=2)
plt.semilogy(Q_PLOT, channels_truth['pm'], ':k', label='pm (spin-flip)', linewidth=2)
plt.semilogy(Q_PLOT, channels_truth['mp'], ':r', label='mp (spin-flip)', linewidth=2)
plt.xlabel('Q / Å⁻¹')
plt.ylabel('Reflectivity')
plt.title(f'Four polarization channels (rho_m={RHO_M_TRUE}, theta_m={THETA_M_TRUE}°)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 3. The spin-resolved depth profile
#
# `Project.magnetic_sld_data_for_model_at_index()` returns the nuclear SLD
# profile alongside the two potentials a neutron in each spin state actually
# experiences: `spin_up = sld + rho_m * cos(theta_m - guide_field_angle)` and
# `spin_down = sld - rho_m * cos(...)`. With a canted moment (not aligned with
# the guide field) the split between the two curves is reduced by that cosine
# factor rather than being the full `rho_m`.
#
# This wraps the model in a `Project`, the same object a GUI application uses
# to manage models, experiments and fitting — and is the entry point for the
# rest of this tutorial too.

# %%
project = Project()
project.calculator = 'refl1d'
project.models = ModelCollection(truth_model)

profiles = project.magnetic_sld_data_for_model_at_index(0)

plt.figure(figsize=(8, 5))
plt.plot(profiles['sld'].x, profiles['sld'].y, '-k', label='Nuclear SLD', linewidth=2)
plt.plot(profiles['spin_up'].x, profiles['spin_up'].y, '-b', label='Spin-up potential', linewidth=2)
plt.plot(profiles['spin_down'].x, profiles['spin_down'].y, '-r', label='Spin-down potential', linewidth=2)
plt.xlabel('z / Å')
plt.ylabel('SLD / 10⁻⁶ Å⁻²')
plt.title('Nuclear SLD and the two spin-dependent potentials')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# The moment magnitude and direction are also available on their own, restricted
# to the depths that actually carry a moment (an angle is meaningless at zero SLD).
print(f'Peak magnetic SLD in the film: {profiles["rho_m"].y.max():.3f} (expected {RHO_M_TRUE})')
print(f'Moment angle inside the film:  {profiles["theta_m"].y.mean():.1f}° (expected {THETA_M_TRUE}°)')

# %% [markdown]
# ## 4. Loading a polarized experiment and forming the spin asymmetry
#
# A polarized measurement typically arrives as one data file per spin channel
# (e.g. `..._uu.dat` for up-up, `..._dd.dat` for down-down). Here we simulate
# that by writing the truth model's `pp`/`mm` reflectivity, with 1% relative
# noise, to two files, then loading them back exactly as a user would with
# real instrument output.
#
# The **spin asymmetry** $SA = (R^{++} - R^{--}) / (R^{++} + R^{--})$ is a
# common way to look at polarized data directly: it cancels the non-magnetic
# (nuclear) part of the reflectivity and isolates the magnetic signal, with
# `Project.spin_asymmetry_for_experiment_at_index()` handling the variance
# propagation and dropping points where the denominator is too small to be
# meaningful.


# %%
def add_relative_noise(
    reflectivity: np.ndarray, relative_sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Add reproducible Gaussian noise scaled to a fixed fraction of the signal.

    :param reflectivity: Noise-free reflectivity values.
    :param relative_sigma: Standard deviation as a fraction of the signal (e.g. 0.01 for 1%).
    :param rng: Seeded random number generator, for reproducible tutorial output.
    :return: Noisy reflectivity and the per-point standard deviation (not variance).
    """
    sigma = relative_sigma * np.abs(reflectivity)
    noisy = rng.normal(loc=reflectivity, scale=sigma)
    return np.clip(noisy, 1e-12, None), sigma


Q_DATA = np.linspace(0.01, 0.25, 60)  # a more realistic, instrument-like grid
rng = np.random.default_rng(seed=42)  # fixed seed: this tutorial's output is reproducible

channels_data_grid = truth_model.interface.polarized_reflectivity_profiles(Q_DATA, truth_model.unique_name)
noisy_channels = {
    channel: add_relative_noise(reflectivity, relative_sigma=0.01, rng=rng)
    for channel, reflectivity in channels_data_grid.items()
}

tmp_dir = Path(tempfile.mkdtemp(prefix='easyreflectometry_polarized_'))
pp_path = tmp_dir / 'fe_film_uu.txt'
mm_path = tmp_dir / 'fe_film_dd.txt'
np.savetxt(pp_path, np.column_stack([Q_DATA, noisy_channels['pp'][0], noisy_channels['pp'][1]]))
np.savetxt(mm_path, np.column_stack([Q_DATA, noisy_channels['mm'][0], noisy_channels['mm'][1]]))

# The filename suffixes ('_uu', '_dd') are recognised automatically.
print(project.suggest_polarized_channel_assignment([pp_path, mm_path]))

experiment_index = project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})
loaded_channels = project.experiment_channels_at_index(experiment_index)
print(f'Loaded polarized experiment at index {experiment_index}, channels: {loaded_channels}')

# %%
spin_asymmetry = project.spin_asymmetry_for_experiment_at_index(experiment_index)
measured, calculated = spin_asymmetry['measured'], spin_asymmetry['calculated']

plt.figure(figsize=(8, 5))
plt.errorbar(
    measured.x,
    measured.y,
    yerr=np.sqrt(measured.ye),
    fmt='o',
    color='0.3',
    markersize=4,
    alpha=0.6,
    label='Measured (loaded files)',
)
plt.plot(calculated.x, calculated.y, '-r', linewidth=2, label='Calculated (truth model)')
plt.xlabel('Q / Å⁻¹')
plt.ylabel('Spin asymmetry')
plt.title('Spin asymmetry: loaded data vs. the model it was generated from')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(
    f'{measured.x.size} of {Q_DATA.size} points kept '
    f'({spin_asymmetry["masked_points"]} masked: '
    f'{spin_asymmetry["low_significance_points"]} low-significance, '
    f'{spin_asymmetry["small_denominator_points"]} small-denominator).'
)

# %% [markdown]
# ## 5. Fitting: recovering the moment magnitude from two channels
#
# The most common polarized experiment measures only the two non-spin-flip
# channels (`pp`, `mm`). If the moment's *direction* is already known from
# other means (sample geometry, prior characterization), that alone is enough
# to recover its *magnitude* — this is the everyday polarized-fitting case.
#
# `MultiFitter.fit_polarized()` takes a `PolarizedDataSet` and fits every
# channel it contains **simultaneously against one shared model**: any
# structural parameter (thickness, roughness, nuclear SLD, scale, background)
# is constrained jointly by all measured channels, and so is `rho_m`/`theta_m`.
# Internally `refl1d` still evaluates all cross-sections from a single kernel
# call, so fitting N channels together costs about as much as fitting one.
#
# We start from a deliberately wrong `rho_m` guess and fit against the two
# noisy channels loaded above; `theta_m` stays fixed at its (assumed known)
# true value.

# %%
fit_model_2ch, fit_film_2ch = build_magnetic_model(rho_m=1.0, theta_m=THETA_M_TRUE, name='Fit: two channels (rho_m only)')

fit_film_2ch.magnetism.rho_m.fixed = False
fit_film_2ch.magnetism.rho_m.bounds = (0.0, 5.0)
fit_film_2ch.magnetism.theta_m.fixed = True  # moment direction assumed known

initial_channels_2ch = fit_model_2ch.interface.polarized_reflectivity_profiles(Q_PLOT, fit_model_2ch.unique_name)

fit_data_2ch = PolarizedDataSet(
    name='Fe film (pp, mm)',
    channels={
        'pp': DataSet1D(name='pp', x=Q_DATA, y=noisy_channels['pp'][0], ye=noisy_channels['pp'][1] ** 2),
        'mm': DataSet1D(name='mm', x=Q_DATA, y=noisy_channels['mm'][0], ye=noisy_channels['mm'][1] ** 2),
    },
    model=fit_model_2ch,  # PolarizedDataSet.model must be the model the fitter is constructed with
)

fitter_2ch = MultiFitter(fit_model_2ch)
results_2ch = fitter_2ch.fit_polarized(fit_data_2ch)

print(f'Channels fitted: {list(results_2ch.keys())}, all successful: {all(r.success for r in results_2ch.values())}')
print(f'rho_m: {fit_film_2ch.magnetism.rho_m.value:.3f} (started at 1.0, true value {RHO_M_TRUE})')
print(f'Reduced chi^2: {fitter_2ch.reduced_chi:.3f}')

fitted_channels_2ch = fit_model_2ch.interface.polarized_reflectivity_profiles(Q_PLOT, fit_model_2ch.unique_name)

# %%
plt.figure(figsize=(8, 5))
plt.errorbar(
    Q_DATA,
    noisy_channels['pp'][0],
    yerr=noisy_channels['pp'][1],
    fmt='o',
    color='0.3',
    markersize=4,
    alpha=0.5,
    label='pp (data)',
)
plt.errorbar(
    Q_DATA,
    noisy_channels['mm'][0],
    yerr=noisy_channels['mm'][1],
    fmt='s',
    color='0.6',
    markersize=4,
    alpha=0.5,
    label='mm (data)',
)
plt.semilogy(Q_PLOT, initial_channels_2ch['pp'], '--k', linewidth=1, alpha=0.6, label='pp (initial guess)')
plt.semilogy(Q_PLOT, initial_channels_2ch['mm'], '--r', linewidth=1, alpha=0.6, label='mm (initial guess)')
plt.semilogy(Q_PLOT, fitted_channels_2ch['pp'], '-k', linewidth=2, label='pp (fitted)')
plt.semilogy(Q_PLOT, fitted_channels_2ch['mm'], '-r', linewidth=2, label='mm (fitted)')
plt.yscale('log')
plt.xlabel('Q / Å⁻¹')
plt.ylabel('Reflectivity')
plt.title('Two-channel fit: rho_m recovered from pp and mm together')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 6. Fitting: recovering the full magnetization vector from four channels
#
# When the spin-flip channels (`pm`, `mp`) are also measured, `fit_polarized`
# can determine the moment's *direction* as well as its magnitude — both
# `rho_m` and `theta_m` are freed and constrained jointly by all four
# channels. This is the distinguishing capability of full polarization
# analysis over a non-spin-flip-only measurement.
#
# We start from wrong guesses for **both** parameters this time.

# %%
fit_model_4ch, fit_film_4ch = build_magnetic_model(rho_m=1.5, theta_m=60.0, name='Fit: four channels (rho_m and theta_m)')

fit_film_4ch.magnetism.rho_m.fixed = False
fit_film_4ch.magnetism.rho_m.bounds = (0.0, 5.0)
fit_film_4ch.magnetism.theta_m.fixed = False
fit_film_4ch.magnetism.theta_m.bounds = (0.0, 90.0)

fit_data_4ch = PolarizedDataSet(
    name='Fe film (pp, pm, mp, mm)',
    channels={
        channel: DataSet1D(name=channel, x=Q_DATA, y=values[0], ye=values[1] ** 2) for channel, values in noisy_channels.items()
    },
    model=fit_model_4ch,
)

fitter_4ch = MultiFitter(fit_model_4ch)
results_4ch = fitter_4ch.fit_polarized(fit_data_4ch)

print(f'Channels fitted: {list(results_4ch.keys())}, all successful: {all(r.success for r in results_4ch.values())}')
print(f'rho_m:   {fit_film_4ch.magnetism.rho_m.value:.3f} (started at 1.5, true value {RHO_M_TRUE})')
print(f'theta_m: {fit_film_4ch.magnetism.theta_m.value:.1f}° (started at 60.0°, true value {THETA_M_TRUE}°)')
print(f'Reduced chi^2: {fitter_4ch.reduced_chi:.3f}')

# %% [markdown]
# ## Summary
#
# New polarization API demonstrated in this tutorial:
#
# - `Layer(..., magnetism=LayerMagnetism(rho_m=, theta_m=))` — attach a
#   fittable magnetic moment to a layer; only `refl1d` supports it.
# - `model.interface.polarized_reflectivity_profiles(q, model_name)` — all
#   four spin cross-sections (`pp`, `pm`, `mp`, `mm`) from one call.
# - `model.interface.reflectivity_profile_channel(q, model_name, channel)` —
#   a single explicit channel, without touching calculator state.
# - `Project.magnetic_sld_data_for_model_at_index()` — nuclear SLD plus the
#   spin-up/spin-down potentials.
# - `Project.load_polarized_experiment({'pp': path, 'mm': path, ...})` and
#   `Project.suggest_polarized_channel_assignment(paths)` — load and
#   auto-detect per-channel data files.
# - `Project.spin_asymmetry_for_experiment_at_index()` — spin asymmetry with
#   proper error propagation and physically-motivated point masking.
# - `PolarizedDataSet(channels={...}, model=model)` and
#   `MultiFitter(model).fit_polarized(data)` — fit any number of measured
#   channels simultaneously against one shared model.
#
# See `docs/docs/tutorials/simulation/magnetism.ipynb` for the basics of
# building magnetic samples and selecting a single channel, and
# `tests/test_polarized_fitting.py` for the full, exhaustively-tested API
# surface this tutorial draws on.
