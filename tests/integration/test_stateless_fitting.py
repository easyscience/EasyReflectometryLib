# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Fitting against the stateless refnx calculator, end to end.

The minimizer writes core ``Parameter`` values on every iteration and then
calls the fit entry point, which rebuilds the refnx structure from the model.
Nothing else connects the two, so these tests pin that the loop converges, that
it never reaches into the backend to read a value, and that a whole fit stays a
single undo step.
"""

import numpy as np
import pytest
import scipp as sc
from easyscience import global_object
from easyscience.fitting.minimizers.factory import AvailableMinimizers

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

TRUE_THICKNESS = 120.0
TRUE_SLD = 3.45


@pytest.fixture
def clear():
    global_object.map._clear()
    global_object.stack.clear()
    global_object.stack.enabled = False
    yield
    global_object.map._clear()
    global_object.stack.clear()
    global_object.stack.enabled = False


def build_model(thickness: float, sld: float, name: str = 'Model') -> Model:
    """Air | film | silicon, with the film thickness and sld free."""
    air = Material(0.0, 0.0, 'Air')
    film = Material(sld, 0.0, 'Film')
    silicon = Material(2.07, 0.0, 'Silicon')
    film_layer = Layer(film, thickness, 3.0, 'FilmLayer')
    sample = Sample(
        Multilayer(Layer(air, 0.0, 0.0, 'AirLayer'), 'Fronting'),
        Multilayer(film_layer, 'Film'),
        Multilayer(Layer(silicon, 0.0, 3.0, 'SiliconLayer'), 'Backing'),
    )
    model = Model(sample, 1.0, 1e-9, PercentageFwhm(1.0), name)
    film_layer.thickness.fixed = False
    film_layer.thickness.min = 50.0
    film_layer.thickness.max = 250.0
    film.sld.fixed = False
    film.sld.min = 1.0
    film.sld.max = 6.0
    return model


def synthetic_data(model: Model, q: np.ndarray) -> sc.DataGroup:
    """Reflectivity of ``model`` on ``q``, in the shape ``MultiFitter`` wants."""
    reflectivity = model.interface().reflectivity_profile(q, model.unique_name)
    return sc.DataGroup({
        'coords': sc.DataGroup({'Qz_0': sc.array(dims=['Qz_0'], values=q)}),
        'data': sc.DataGroup({
            'R_0': sc.array(
                dims=['Qz_0'],
                values=reflectivity,
                variances=(0.02 * reflectivity) ** 2,
            )
        }),
        'attrs': sc.DataGroup({}),
    })


@pytest.fixture
def data_and_model(clear):
    """A dataset generated from the true model, and a perturbed model to fit."""
    q = np.linspace(0.01, 0.3, 120)
    truth = build_model(TRUE_THICKNESS, TRUE_SLD, 'Truth')
    truth.interface = CalculatorFactory()
    data = synthetic_data(truth, q)

    # Start near the truth: reflectivity is strongly multimodal in the layer
    # thickness (each fringe is a local minimum), so a far start says nothing
    # about the machinery under test.
    model = build_model(110.0, 3.0, 'Fitted')
    model.interface = CalculatorFactory()
    return data, model


@pytest.mark.parametrize(
    'minimizer',
    [AvailableMinimizers.LMFit, AvailableMinimizers.Bumps, AvailableMinimizers.DFO],
)
def test_the_fit_recovers_the_true_parameters(data_and_model, minimizer):
    # Given
    data, model = data_and_model
    fitter = MultiFitter(model)
    fitter.easy_science_multi_fitter.switch_minimizer(minimizer)

    # When
    analysed = fitter.fit(data)

    # Then the values live in the model, which is the only store there is
    assert analysed['success']
    assert model.sample[1].layers[0].thickness.value == pytest.approx(TRUE_THICKNESS, rel=0.02)
    assert model.sample[1].layers[0].material.sld.value == pytest.approx(TRUE_SLD, rel=0.02)


def test_the_fit_loop_never_reads_through_the_backend(data_and_model):
    """The pull-on-read path is dead: no parameter can reach a calculator.

    The mirrored-state design read every ``parameter.value`` through the
    backend, twice per parameter per iteration (once in the minimizer's own
    comparison, once in the undo decorator). Nothing binds a parameter to a
    calculator any more, so both sites are plain attribute reads.
    """
    # Given
    data, model = data_and_model
    fitter = MultiFitter(model)

    # When
    fitter.fit(data)

    # Then no parameter carries a wired backend callback; current easyscience
    # always gives a Parameter an inert `_callback` (`property()` with no
    # accessors), future cores drop the attribute altogether.
    for parameter in model.get_all_parameters():
        callback = getattr(parameter, '_callback', None)
        assert callback is None or (callback.fget is None and callback.fset is None)


def test_the_structure_is_rebuilt_for_every_evaluation(data_and_model):
    # Given
    data, model = data_and_model
    calculator = model.interface()
    builds = []
    original_build = calculator._wrapper._build_structure

    def counting_build(model, proxies=None):
        builds.append(model.unique_name)
        return original_build(model, proxies)

    calculator._wrapper._build_structure = counting_build

    # When
    fitter = MultiFitter(model)
    fitter.fit(data)

    # Then every evaluation went through a fresh build of this model
    assert len(builds) > 1
    assert set(builds) == {model.unique_name}


def test_a_whole_fit_is_one_undo_step(data_and_model):
    # Given
    data, model = data_and_model
    thickness = model.sample[1].layers[0].thickness
    before = thickness.value
    global_object.stack.clear()
    global_object.stack.enabled = True

    # When
    MultiFitter(model).fit(data)

    # Then
    assert len(global_object.stack.history) == 1
    assert global_object.stack.undoText() == 'Fitting routine'
    assert thickness.value != before

    # And it reverts in one step
    global_object.stack.undo()
    assert thickness.value == before


def test_the_measurement_path_still_works(data_and_model):
    """`Data`/`Measurement` code calls the factory's fit_func directly."""
    # Given
    data, model = data_and_model
    q = np.linspace(0.01, 0.3, 20)

    # When
    reflectivity = model.interface.fit_func(q, model.unique_name)

    # Then
    assert reflectivity.shape == q.shape
    assert np.all(reflectivity > 0)


def test_two_models_fit_side_by_side_on_one_calculator(clear):
    # Given two models sharing a factory, which is the `Project` layout
    q = np.linspace(0.01, 0.3, 120)
    factory = CalculatorFactory()

    first_truth = build_model(TRUE_THICKNESS, TRUE_SLD, 'FirstTruth')
    first_truth.interface = factory
    first_data = synthetic_data(first_truth, q)

    second_truth = build_model(60.0, 5.0, 'SecondTruth')
    second_truth.interface = factory
    second_data = synthetic_data(second_truth, q)

    first = build_model(110.0, 3.0, 'First')
    second = build_model(65.0, 4.5, 'Second')
    for model in (first, second):
        model.interface = factory

    # When each is fitted against its own data
    MultiFitter(first).fit(first_data)
    MultiFitter(second).fit(second_data)

    # Then the registry kept them apart
    assert first.sample[1].layers[0].thickness.value == pytest.approx(TRUE_THICKNESS, rel=0.02)
    assert second.sample[1].layers[0].thickness.value == pytest.approx(60.0, rel=0.02)
