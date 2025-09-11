__author__ = 'github.com/arm61'

import os

import pytest
from easyscience.fitting.minimizers.factory import AvailableMinimizers

import easyreflectometry
from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data.measurement import load
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


@pytest.mark.parametrize('minimizer', [AvailableMinimizers.Bumps, AvailableMinimizers.LMFit])
def test_fitting(minimizer):
    fpath = os.path.join(PATH_STATIC, 'example.ort')
    data = load(fpath)
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
    # Thicknesses
    sio2_layer.thickness.fixed = False
    sio2_layer.thickness.bounds = (15, 50)
    film_layer.thickness.fixed = False
    film_layer.thickness.bounds = (200, 300)
    # Roughnesses
    si_layer.roughness.fixed = True
    sio2_layer.roughness.bounds = (1, 15)
    film_layer.roughness.fixed = False
    film_layer.roughness.bounds = (1, 15)
    superphase.roughness.fixed = True
    superphase.roughness.bounds = (1, 15)
    # Scattering length density
    film.sld.fixed = False
    film.sld.bounds = (0.1, 3)
    # Background
    model.background.fixed = False
    model.background.bounds = (1e-7, 1e-5)
    # Scale
    model.scale.fixed = False
    model.scale.bounds = (0.5, 1.5)
    interface = CalculatorFactory()
    model.interface = interface
    fitter = MultiFitter(model)
    fitter.easy_science_multi_fitter.switch_minimizer(minimizer)
    analysed = fitter.fit(data)
    assert 'R_0_model' in analysed.keys()
    assert 'SLD_0' in analysed.keys()
    assert 'success' in analysed.keys()
    assert analysed['success']
