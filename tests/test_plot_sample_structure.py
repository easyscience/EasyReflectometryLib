import pytest  # type: ignore[import-not-found]

from easyreflectometry.plot import plot_sample_structure
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')

def test_plot_sample_structure_draws_layers():
    superphase = Layer(material=Material(sld=0.0, isld=0.0, name='Air'), thickness=0.0, roughness=3.0, name='Air')
    layer_one = Layer(material=Material(sld=2.0, isld=0.0, name='Layer 1'), thickness=10.0, roughness=3.0, name='Layer 1')
    layer_two = Layer(material=Material(sld=4.0, isld=0.0, name='Layer 2'), thickness=20.0, roughness=3.0, name='Layer 2')
    back_layer = Layer(material=Material(sld=6.0, isld=0.0, name='Substrate'), thickness=40.0, roughness=3.0, name='Substrate')

    sample = Sample(
        Multilayer(superphase),
        Multilayer(layer_one),
        Multilayer(layer_two),
        Multilayer(back_layer),
        name='Example sample',
    )

    ax = plot_sample_structure(sample)

    assert len(ax.patches) == 3  # back layer + two finite layers
    texts = [text.get_text() for text in ax.texts]
    assert any('Back layer' in text for text in texts)
    assert any('superphase' in text.lower() for text in texts)

    total_thickness = sum(layer.thickness.value for layer in (back_layer, layer_two, layer_one))
    assert ax.get_ylim()[1] >= total_thickness
