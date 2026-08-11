import pooch
import refl1d
import refnx

import easyreflectometry
from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import load
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

print(f'easyreflectometry: {easyreflectometry.__version__}')
print(f'refnx: {refnx.__version__}')
print(f'refl1d: {refl1d.__version__}')


file_path = pooch.retrieve(
    # Fetch test data from the easyscience/reflectometry data repository
    url='https://raw.githubusercontent.com/easyscience/reflectometry/master/data/example.ort',
    known_hash='82d0c95c069092279a799a8131ad3710335f601d9f1080754b387f42e407dfab',
)
data = load(file_path)

si = Material(sld=2.07, isld=0, name='Si')
sio2 = Material(sld=3.47, isld=0, name='SiO2')
film = Material(sld=2.0, isld=0, name='Film')
d2o = Material(sld=6.36, isld=0, name='D2O')

si_layer = Layer(material=si, thickness=0, roughness=0, name='Si layer')
sio2_layer = Layer(material=sio2, thickness=30, roughness=3, name='SiO2 layer')
film_layer = Layer(material=film, thickness=250, roughness=3, name='Film Layer')
subphase = Layer(material=d2o, thickness=0, roughness=3, name='D2O Subphase')

superphase = Multilayer([si_layer, sio2_layer], name='Si/SiO2 Superphase')

sample = Sample(superphase, Multilayer(film_layer), Multilayer(subphase), name='Film Structure')

resolution_function = PercentageFwhm(0.02)
model = Model(sample=sample, scale=1, background=1e-6, resolution_function=resolution_function, name='Film Model')

# Thicknesses
sio2_layer.thickness.bounds = (15, 50)
film_layer.thickness.bounds = (200, 300)
# Roughnesses
sio2_layer.roughness.bounds = (1, 15)
film_layer.roughness.bounds = (1, 15)
subphase.roughness.bounds = (1, 15)
# Scattering length density
film_layer.material.sld.bounds = (0.1, 3)

# Background
model.background.bounds = (1e-8, 1e-5)
# Scale
model.scale.bounds = (0.5, 1.5)

interface = CalculatorFactory()
model.interface = interface
fitter = MultiFitter(model)

analysed_refnx = fitter.fit(data)
analysed = analysed_refnx


si_refl1d = Material(sld=2.07, isld=0, name='Si')
sio2_refl1d = Material(sld=3.47, isld=0, name='SiO2')
film_refl1d = Material(sld=2.0, isld=0, name='Film')
d2o_refl1d = Material(sld=6.36, isld=0, name='D2O')

si_layer_refl1d = Layer(material=si_refl1d, thickness=0, roughness=0, name='Si layer')
sio2_layer_refl1d = Layer(material=sio2_refl1d, thickness=30, roughness=3, name='SiO2 layer')
film_layer_refl1d = Layer(material=film_refl1d, thickness=250, roughness=3, name='Film Layer')
subphase_refl1d = Layer(material=d2o_refl1d, thickness=0, roughness=3, name='D2O Subphase')

superphase_refl1d = Multilayer([si_layer_refl1d, sio2_layer_refl1d], name='Si/SiO2 Superphase')
sample_refl1d = Sample(superphase_refl1d, Multilayer(film_layer_refl1d), Multilayer(subphase_refl1d), name='Film Structure')

resolution_function_refl1d = PercentageFwhm(0.02)
model_refl1d = Model(
    sample=sample_refl1d, scale=1, background=1e-6, resolution_function=resolution_function_refl1d, name='Film Model (Refl1D)'
)

sio2_layer_refl1d.thickness.bounds = (15, 50)
film_layer_refl1d.thickness.bounds = (200, 300)
sio2_layer_refl1d.roughness.bounds = (1, 15)
film_layer_refl1d.roughness.bounds = (1, 15)
subphase_refl1d.roughness.bounds = (1, 15)
film_layer_refl1d.material.sld.bounds = (0.1, 3)
model_refl1d.background.bounds = (1e-8, 1e-5)
model_refl1d.scale.bounds = (0.5, 1.5)

interface_refl1d = CalculatorFactory()
interface_refl1d.switch('refl1d')
model_refl1d.interface = interface_refl1d

# reach into the calculator's native storage and change the refnx Slab directly
native_slab = interface()._wrapper.storage['layer'][sio2_layer.unique_name]
native_slab.thick.value = 42.0  # engine-side change, core knows nothing yet

print(sio2_layer.thickness.value_no_call_back)  # 30.0  — cached core scalar, no pull
print(sio2_layer.thickness.value)  # 42.0  — fget pulls from refnx and syncs
print(sio2_layer.thickness.value_no_call_back)  # 42.0  — cache now updated by the pull


fitter_refl1d = MultiFitter(model_refl1d)
analysed_refl1d = fitter_refl1d.fit(data)

print(f'refnx reduced chi: {analysed_refnx["reduced_chi"]:.4f}')
print(f'refl1d reduced chi: {analysed_refl1d["reduced_chi"]:.4f}')
