import numpy as np
import scipp as sc
from orsopy.fileio import Header
from orsopy.fileio import orso
from orsopy.fileio.base import ComplexValue

from easyreflectometry.data import DataSet1D

from .sample.assemblies.multilayer import Multilayer
from .sample.collections.sample import Sample
from .sample.elements.layers.layer import Layer
from .sample.elements.materials.material import Material
from .sample.elements.materials.material_density import MaterialDensity


def LoadOrso(orso_str: str):
    """Load a model from an ORSO file."""

    sample = load_orso_model(orso_str)
    data = load_orso_data(orso_str)

    return sample, data

def load_data_from_orso_file(fname: str) -> sc.DataGroup:
    """Load data from an ORSO file."""
    try:
        orso_data = orso.load_orso(fname)
    except Exception as e:
        raise ValueError(f"Error loading ORSO file: {e}")
    return load_orso_data(orso_data)

def load_orso_model(orso_str: str) -> Sample:
    """Load a model from an ORSO file and return a Sample object."""
    erl_layers = []
    # Extract material from the layers
    layers = orso_str[0].info.data_source.sample.model.layers
    for layer in layers:
        #layer_name = layer
        # extract material properties
        # sld and isld
        if layers[layer].material.sld is None:
            m_sld = 0.0
            m_isld = 0.0
        else:
            if isinstance(layers[layer].material.sld, ComplexValue):
                m_sld = layers[layer].material.sld.real
                m_isld = layers[layer].material.sld.imag
            else:
                m_sld = layers[layer].material.sld
                m_isld = 0.0
        m_name = layers[layer].material.formula
        # mass density
        m_density = None
        if hasattr(layers[layer].material, 'mass_density') and layers[layer].material.mass_density is not None:
            m_density = layers[layer].material.mass_density
            density = MaterialDensity(
                chemical_structure=m_name,
                density=m_density
            )
            m_sld = density.sld.value
            m_isld = density.isld.value
        # name
        if m_name is None:
            m_name = layer
        print(f"Layer: {layer}, Material SLD: {m_sld}, Material ISLD: {m_isld}, Name: {m_name}")

        # extract layer properties
        layer_thickness = layers[layer].thickness
        layer_roughness = layers[layer].roughness

        # Create a Layer object with the extracted properties and material
        erl_layer = Layer(
            material=Material(sld=m_sld, isld=m_isld, name=m_name),
            thickness=layer_thickness,
            roughness=layer_roughness,
            name=layer
        )
        erl_layers.append(erl_layer)

    # Create a Multilayer object with the extracted layers
    orso_layers = Multilayer(erl_layers, name='Multi Layer Sample from ORSO')

    # Sample from the file
    sample = orso_str[0].info.data_source.sample
    sample_name = sample.name if sample.name else 'ORSO Sample'
    # sample_stack = sample.model.stack # e.g 'air | m1 | SiO2 | Si'
    sample = Sample(orso_layers, name=sample_name)
    return sample

def load_orso_data(orso_str: str) -> DataSet1D:
    data = {}
    coords = {}
    attrs = {}
    for i, o in enumerate(orso_str):
        name = i
        if o.info.data_set is not None:
            name = o.info.data_set
        coords[f'Qz_{name}'] = sc.array(
            dims=[f'{o.info.columns[0].name}_{name}'],
            values=o.data[:, 0],
            variances=np.square(o.data[:, 3]),
            unit=sc.Unit(o.info.columns[0].unit),
        )
        try:
            data[f'R_{name}'] = sc.array(
                dims=[f'{o.info.columns[0].name}_{name}'],
                values=o.data[:, 1],
                variances=np.square(o.data[:, 2]),
                unit=sc.Unit(o.info.columns[1].unit),
            )
        except TypeError:
            data[f'R_{name}'] = sc.array(
                dims=[f'{o.info.columns[0].name}_{name}'],
                values=o.data[:, 1],
                variances=np.square(o.data[:, 2]),
            )
        attrs[f'R_{name}'] = {'orso_header': sc.scalar(Header.asdict(o.info))}
    data_group = sc.DataGroup(data=data, coords=coords, attrs=attrs)
    return data_group
    # if 'Qz_spin_up' in data_group['coords']:
    #     component = 'spin_up'
    # else:
    #     component = '0'
    # dataset = DataSet1D(
    #     x=data_group['coords'][f'Qz_{component}'].values,
    #     y=data_group['data'][f'R_{component}'].values,
    #     ye=data_group['data'][f'R_{component}'].variances,
    #     xe=data_group['coords'][f'Qz_{component}'].variances,
    # )
    # return dataset

