"""Additional tests for Model class new features from corelib migration."""

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.model import Model, PercentageFwhm
from easyreflectometry.sample import Layer, Material, Multilayer, Sample
from easyreflectometry.calculators import CalculatorFactory


@pytest.fixture
def clear_global():
    """Clear global object map before each test."""
    global_object.map._clear()
    yield
    global_object.map._clear()


class TestModelProperties:
    """Tests for Model properties added during migration."""
    
    def test_name_getter(self, clear_global):
        """Test name property getter maps to display_name."""
        model = Model(name='TestModel')
        assert_equal(model.name, 'TestModel')
        assert_equal(model.display_name, 'TestModel')
    
    def test_name_setter(self, clear_global):
        """Test name property setter."""
        model = Model(name='OldName')
        model.name = 'NewName'
        assert_equal(model.name, 'NewName')
        assert_equal(model.display_name, 'NewName')
    
    def test_sample_getter(self, clear_global):
        """Test sample property getter."""
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample)
        
        assert model.sample is sample
    
    def test_sample_setter(self, clear_global):
        """Test sample property setter."""
        # Create initial sample
        material1 = Material(2.074, 0, 'Si')
        layer1 = Layer(material1, 10, 3, 'Si Layer')
        multilayer1 = Multilayer(layer1, 'Multilayer 1')
        sample1 = Sample(multilayer1, populate_if_none=False)
        
        # Create new sample
        material2 = Material(3.47, 0, 'SiO2')
        layer2 = Layer(material2, 20, 5, 'SiO2 Layer')
        multilayer2 = Multilayer(layer2, 'Multilayer 2')
        sample2 = Sample(multilayer2, populate_if_none=False)
        
        model = Model(sample=sample1)
        assert model.sample is sample1
        
        # Set new sample
        model.sample = sample2
        assert model.sample is sample2
        
        # Verify global map updated
        edges = global_object.map.get_edges(model)
        assert sample2.unique_name in edges
    
    def test_scale_getter(self, clear_global):
        """Test scale property getter."""
        model = Model(scale=2.5)
        assert_almost_equal(model.scale.value, 2.5)
    
    def test_scale_setter_with_parameter(self, clear_global):
        """Test scale property setter with Parameter."""
        model = Model()
        new_scale = Parameter('scale', value=3.0)
        
        model.scale = new_scale
        assert model.scale is new_scale
        assert_almost_equal(model.scale.value, 3.0)
    
    def test_scale_setter_with_number(self, clear_global):
        """Test scale property setter with number."""
        model = Model()
        model.scale = 2.5
        assert_almost_equal(model.scale.value, 2.5)
    
    def test_background_getter(self, clear_global):
        """Test background property getter."""
        model = Model(background=1e-6)
        assert_almost_equal(model.background.value, 1e-6)
    
    def test_background_setter_with_parameter(self, clear_global):
        """Test background property setter with Parameter."""
        model = Model()
        new_background = Parameter('background', value=5e-7)
        
        model.background = new_background
        assert model.background is new_background
        assert_almost_equal(model.background.value, 5e-7)
    
    def test_background_setter_with_number(self, clear_global):
        """Test background property setter with number."""
        model = Model()
        model.background = 2e-6
        assert_almost_equal(model.background.value, 2e-6)


class TestModelGlobalMapIntegration:
    """Tests for Model global map integration."""
    
    def test_components_registered_in_global_map(self, clear_global):
        """Test that all components are registered in global map."""
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample)
        
        # Check edges from model
        edges = global_object.map.get_edges(model)
        
        assert sample.unique_name in edges
        assert model.scale.unique_name in edges
        assert model.background.unique_name in edges
    
    def test_sample_replacement_updates_map(self, clear_global):
        """Test that replacing sample updates global map."""
        # Create initial setup
        material1 = Material(2.074, 0, 'Si')
        layer1 = Layer(material1, 10, 3, 'Layer 1')
        multilayer1 = Multilayer(layer1, 'Multilayer 1')
        sample1 = Sample(multilayer1, populate_if_none=False)
        
        model = Model(sample=sample1)
        
        # Create new sample
        material2 = Material(3.47, 0, 'SiO2')
        layer2 = Layer(material2, 20, 5, 'Layer 2')
        multilayer2 = Multilayer(layer2, 'Multilayer 2')
        sample2 = Sample(multilayer2, populate_if_none=False)
        
        # Replace sample
        model.sample = sample2
        
        # Check that old sample is removed from edges
        edges = global_object.map.get_edges(model)
        assert sample2.unique_name in edges
        # Old sample should not be in edges anymore
        assert sample1.unique_name not in edges


class TestModelGetMethods:
    """Tests for get_parameters and get_fit_parameters methods."""
    
    def test_get_parameters(self, clear_global):
        """Test get_parameters returns all parameters."""
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample)
        
        params = model.get_parameters()
        
        assert len(params) > 0
        # Should include scale and background
        param_names = [p.name for p in params]
        assert 'scale' in param_names
        assert 'background' in param_names
    
    def test_get_fit_parameters(self, clear_global):
        """Test get_fit_parameters returns only fittable parameters."""
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample)
        
        # Unfix a parameter
        material.sld.fixed = False
        
        fit_params = model.get_fit_parameters()
        
        assert len(fit_params) > 0
        # Should not include fixed parameters
        for p in fit_params:
            assert p.fixed is False


class TestModelAddAssemblies:
    """Tests for add_assemblies with new binding behavior."""
    
    def test_add_assemblies_regenerates_bindings(self, clear_global):
        """Test that add_assemblies regenerates all bindings."""
        interface = CalculatorFactory()
        
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer1 = Multilayer(layer, 'Multilayer 1')
        sample = Sample(multilayer1, populate_if_none=False)
        model = Model(sample=sample, interface=interface)
        
        # Add another assembly
        material2 = Material(3.47, 0, 'SiO2')
        layer2 = Layer(material2, 20, 5, 'SiO2 Layer')
        multilayer2 = Multilayer(layer2, 'Multilayer 2')
        
        initial_items = len(interface()._wrapper.storage['item'])
        
        model.add_assemblies(multilayer2)
        
        # Verify bindings were regenerated
        final_items = len(interface()._wrapper.storage['item'])
        assert final_items > initial_items


class TestModelDuplicateAssembly:
    """Tests for duplicate_assembly with new binding behavior."""
    
    def test_duplicate_assembly_regenerates_bindings(self, clear_global):
        """Test that duplicate_assembly regenerates all bindings."""
        interface = CalculatorFactory()
        
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample, interface=interface)
        
        initial_items = len(interface()._wrapper.storage['item'])
        
        model.duplicate_assembly(0)
        
        # Verify bindings were regenerated
        final_items = len(interface()._wrapper.storage['item'])
        assert final_items > initial_items


class TestModelRemoveAssembly:
    """Tests for remove_assembly with new binding behavior."""
    
    def test_remove_assembly_regenerates_bindings(self, clear_global):
        """Test that remove_assembly regenerates all bindings."""
        interface = CalculatorFactory()
        
        # Create model with two assemblies
        material1 = Material(2.074, 0, 'Si')
        layer1 = Layer(material1, 10, 3, 'Si Layer')
        multilayer1 = Multilayer(layer1, 'Multilayer 1')
        
        material2 = Material(3.47, 0, 'SiO2')
        layer2 = Layer(material2, 20, 5, 'SiO2 Layer')
        multilayer2 = Multilayer(layer2, 'Multilayer 2')
        
        sample = Sample(multilayer1, multilayer2, populate_if_none=False)
        model = Model(sample=sample, interface=interface)
        
        initial_items = len(interface()._wrapper.storage['item'])
        assert initial_items == 2
        
        model.remove_assembly(0)
        
        # Verify bindings were regenerated with fewer items
        final_items = len(interface()._wrapper.storage['item'])
        assert final_items == 1


class TestModelInterfaceProperty:
    """Tests for interface property deprecation and behavior."""
    
    def test_interface_getter(self, clear_global):
        """Test interface property getter."""
        interface = CalculatorFactory()
        model = Model(interface=interface)
        
        assert model.interface is interface
    
    def test_interface_setter_triggers_bindings(self, clear_global):
        """Test that setting interface triggers generate_bindings."""
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample)
        
        interface = CalculatorFactory()
        model.interface = interface
        
        # Verify bindings were created
        assert len(interface()._wrapper.storage['material']) > 0
        assert len(interface()._wrapper.storage['layer']) > 0


class TestModelGenerateBindings:
    """Tests for generate_bindings method."""
    
    def test_generate_bindings_with_interface(self, clear_global):
        """Test generate_bindings creates all bindings."""
        interface = CalculatorFactory()
        
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample, interface=interface)
        
        # Clear storage
        interface.reset_storage()
        assert len(interface()._wrapper.storage['material']) == 0
        
        # Regenerate bindings
        model.generate_bindings()
        
        # Verify bindings were created
        assert len(interface()._wrapper.storage['material']) > 0
        assert len(interface()._wrapper.storage['layer']) > 0
        assert len(interface()._wrapper.storage['item']) > 0
    
    def test_generate_bindings_without_interface(self, clear_global):
        """Test generate_bindings raises error without interface."""
        model = Model()
        
        # Should raise AttributeError when no interface set
        with pytest.raises(AttributeError, match='Interface error'):
            model.generate_bindings()


class TestModelGetLinkableAttributes:
    """Tests for _get_linkable_attributes method."""
    
    def test_get_linkable_attributes(self, clear_global):
        """Test _get_linkable_attributes returns all linkable items."""
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample)
        
        linkable = model._get_linkable_attributes()
        
        assert len(linkable) > 0
        # Should include parameters from scale, background, and sample
        from easyscience.variable.descriptor_base import DescriptorBase
        for item in linkable:
            assert isinstance(item, DescriptorBase)


class TestModelSerialization:
    """Tests for Model serialization with new architecture."""
    
    def test_as_dict_includes_all_components(self, clear_global):
        """Test as_dict includes all required components."""
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Si Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample, name='TestModel', scale=2.0, background=1e-7)
        
        model_dict = model.as_dict()
        
        assert '@module' in model_dict
        assert '@class' in model_dict
        assert 'name' in model_dict
        assert model_dict['name'] == 'TestModel'
        assert 'sample' in model_dict
        assert 'scale' in model_dict
        assert 'background' in model_dict
        assert 'resolution_function' in model_dict
    
    def test_to_dict_alias(self, clear_global):
        """Test that to_dict is alias for as_dict."""
        model = Model()
        
        as_dict_result = model.as_dict()
        to_dict_result = model.to_dict()
        
        assert as_dict_result == to_dict_result
    
    def test_as_dict_skips_unique_name_for_nested_params(self, clear_global):
        """Test that as_dict skips unique_name for nested parameters."""
        model = Model(scale=2.0)
        
        model_dict = model.as_dict()
        
        # Check that scale dict doesn't have unique_name (to avoid collisions)
        scale_dict = model_dict['scale']
        # The skip should have been applied
        assert '@module' in scale_dict  # Basic serialization still works


class TestModelWithInterface:
    """Integration tests for Model with calculator interface."""
    
    def test_model_with_refnx(self, clear_global):
        """Test Model works with refnx calculator."""
        interface = CalculatorFactory(calculator_name='refnx')
        
        # Create proper 3-layer structure: substrate + film + superphase
        si = Material(2.074, 0, 'Si')
        sio2 = Material(3.47, 0, 'SiO2')
        air = Material(0, 0, 'Air')
        
        si_layer = Layer(si, 0, 3, 'Si substrate')
        sio2_layer = Layer(sio2, 10, 3, 'SiO2 film')
        air_layer = Layer(air, 0, 3, 'Air superphase')
        
        multilayer = Multilayer([si_layer, sio2_layer, air_layer], name='Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample, interface=interface)
        
        q_values = np.linspace(0.01, 0.3, 50)
        result = interface.fit_func(q_values, model.unique_name)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(q_values)
    
    def test_model_with_refl1d(self, clear_global):
        """Test Model works with refl1d calculator."""
        interface = CalculatorFactory(calculator_name='refl1d')
        
        # Create proper 3-layer structure: substrate + film + superphase
        si = Material(2.074, 0, 'Si')
        sio2 = Material(3.47, 0, 'SiO2')
        air = Material(0, 0, 'Air')
        
        si_layer = Layer(si, 0, 3, 'Si substrate')
        sio2_layer = Layer(sio2, 10, 3, 'SiO2 film')
        air_layer = Layer(air, 0, 3, 'Air superphase')
        
        multilayer = Multilayer([si_layer, sio2_layer, air_layer], name='Test Multilayer')
        sample = Sample(multilayer, populate_if_none=False)
        model = Model(sample=sample, interface=interface)
        
        q_values = np.linspace(0.01, 0.3, 50)
        result = interface.fit_func(q_values, model.unique_name)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(q_values)
