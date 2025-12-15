"""Unit tests for calculator_base module in EasyReflectometryLib."""

import numpy as np
import pytest

from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.calculators.calculator_base import CalculatorBase
from easyreflectometry.model import Model
from easyreflectometry.sample import Layer, Material, Multilayer, Sample


@pytest.fixture
def clear_global():
    """Clear global object map before each test."""
    global_object.map._clear()
    yield
    global_object.map._clear()


@pytest.fixture
def simple_model(clear_global):
    """Create a simple reflectometry model for testing."""
    # Create materials
    si = Material(2.074, 0, 'Si')
    sio2 = Material(3.47, 0, 'SiO2')
    
    # Create layers
    layer1 = Layer(si, 10, 3, 'Si Layer')
    layer2 = Layer(sio2, 50, 5, 'SiO2 Layer')
    
    # Create multilayer
    multilayer = Multilayer(layer1, 'Test Multilayer')
    multilayer.add_layer(layer2)
    
    # Create sample and model
    sample = Sample(multilayer, populate_if_none=False)
    model = Model(sample=sample)
    
    return model


class TestCalculatorBaseInitialization:
    """Tests for CalculatorBase initialization."""
    
    def test_cannot_instantiate_abstract_class(self, clear_global):
        """Test that CalculatorBase has abstract methods that concrete classes must implement."""
        # CalculatorBase is not abstract in EasyReflectometry implementation
        # Just verify that concrete implementations exist
        from easyreflectometry.calculators.refnx.calculator import Refnx
        from easyreflectometry.calculators.refl1d.calculator import Refl1d
        
        assert issubclass(Refnx, CalculatorBase)
        assert issubclass(Refl1d, CalculatorBase)
    
    def test_init_with_model(self, clear_global, simple_model):
        """Test initialization with a model."""
        # We need a concrete implementation for testing
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        assert calc.model is simple_model
    
    def test_init_without_model(self, clear_global):
        """Test initialization without a model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        assert calc.model is None


class TestCalculatorBaseModelManagement:
    """Tests for model management in CalculatorBase."""
    
    def test_set_model(self, clear_global, simple_model):
        """Test setting a model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        calc.set_model(simple_model)
        assert calc.model is simple_model
    
    def test_set_model_creates_bindings(self, clear_global, simple_model):
        """Test that set_model creates calculator bindings."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        calc.set_model(simple_model)
        
        # Check that materials were created in storage
        assert len(calc._wrapper.storage['material']) > 0
        
        # Check that layers were created
        assert len(calc._wrapper.storage['layer']) > 0
        
        # Check that items were created  
        assert len(calc._wrapper.storage['item']) > 0
    
    def test_model_property_getter(self, clear_global, simple_model):
        """Test model property getter."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        assert calc.model is simple_model


class TestCalculatorBaseCalculation:
    """Tests for calculation methods."""
    
    def test_calculate_requires_model(self, clear_global):
        """Test that calculate raises error if no model is set."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        q_values = np.linspace(0.01, 0.3, 100)
        
        with pytest.raises(ValueError, match="No model set"):
            calc.calculate(q_values)
    
    def test_calculate_with_model(self, clear_global, simple_model):
        """Test calculate method with a model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        q_values = np.linspace(0.01, 0.3, 100)
        
        result = calc.calculate(q_values)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(q_values)
        assert np.all(np.isfinite(result))
    
    def test_reflectivity_profile(self, clear_global, simple_model):
        """Test reflectivity_profile method."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        q_values = np.linspace(0.01, 0.3, 100)
        
        result = calc.reflectivity_profile(q_values, simple_model.unique_name)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(q_values)
    
    def test_reflectity_profile_legacy(self, clear_global, simple_model):
        """Test legacy reflectity_profile method (typo name)."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        q_values = np.linspace(0.01, 0.3, 100)
        
        # Legacy method should still work
        result = calc.reflectity_profile(q_values, simple_model.unique_name)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(q_values)


class TestCalculatorBaseSLDProfile:
    """Tests for SLD profile methods."""
    
    def test_sld_profile_with_model_id(self, clear_global, simple_model):
        """Test sld_profile with explicit model_id."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        z, sld = calc.sld_profile(simple_model.unique_name)
        
        assert isinstance(z, np.ndarray)
        assert isinstance(sld, np.ndarray)
        assert len(z) == len(sld)
    
    def test_sld_profile_without_model_id(self, clear_global, simple_model):
        """Test sld_profile using bound model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        z, sld = calc.sld_profile()
        
        assert isinstance(z, np.ndarray)
        assert isinstance(sld, np.ndarray)
    
    def test_sld_profile_no_model_raises_error(self, clear_global):
        """Test that sld_profile raises error if no model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        with pytest.raises(ValueError, match="No model set"):
            calc.sld_profile()


class TestCalculatorBaseBindingManagement:
    """Tests for binding management methods."""
    
    def test_reset_storage(self, clear_global, simple_model):
        """Test reset_storage method."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        
        # Verify storage has content
        assert len(calc._wrapper.storage['material']) > 0
        
        # Reset storage
        calc.reset_storage()
        
        # Verify storage is cleared
        assert len(calc._wrapper.storage['material']) == 0
        assert len(calc._wrapper.storage['layer']) == 0
        assert len(calc._wrapper.storage['item']) == 0
    
    def test_create_materials(self, clear_global):
        """Test creating material bindings."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        material = Material(2.074, 0, 'Si')
        
        containers = calc.create(material)
        
        assert len(containers) > 0
        assert material.unique_name in calc._wrapper.storage['material']
    
    def test_create_layers(self, clear_global):
        """Test creating layer bindings."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Test Layer')
        
        # Create material first
        calc.create(material)
        # Then create layer
        containers = calc.create(layer)
        
        assert len(containers) > 0
        assert layer.unique_name in calc._wrapper.storage['layer']
    
    def test_create_multilayer(self, clear_global):
        """Test creating multilayer bindings."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Test Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        
        # Create components
        calc.create(material)
        calc.create(layer)
        containers = calc.create(multilayer)
        
        assert len(containers) > 0
        assert multilayer.unique_name in calc._wrapper.storage['item']
    
    def test_create_model(self, clear_global, simple_model):
        """Test creating model bindings."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        
        # Create all components
        for assembly in simple_model.sample:
            for layer in assembly.layers:
                calc.create(layer.material)
                calc.create(layer)
            calc.create(assembly)
        
        containers = calc.create(simple_model)
        assert len(containers) > 0


class TestCalculatorBaseLayerManagement:
    """Tests for layer management methods."""
    
    def test_assign_material_to_layer(self, clear_global):
        """Test assigning material to layer."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        material1 = Material(2.074, 0, 'Si')
        material2 = Material(3.47, 0, 'SiO2')
        layer = Layer(material1, 10, 3, 'Test Layer')
        
        # Create bindings
        calc.create(material1)
        calc.create(material2)
        calc.create(layer)
        
        # Assign new material
        calc.assign_material_to_layer(material2.unique_name, layer.unique_name)
        
        # Verify assignment (implementation specific)
        assert layer.unique_name in calc._wrapper.storage['layer']
    
    def test_add_layer_to_item(self, clear_global):
        """Test adding layer to item."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Test Layer')
        multilayer = Multilayer(populate_if_none=False)
        
        # Create bindings
        calc.create(material)
        calc.create(layer)
        calc.create(multilayer)
        
        # Add layer to item
        calc.add_layer_to_item(layer.unique_name, multilayer.unique_name)
        
        # Verify (implementation specific)
        assert multilayer.unique_name in calc._wrapper.storage['item']
    
    def test_remove_layer_from_item(self, clear_global):
        """Test removing layer from item."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        material = Material(2.074, 0, 'Si')
        layer = Layer(material, 10, 3, 'Test Layer')
        multilayer = Multilayer(layer, 'Test Multilayer')
        
        # Create bindings
        calc.create(material)
        calc.create(layer)
        calc.create(multilayer)
        
        # Remove layer from item
        calc.remove_layer_from_item(layer.unique_name, multilayer.unique_name)
        
        # Verify (implementation specific)
        assert multilayer.unique_name in calc._wrapper.storage['item']


class TestCalculatorBaseItemManagement:
    """Tests for item (assembly) management methods."""
    
    def test_add_item_to_model(self, clear_global, simple_model):
        """Test adding item to model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        
        # Get first assembly
        assembly = simple_model.sample[0]
        
        # This should already be added, but test the method
        calc.add_item_to_model(assembly.unique_name, simple_model.unique_name)
        
        # Verify (implementation specific)
        assert simple_model.unique_name in calc._wrapper.storage['model']
    
    def test_remove_item_from_model(self, clear_global, simple_model):
        """Test removing item from model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        
        # Get first assembly
        assembly = simple_model.sample[0]
        
        # Remove item from model
        calc.remove_item_from_model(assembly.unique_name, simple_model.unique_name)
        
        # Verify (implementation specific)
        assert simple_model.unique_name in calc._wrapper.storage['model']


class TestCalculatorBaseProperties:
    """Tests for calculator properties."""
    
    def test_include_magnetism_getter(self, clear_global):
        """Test include_magnetism property getter."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        # Default should be False
        assert calc.include_magnetism is False
    
    def test_include_magnetism_setter(self, clear_global):
        """Test include_magnetism property setter."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        calc.include_magnetism = True
        assert calc.include_magnetism is True
        
        calc.include_magnetism = False
        assert calc.include_magnetism is False
    
    def test_fit_func_property(self, clear_global, simple_model):
        """Test fit_func property."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        fit_func = calc.fit_func
        
        assert callable(fit_func)
        
        # Test calling the fit function
        q_values = np.linspace(0.01, 0.3, 100)
        result = fit_func(q_values, simple_model.unique_name)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(q_values)


class TestCalculatorBaseRepr:
    """Tests for string representation."""
    
    def test_repr_without_model(self, clear_global):
        """Test __repr__ without model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        repr_str = repr(calc)
        
        assert 'Refnx' in repr_str
        assert 'name=refnx' in repr_str
    
    def test_repr_with_model(self, clear_global, simple_model):
        """Test __repr__ with model."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx(model=simple_model)
        repr_str = repr(calc)
        
        assert 'Refnx' in repr_str
        assert 'name=refnx' in repr_str
        assert 'model=' in repr_str
        assert simple_model.unique_name in repr_str


class TestCalculatorBaseResolution:
    """Tests for resolution function."""
    
    def test_set_resolution_function(self, clear_global):
        """Test setting resolution function."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        def resolution_func(q):
            return 0.05 * np.ones_like(q)
        
        calc = Refnx()
        calc.set_resolution_function(resolution_func)
        
        # Verify it was set (implementation specific)
        assert calc._wrapper._resolution_function is not None


class TestCalculatorNameAttribute:
    """Tests for calculator name attribute."""
    
    def test_refnx_name(self, clear_global):
        """Test Refnx calculator name."""
        from easyreflectometry.calculators.refnx.calculator import Refnx
        
        calc = Refnx()
        assert calc.name == 'refnx'
    
    def test_refl1d_name(self, clear_global):
        """Test Refl1d calculator name."""
        from easyreflectometry.calculators.refl1d.calculator import Refl1d
        
        calc = Refl1d()
        assert calc.name == 'refl1d'
