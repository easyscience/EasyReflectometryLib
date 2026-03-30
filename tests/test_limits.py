import numpy as np
import pytest
from easyscience.variable import Parameter

from easyreflectometry.limits import SCALE_LIMITS
from easyreflectometry.limits import SLD_LIMITS
from easyreflectometry.limits import apply_default_limits


class TestApplyDefaultLimits:
    def test_sld_with_inf_bounds(self):
        param = Parameter('sld', 4.186, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'sld')
        assert param.min == SLD_LIMITS[0]
        assert param.max == SLD_LIMITS[1]

    def test_isld_with_inf_bounds(self):
        param = Parameter('isld', 0.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'isld')
        assert param.min == SLD_LIMITS[0]
        assert param.max == SLD_LIMITS[1]

    def test_sld_preserves_finite_bounds(self):
        param = Parameter('sld', 4.0, min=-2.0, max=8.0)
        apply_default_limits(param, 'sld')
        assert param.min == -2.0
        assert param.max == 8.0

    def test_scale_with_inf_bounds(self):
        param = Parameter('scale', 1.0, min=0, max=np.inf)
        apply_default_limits(param, 'scale')
        assert param.min == SCALE_LIMITS[0]
        assert param.max == SCALE_LIMITS[1]

    def test_scale_preserves_finite_bounds(self):
        param = Parameter('scale', 1.0, min=0.5, max=2.0)
        apply_default_limits(param, 'scale')
        assert param.min == 0.5
        assert param.max == 2.0

    def test_thickness_percentage_limits(self):
        param = Parameter('thickness', 10.0, min=0.0, max=np.inf)
        apply_default_limits(param, 'thickness')
        assert param.min == 0.0  # 0.0 is finite, not overwritten
        assert param.max == 20.0  # 2.0 * 10.0

    def test_thickness_both_inf(self):
        param = Parameter('thickness', 10.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'thickness')
        assert param.min == 5.0  # 0.5 * 10.0
        assert param.max == 20.0  # 2.0 * 10.0

    def test_roughness_percentage_limits(self):
        param = Parameter('roughness', 3.3, min=0.0, max=np.inf)
        apply_default_limits(param, 'roughness')
        assert param.min == 0.0  # 0.0 is finite, not overwritten
        assert param.max == pytest.approx(6.6)  # 2.0 * 3.3

    def test_thickness_zero_value_unchanged(self):
        param = Parameter('thickness', 0.0, min=0.0, max=np.inf)
        apply_default_limits(param, 'thickness')
        assert param.min == 0.0
        assert param.max == np.inf  # unchanged, zero-valued skip

    def test_roughness_zero_value_unchanged(self):
        param = Parameter('roughness', 0.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'roughness')
        assert param.min == -np.inf
        assert param.max == np.inf

    def test_thickness_preserves_finite_bounds(self):
        param = Parameter('thickness', 10.0, min=2.0, max=25.0)
        apply_default_limits(param, 'thickness')
        assert param.min == 2.0
        assert param.max == 25.0

    def test_dependent_parameter_skipped(self):
        independent_param = Parameter('sld_main', 4.0, min=-np.inf, max=np.inf)
        dependent_param = Parameter('sld_dep', 4.0, min=-np.inf, max=np.inf)
        dependent_param.make_dependent_on(dependency_expression='a', dependency_map={'a': independent_param})
        apply_default_limits(dependent_param, 'sld')
        assert np.isinf(dependent_param.min)
        assert np.isinf(dependent_param.max)

    def test_unknown_kind_is_noop(self):
        param = Parameter('foo', 5.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'unknown')
        assert np.isinf(param.min)
        assert np.isinf(param.max)


class TestIntegrationWithConstructors:
    def test_material_gets_sld_limits(self):
        from easyreflectometry.sample.elements.materials.material import Material

        mat = Material(sld=6.36, isld=0.0)
        assert mat.sld.min == SLD_LIMITS[0]
        assert mat.sld.max == SLD_LIMITS[1]
        assert mat.isld.min == SLD_LIMITS[0]
        assert mat.isld.max == SLD_LIMITS[1]

    def test_layer_gets_percentage_limits(self):
        from easyreflectometry.sample.elements.layers.layer import Layer

        layer = Layer(thickness=20.0, roughness=5.0)
        assert layer.thickness.min == 0.0  # 0.0 is finite, kept
        assert layer.thickness.max == 40.0  # 2.0 * 20.0
        assert layer.roughness.min == 0.0  # 0.0 is finite, kept
        assert layer.roughness.max == 10.0  # 2.0 * 5.0

    def test_layer_zero_thickness_unchanged(self):
        from easyreflectometry.sample.elements.layers.layer import Layer

        layer = Layer(thickness=0.0, roughness=0.0)
        assert layer.thickness.min == 0.0
        assert layer.thickness.max == np.inf
        assert layer.roughness.min == 0.0
        assert layer.roughness.max == np.inf

    def test_model_gets_scale_limits(self):
        from easyreflectometry.model.model import Model

        model = Model(scale=1.0)
        assert model.scale.min == SCALE_LIMITS[0]
        assert model.scale.max == SCALE_LIMITS[1]

    def test_existing_parameter_bounds_preserved(self):
        from easyreflectometry.sample.elements.materials.material import Material

        custom_sld = Parameter('sld', 4.0, min=-0.5, max=7.0)
        mat = Material(sld=custom_sld)
        assert mat.sld.min == -0.5
        assert mat.sld.max == 7.0
