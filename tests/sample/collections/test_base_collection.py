# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

from easyreflectometry.sample.collections.base_collection import BaseCollection
from easyreflectometry.sample.elements.layers.layer import Layer


class TestBaseCollection:
    def test_constructor(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        mock_interface = MagicMock()

        # Then
        p = BaseCollection('name', mock_interface, elem_1, elem_2)

        # Expect
        p._interface = mock_interface
        len(p) == 2

    def test_names(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        mock_interface = MagicMock()

        # Then
        p = BaseCollection('name', mock_interface, elem_1, elem_2)

        # Expect
        assert p.names == ['layer_1', 'layer_2']

    def test_dict_repr(self):
        # When
        elem = Layer(name='layer')
        mock_interface = MagicMock()

        # Then
        p = BaseCollection('name', mock_interface, elem)

        # Expect
        assert p._dict_repr == {
            'name': [
                {
                    'layer': {
                        'material': {'EasyMaterial': {'isld': '0.000e-6 1/Å^2', 'sld': '4.186e-6 1/Å^2'}},
                        'roughness': '3.300 Å',
                        'thickness': '10.000 Å',
                    }
                }
            ]
        }

    def test_as_dict(self):
        # When
        elem = Layer(name='layer')
        mock_interface = MagicMock()

        # Then
        p = BaseCollection('name', mock_interface, elem)

        # Expect
        assert p.as_dict()['name'] == 'name'
        assert len(p.as_dict()['data']) == 1
        assert p.as_dict()['data'][0]['name'] == 'layer'

    def test_move_up(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))

        # Then
        p.move_up(3)

        # Expect
        assert p[0].name == 'layer_1'
        assert p[1].name == 'layer_2'
        assert p[2].name == 'layer_4'
        assert p[3].name == 'layer_3'

    def test_move_up_to_top_and_further(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))

        # Then
        p.move_up(3)
        p.move_up(2)
        p.move_up(1)
        p.move_up(0)

        # Then
        assert p[0].name == 'layer_4'
        assert p[1].name == 'layer_1'
        assert p[2].name == 'layer_2'
        assert p[3].name == 'layer_3'

    def test_move_down(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))

        # Then
        p.move_down(2)

        # Expect
        assert p[0].name == 'layer_1'
        assert p[1].name == 'layer_2'
        assert p[2].name == 'layer_4'
        assert p[3].name == 'layer_3'

    def test_move_down_to_bottom_and_further(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))
        p.append(Layer(name='layer_5'))

        # Then
        p.move_down(3)
        p.move_down(4)

        # Then
        assert p[0].name == 'layer_1'
        assert p[1].name == 'layer_2'
        assert p[2].name == 'layer_3'
        assert p[3].name == 'layer_5'
        assert p[4].name == 'layer_4'

    def test_remove(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))

        # Then
        p.remove(1)

        # Then
        assert len(p) == 3
        assert p[0].name == 'layer_1'
        assert p[1].name == 'layer_3'
        assert p[2].name == 'layer_4'
