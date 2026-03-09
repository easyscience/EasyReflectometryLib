from easyscience import global_object

from easyreflectometry.model.model import COLORS
from easyreflectometry.model.model import Model
from easyreflectometry.model.model_collection import ModelCollection


class TestModelCollection:
    def test_default(self):
        # When Then
        collection = ModelCollection()

        # Expect
        assert collection.name == 'Models'
        assert collection.interface is None
        assert len(collection) == 1
        assert collection[0].name == 'Model'

    def test_dont_populate(self):
        p = ModelCollection(populate_if_none=False)
        assert p.name == 'Models'
        assert p.interface is None
        assert len(p) == 0

    def test_from_pars(self):
        # When
        model_1 = Model(name='Model1')
        model_2 = Model(name='Model2')
        model_3 = Model(name='Model3')

        # Then
        collection = ModelCollection(model_1, model_2, model_3)

        # Expect
        assert collection.name == 'Models'
        assert collection.interface is None
        assert len(collection) == 3
        assert collection[0].name == 'Model1'
        assert collection[1].name == 'Model2'
        assert collection[2].name == 'Model3'

    def test_add_model(self):
        # When
        model_1 = Model(name='Model1')
        model_2 = Model(name='Model2')

        # Then
        collection = ModelCollection(model_1)
        collection.add_model(model_2)

        # Expect
        assert len(collection) == 2
        assert collection[0].name == 'Model1'
        assert collection[1].name == 'Model2'

    def test_add_model_color_cycle(self):
        collection = ModelCollection(populate_if_none=False)

        collection.add_model()
        assert collection[0].color == COLORS[0]

        collection.add_model()
        assert collection[1].color == COLORS[1]

        collection.remove(0)
        collection.add_model()

        assert collection[0].color == COLORS[1]
        assert collection[1].color == COLORS[2]

    def test_add_model_color_wrap(self):
        collection = ModelCollection(populate_if_none=False)

        for _ in range(len(COLORS)):
            collection.add_model()

        collection.add_model()

        assert collection[-1].color == COLORS[0]

    def test_add_model_preserves_explicit_color(self):
        collection = ModelCollection(populate_if_none=False)
        collection.add_model()
        expected_index = collection._next_color_index

        custom_color = '#ABCDEF'
        custom_model = Model(name='Custom', color=custom_color)

        collection.add_model(custom_model)

        assert collection[-1].color == custom_color
        assert collection._next_color_index == (expected_index + 1) % len(COLORS)

    def test_delete_model(self):
        # When
        model_1 = Model(name='Model1')
        model_2 = Model(name='Model2')

        # Then
        collection = ModelCollection(model_1, model_2)
        collection.remove(0)

        # Expect
        assert len(collection) == 1
        assert collection[0].name == 'Model2'

    def test_as_dict(self):
        # When
        model_1 = Model(name='Model1')
        collection = ModelCollection(model_1)

        # Then
        dict_repr = collection.as_dict()

        # Expect
        assert dict_repr['data'][0]['resolution_function'] == {'smearing': 'PercentageFwhm', 'constant': 5.0}

    def test_dict_round_trip(self):
        # When
        model_1 = Model(name='Model1')
        model_2 = Model(name='Model2')
        model_3 = Model(name='Model3')
        p = ModelCollection(model_1, model_2, model_3)
        p_dict = p.as_dict()
        global_object.map._clear()

        # Then
        q = ModelCollection.from_dict(p_dict)

        # Expect
        # We have to skip the resolution_function and interface
        assert sorted(p.as_dict(skip=['resolution_function', 'interface'])) == sorted(
            q.as_dict(skip=['resolution_function', 'interface'])
        )
        assert p[0]._resolution_function.smearing(5.5) == q[0]._resolution_function.smearing(5.5)

    def test_next_color_index_round_trip(self):
        collection = ModelCollection(populate_if_none=False)
        for _ in range(3):
            collection.add_model()

        expected_index = collection._next_color_index
        dict_repr = collection.as_dict()
        global_object.map._clear()

        restored = ModelCollection.from_dict(dict_repr)

        assert restored._next_color_index == expected_index

    def test_legacy_from_dict_sets_color_index(self):
        collection = ModelCollection()
        legacy_dict = collection.as_dict()
        legacy_dict.pop('next_color_index', None)
        global_object.map._clear()

        restored = ModelCollection.from_dict(legacy_dict)
        restored.add_model()

        assert [model.color for model in restored] == [COLORS[0], COLORS[1]]
