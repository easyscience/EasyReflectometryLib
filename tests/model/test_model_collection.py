from easyscience import global_object

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
        assert sorted(p.as_data_dict(skip=['resolution_function', 'interface'])) == sorted(
            q.as_data_dict(skip=['resolution_function', 'interface'])
        )
        assert p[0]._resolution_function.smearing(5.5) == q[0]._resolution_function.smearing(5.5)
