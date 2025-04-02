Defining Layers
===============

Similar to a range of different `materials`_, there are a few different ways that a layer can be defined in :py:mod:`easyreflectometry`.

:py:class:`Layer`
-----------------

The :py:class:`Layer` is the simplest possible type of layer, taking a :py:class:`Material` and two floats associated with the thickness and upper (that is closer to the source of the incident radiation) roughness. 
So we construct a :py:class:`Layer` as follows for a 100 Å thick layer of boron with a roughness of 10 Å. 

.. code-block:: python

    from easyreflectometry.sample import Material
    from easyreflectometry.sample import Layer

    boron = Material(
        sld=6.908,
        isld=-0.278,
        name='Boron'
    )
    boron_layer = Layer(
        material=boron,
        thickness=100, 
        roughness=10,
        name='Boron Layer'
    )

This type of layer is used extensively in the `tutorials`_

To create a semi-infinite layer one needs to set the thickness to 0 and the roughness to 0.

.. code-block:: python

    from easyreflectometry.sample import Material
    from easyreflectometry.sample import Layer

    si = Material(
        sld=2.07,
        isld=0,
        name='Si'
    )
    semi_infinite_layer = Layer(
        material=si,
        thickness=0,
        roughness=0,
        name='Si layer'
    )

:py:class:`LayerAreaPerMolecule`
--------------------------------

The :py:class:`LayerAreaPerMolecule` layer type is the fundation of the :py:class:`SurfactantLayer` assemblies type (further information on this can be found in the `assemblies library`_).
The purpose of the :py:class:`LayerAreaPerMolecule` is to allow a layer to be defined in terms of the chemical formula of the material and the area per molecule of the layer. 
The area per molecule is a common description of surface density in the surfactant monolayer and bilayer community. 

We can construct a 10 Å thick :py:class:`LayerAreaPerMolecule` of phosphatidylcholine, with an area per molecule of 48 Å squared and a roughness of 3 Å that has 20 % solvent surface coverage with D2O using the following.

.. code-block:: python

    from easyreflectometry.sample import Material
    from easyreflectometry.sample import LayerAreaPerMolecule

    d2o = Material(
        sld=6.36,
        isld=0,
        name='D2O'
    )
    molecular_formula = 'C10H18NO8P'
    pc = LayerAreaPerMolecule(
        molecular_formula=molecular_formula, 
        thickness=10, 
        solvent=d2o, 
        solvent_fraction=.2,
        area_per_molecule=48, 
        roughness=3,
        name='PC Layer'
    )

It is expected that the typical user will not interface directly with the :py:class:`LayerAreaPerMolecule` assembly type, but instead the :py:class:`SurfactantLayer` `assemblies library`_ will be used instead. 

.. _`materials`: ./material_library.html
.. _`tutorials`: ../tutorials/tutorials.html
.. _`assemblies library`: ./assemblies_library.html