Creating multilayers and surfactant layers
===========================================

:py:mod:`easyreflectometry` is designed to be used with a broad range of different assemblies.  
Assemblies are collective layers behaving as a single object, for example, a multilayer or a surfactant layer.
These assemblies offer flexibility for the user and enable more powerful analysis by making chemical and physical constraints available with limited code. 
In this page, we will document the assemblies that are available with simple examples of the constructors that exist.
Full API documentation is also available for the :py:mod:`easyreflectometry.sample.assemblies` module.

:py:class:`Multilayer`
----------------------

This assembly should be used for a series of layers that should be thought of as a single object. 
For example, in the `simple fitting tutorial`_ this assembly type is used to combine the silicon and silicon dioxide layer that as formed into a single object. 
All of the separate layers in these objects will be fitted individually, i.e. there is no constraints present, however, there is some cognative benefit to grouping layers together. 

To create a :py:class:`Multilayer` object, we use the following construction.

.. code-block:: python 

    from easyreflectometry.sample import Layer
    from easyreflectometry.sample import Material
    from easyreflectometry.sample import Multilayer

    si = Material(
        sld=2.07,
        isld=0,
        name='Si'
    )
    sio2 = Material(
        sld=3.47,
        isld=0,
        name='SiO2'
    )
    si_layer = Layer(
        material=si,
        thickness=0,
        roughness=0,
        name='Si layer'
    )
    sio2_layer = Layer(
        material=sio2,
        thickness=30,
        roughness=3,
        name='SiO2 layer'
    )

    subphase = Multilayer(
        layers=[si_layer, sio2_layer], 
        name='Si/SiO2 subphase'
    )

This will create a :py:class:`Multilayer` object named :code:`subphase` which we can use in some :py:class:`Structure` for our analysis. 

:py:class:`RepeatingMultilayer`
-------------------------------

The :py:class:`RepeatingMultilayer` assembly type is an extension of the :py:class:`Multilayer` for the analysis of systems with a multilayer that has some number of repeats. 
This assembly type imposes some constraints, specifically that all of the repeats have the exact same structure (i.e. thicknesses, roughnesses, and scattering length densities), 
which brings with it some computational saving as the reflectometry coefficients only needs to be calculated once for this structure and propagated for the correct number of repeats. 
There is a `tutorial`_ that discusses the utilisation of this assembly type for a nickel-titanium multilayer system. 

The creation of a :py:class:`RepeatingMultilayer` object is very similar to that for the :py:class:`Multilayer`, with the addition of a number of repetitions. 

.. code-block:: python 

    from easyreflectometry.sample import Layer
    from easyreflectometry.sample import Material
    from easyreflectometry.sample import RepeatingMultilayer

    ti = Material(
        sld=-1.9493,
        isld=0,
        name='Ti'
    )
    ni = Material(
        sld=9.4245,
        isld=0,
        name='Ni'
    )
    ti_layer = Layer(
        material=ti,
        thickness=40,
        roughness=0,
        name='Ti Layer'
    )
    ni_layer = Layer(
        material=ni,
        thickness=70,
        roughness=0,
        name='Ni Layer'
    )
    ni_ti = RepeatingMultilayer(
        layers=[ti_layer, ni_layer], 
        repetitions=10, 
        name='Ni/Ti Multilayer'
    )

The number of repeats is a parameter that can be varied in the optimisation process, however given this is a value that depends on the synthesis of the sample this is unlikely to be necessary.

:py:class:`SurfactantLayer`
---------------------------

The :py:class:`SurfactantLayer` assembly type allows for the creating of a model to describe a monolayer of surfactant at some interface. 
Using this assembly, we can define our surfactant in terms of the chemistry of the head and tail groups and be confident that the constraints are present to ensure the number density if kept constant. 
The `surfactant monolayer tutorial`_ looks in detail at the definition of the scattering length density in the :py:class:`SurfactantLayer`. 
However, it is founded on the chemical formula for the head and tail group and the area per molecule that these groups occupy. 

The creation of a :py:class:`SurfactantLayer` object is shown below. 

.. code-block:: python
   
    from easyreflectometry.sample import LayerAreaPerMolecule
    from easyreflectometry.sample import Material
    from easyreflectometry.sample import SurfactantLayer

    area_per_molecule = 48
    roughness = 3.3
    subphase = Material(
        sld=6.36,
        isld=0.0,
        name='D2O'
    )
    superphase = Material(
        sld=0.0,
        isld=0.0,
        name='Air'
    )
    tail_layer = LayerAreaPerMolecule(
        molecular_formula='C30D64',
        thickness=16.0,
        solvent=superphase,
        solvent_fraction=0.0, 
        area_per_molecule=area_per_molecule,
        roughness=roughness
    )
    head_layer = LayerAreaPerMolecule(
        molecular_formula='C10H18NO8P',
        thickness=10.0,
        solvent=subphase,
        solvent_fraction=0.2, 
        area_per_molecule=area_per_molecule,
        roughness=roughness
    )
    dspc = SurfactantLayer(
        tail_layer=tail_layer,
        head_layer=head_layer
    )
    
On creation, the area per molecule and roughness above both the head and tail layers can be constrained to be the same. 
These constraints can be addded by setting :code:`dppc.constrain_area_per_molecule = True` or :code:`dppc.conformal_roughness = True`. 
Furthermore, as shown in the `surfactant monolayer tutorial`_ the conformal roughness can be defined by that of the subphase. 

The use of the :py:class:`SurfactantLayer` in multiple contrast data analysis is shown in a `multiple contrast tutorial`_. 

:py:class:`Bilayer`
-------------------

The :py:class:`Bilayer` assembly type represents a phospholipid bilayer at an interface.
It consists of two surfactant layers where one is inverted, creating the structure:

.. code-block:: text

    Head₁ - Tail₁ - Tail₂ - Head₂

This assembly is particularly useful for studying supported lipid bilayers and membrane systems.
The bilayer comes pre-populated with physically meaningful constraints:

- Both tail layers share the same structural parameters (thickness, area per molecule)
- Head layers share thickness and area per molecule (different hydration/solvent fraction allowed)
- A single roughness parameter applies to all interfaces (conformal roughness)

These default constraints can be enabled or disabled as needed for specific analyses.

The creation of a :py:class:`Bilayer` object is shown below.

.. code-block:: python

    from easyreflectometry.sample import Bilayer
    from easyreflectometry.sample import LayerAreaPerMolecule
    from easyreflectometry.sample import Material

    # Create materials for solvents
    d2o = Material(sld=6.36, isld=0.0, name='D2O')
    air = Material(sld=0.0, isld=0.0, name='Air')
    
    # Create head layer (used for front, back head will be auto-created with constraints)
    head = LayerAreaPerMolecule(
        molecular_formula='C10H18NO8P',
        thickness=10.0,
        solvent=d2o,
        solvent_fraction=0.3,
        area_per_molecule=48.2,
        roughness=3.0,
        name='DPPC Head'
    )
    
    # Create tail layer (both tail positions will share these parameters)
    tail = LayerAreaPerMolecule(
        molecular_formula='C32D64',
        thickness=16.0,
        solvent=air,
        solvent_fraction=0.0,
        area_per_molecule=48.2,
        roughness=3.0,
        name='DPPC Tail'
    )
    
    # Create bilayer with default constraints
    bilayer = Bilayer(
        front_head_layer=head,
        tail_layer=tail,
        constrain_heads=True,
        conformal_roughness=True,
        name='DPPC Bilayer'
    )

The head layers can have different solvent fractions (hydration) even when constrained, 
enabling the modeling of asymmetric bilayers at interfaces where the two sides of the 
bilayer may have different solvent exposure.

The constraints can be controlled at runtime:

.. code-block:: python

    # Disable head constraints to allow different head layer structures
    bilayer.constrain_heads = False
    
    # Disable conformal roughness to allow different roughness values
    bilayer.conformal_roughness = False

Individual layers can be accessed via properties:

.. code-block:: python

    # Access the four layers
    bilayer.front_head_layer  # First head layer
    bilayer.front_tail_layer  # First tail layer  
    bilayer.back_tail_layer   # Second tail layer (constrained to front tail)
    bilayer.back_head_layer   # Second head layer

For more detailed examples including simulation and parameter access, see the `bilayer tutorial`_.


.. _`simple fitting tutorial`: ../tutorials/simple_fitting.html
.. _`tutorial`: ../tutorials/repeating.html
.. _`surfactant monolayer tutorial`: ../tutorials/monolayer.html
.. _`multiple contrast tutorial`: ../tutorials/multi_contrast.html
.. _`bilayer tutorial`: ../tutorials/simulation/bilayer.html