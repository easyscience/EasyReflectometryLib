Defining materials
==================

In order to support a wide range of applications (and to build complex `assemblies`_) there are a few different types of material that can be utilised in :py:mod:`easyreflectometry`. 
These can include constraints or enable the user to define the material based on chemical or physical properties. 
Full API documentation for the :py:mod:`easyreflectometry.sample.elements.material` module is also available, but here we will give some simple uses for them. 

:py:class:`Material`
--------------------

The simplest type of material that is available is the :py:class:`Material`.
This allows the user to define a single type of material, with a real and imaginary component to the scattering length density. 
The construction of a :py:class:`Material` is achieved as shown below. 

.. code-block:: python 

    from easyreflectometry.sample import Material

    boron = Material(
        sld=6.908,
        isld=-0.278,
        name='Boron'
    )

The above object will have the properties of :py:attr:`sld` and :py:attr:`isld`, which will have values of :code:`6.908 1/angstrom^2` and :code:`-0.278 1/angstrom^2` respectively. 
As is shown in the `tutorials`_, a material can be used to construct a :py:class:`Layer` from which `slab models`_ are created.

:py:class:`MaterialDensity`
---------------------------

In addition to defining a material by its scattering length density, it may be useful to define a material by the mass density and chemical formula. 
This is possible with the :py:class:`MaterialDensity` material type, which uses the scattering length and atomic mass from the chemical formula and the density to determine the scattering length density. 
It is then possible to vary the density, which defines the scattering length density in turn. 
The :py:class:`MaterialDensity` material can be create as follows. 

.. code-block:: python 

    from easyreflectometry.sample import MaterialDensity 

    chemical_structure = 'SiO2'
    si = MaterialDensity(
        chemical_structure=chemical_structure,
        density=2.65,
        name='SiO2 Material'
    )

The density should be in units of grams per cubic centimeter and the scattering length is calculated from :code:`'SiO2'`. 

:py:class:`MaterialSolvated`
----------------------------

Sometimes it is desirable to have a layer that consists of a material and a solvent in some ratio.
An example of this is shown in the `solvation tutorial`_, where a polymer film solvated with D2O is modelled. 
To produce a material that is described by such a mixture, there is :py:class:`MaterialSolvated`. 
This is constructed from two constituent :py:class:`Materials` and the fractional amount of the material in the solvent. 
So to produce a :py:class:`MaterialSolvated` that is 20 % D2O in a polymer, the following is used. 

.. code-block:: python

    from easyreflectometry.sample import Material 
    from easyreflectometry.sample import MaterialSolvated

    polymer = Material(
        sld=2.,
        isld=0.,
        name='Polymer'
    )
    d2o = Material(
        sld=6.36,
        isld=0, 
        name='D2O'
    )

    solvated_polymer = MaterialSolvated(
        material=polymer, 
        solvent=d2o, 
        solvent_fraction=0.2, 
        name='Solvated Polymer'
    )

For the :py:attr:`solvated_polymer` object, the :py:attr:`sld` will be :code:`2.872 1/angstrom^2` (the weighted average of the two scattering length densities). 
The :py:class:`MaterialSolvated` includes a constraint such that if the value of either constituent scattering length densities (both real and imaginary components) or the fraction changes, then the resulting material :py:attr:`sld` and :py:attr:`isld` will change appropriately. 

.. _`assemblies`: ./assemblies_library.html
.. _`tutorials`: ../tutorials/tutorials.html
.. _`slab models`: https://www.reflectometry.org/isis_school/3_reflectometry_slab_models/the_slab_model.html
.. _`solvation tutorial`: ../tutorials/solvation.html