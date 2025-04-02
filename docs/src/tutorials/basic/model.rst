Creating a model
<<<<<<< HEAD
================
=======
=====
>>>>>>> b1ca245ea99629ea738a82d3a22b7481aa5fc67f

The main component of an experiment in :py:mod:`easyreflectometry` is the :py:class:`Model`. 
This is a description of the :py:class:`Sample` and the environment in which the experiment is performed. 
The :py:class:`Model` is used to calculate the reflectivity of the :py:class:`Sample` at a given set of angles (Q-points).
The :py:func:`resolution_functions` are used to quantify the experimental uncertainties in wavelength and angle, allowing the :py:class:`Model` to accurately describe the data.

:py:class:`Model`
-----------------

A :py:class:`Model` instance contains a :py:class:`Sample` and variables describing experimental settings.
To be able to compute reflectivities it is also necessary to have a :py:class:`Calculator` (interface).

.. code-block:: python 

      from easyreflectometry.calculators import CalculatorFactory
      from easyreflectometry.model import Model
      from easyreflectometry.sample import Sample

      default_sample = Sample()
      model = Model(
         sample=default_sample,
         scale=1.0,
         background=1e-6
      )

      interface = CalculatorFactory()
      model.interface = interface

This will create a :py:class:`Model` instance with the :py:attr:`default_sample` and the environment variables :py:attr:`scale` factor set to 1.0 and a :py:attr:`background` of 1e-6.
Following the :py:attr:`interface` is set to the default calculator that is :py:class:`Refnx`.


:py:mod:`resolution_functions`
------------------------------
A resolution function enables the :py:mod:`easyreflectometry` model to incorporate the experimental uncertainties in wavelength and incident angle into the model.
In its essence the resolution function controls the smearing to apply when determing the reflectivtiy at a given Q-point.
For a given Q-point the smearing to apply is given as a weighted average of the neighboring Q-point, which weigths are by a normal distribution.
This normal distribution is then defined by a Q-point dependent Full Width at the Half Maximum (FWHM) that is given by the resolution function.

:py:class:`PercentageFwhm`
Often we rely on a resolution function that has a simple functional dependecy of the Q-point.
By this is understood that the applied smearing in an Q-point has a FWHM that is simply a percentage of the value of the Q-point.

.. code-block:: python 

      from easyreflectometry.model import Model
      from easyreflectometry.model import PercentageFwhm

      resolution_function = PercentageFwhm(1.1)

      m = Model(
         resolution_function=resolution_function
      )

This will create a :py:class:`Model` instance where the resolution function is defined as 1.1% of the Q-point value, which again is the FWHM for the smearing.


:py:func:`LinearSpline`
Alternatively the FWHM value might be determined and declared directly for each measured Q-point.
When this is the case the provided Q-points and the corresponding FWHM values can be used to declare a linear spline function
and thereby enable a determination of the reflectivity at an arbitrary point within the provided range of discrete Q-points.

.. code-block:: python 

      from easyreflectometry.model import Model
      from easyreflectometry.model import LinearSpline

      m = Model()

      resolution_function = LinearSpline(
         q_data_points=[0.01, 0.2, 0.31],
         fwhm_values=[0.001, 0.043, 0.026]
      )

      m.resolution_function = resolution_function

This will create a :py:class:`Model` instance where the resolution function defining the FWHM is determined from a linear interpolation.
In the present case the provided data Q-points are (`[0.01, 0.2, 0.31]`) and the corresponding FWHM function values are (`[0.001, 0.043, 0.026]`).
