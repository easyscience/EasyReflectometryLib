<<<<<<< HEAD
==========
How to use
==========
=======
================
**How to use**
================
>>>>>>> b1ca245ea99629ea738a82d3a22b7481aa5fc67f

Dictionary
==========
The following serves to clarify what we mean by the terms we use in this project.

Sample
------
A sample is an ideal representation of a the full physical setup.
This includes the layer(s) under investigation, the surrounding superphase, and the subphase.

Calculator
----------
A calculator is the physics engine which calculates the reflectivity curve from our inputted sample parameters.
We rely on third party software to provide the necessary calculators.
Different calculators might have different capabilities and limitations. 

Model
-----
A model combines a sample and calculator.
The model is also responsible for including instrumental effects such as background, scale, and resolution.


Calculators & Optimisation
==========================

:py:mod:`easyreflectometry` is built on the :py:mod:`easyscience` framework which facilities the use of a range of different reflectometry calculation engines and optimiser solutions. 
Currently, :py:mod:`easyreflectometry` can offer two different calculation engines, namely:

* `refnx`_
* `Refl1D`_

And we are working to add more, in particular `bornagain`_ and `GenX`_. 

.. _`refnx`: https://refnx.readthedocs.io/
.. _`Refl1D`: https://refl1d.readthedocs.io/en/latest/
.. _`BornAgain`: https://www.bornagainproject.org
.. _`GenX`: https://aglavic.github.io/genx/doc/

.. toctree:: 
   :maxdepth: 2

<<<<<<< HEAD
   basic/basic
   simulation/simulation
   fitting/fitting
   advancedfitting/advancedfitting
   extra/extra
=======
   tutorials/basic
   tutorials/simulation
   tutorials/fitting
   tutorials/advancedfitting
   tutorials/extra
>>>>>>> b1ca245ea99629ea738a82d3a22b7481aa5fc67f
