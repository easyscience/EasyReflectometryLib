===============
Getting started
===============

To use :py:mod:`easyreflectometry` in a project::

    import easyreflectometry
    from easyreflectometry.sample import Material, Layer
    from easyreflectometry.model import Model
    from easyreflectometry.fitting import MultiFitter
    from easyreflectometry.plot import plot

    # Define your Material
    material = Material(...)

    # Create a Layer
    layer = Layer(material=material, ...)

    # Make a Sample out of the Layer
    sample = Sample(layer, ...)

    # Define a Model of the experiment
    model = Model(
    sample=sample,
    scale=1,
    background=1e-6,
    ...
    )

    # Set parameter bounds for fit
    ...

    # Perform the fit and plot
    fitter = MultiFitter(model)
    analysed = fitter.fit(data)

    plot(analysed)


Details of specific usage of :py:mod:`easyreflectometry` can be found in the `tutorials`_.

.. _`tutorials`: ./tutorials/tutorials.rst