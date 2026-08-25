Constraints
===========

EasyReflectometry offers three kinds of constraints between model parameters.

Equality constraints (dependencies)
-----------------------------------

A parameter can be tied to an arbitrary expression of other parameters. It
then leaves the set of free fit parameters and follows the expression::

    from easyreflectometry import constrain, constrain_equal, unconstrain

    constrain_equal(layer_b.roughness, to=layer_a.roughness)
    constrain(layer_b.thickness, '2 * t', t=layer_a.thickness)
    unconstrain(layer_b.thickness)

Constraints survive ``Project`` save/load: the dependency expression and the
ids of the parameters it refers to are stored with every parameter and the
graph is rebuilt when the project is loaded.

Derived (read-only) parameters
------------------------------

A *derived parameter* is a dependent parameter that belongs to no layer: a
live calculation that can be shown, referenced from other constraints and used
in inequalities — the counterpart of a bumps ``Calculation`` slot::

    from easyreflectometry import derived_parameter, constrain_to_sum

    total = derived_parameter('total', 'a + b', a=layer_a.thickness, b=layer_b.thickness)
    # keep the film thickness fixed at 120 Å while the split is fitted
    constrain_to_sum(layer_b.thickness, [layer_a.thickness, layer_b.thickness], total=120.0)

Every :class:`~easyreflectometry.model.Model` exposes
:attr:`~easyreflectometry.model.Model.total_thickness`: the summed thickness of
the layers between the superphase and the subphase, re-derived whenever the
layer structure changes.

Inequality constraints (fit penalties)
--------------------------------------

Cross-parameter inequalities such as ``t_head < t_tail`` or
``t1 + t2 <= total`` are not dependencies: no parameter is removed from the
fit. They are declared as :class:`~easyreflectometry.inequality_constraints.InequalitySpec`
objects on the project and enforced by the **BUMPS** engines (``Bumps*``
minimizers and the DREAM sampler) as penalties on the fit problem. LMFit and
DFO-LS cannot enforce them; ``fit`` raises ``ValueError`` in that case::

    from easyreflectometry import InequalitySpec
    from easyscience.fitting import AvailableMinimizers

    project.minimizer = AvailableMinimizers.Bumps
    t_a = project.parameter_path(layer_a.thickness)       # 'models/0/sample/1/layers/0/thickness'
    t_b = project.parameter_path(layer_b.thickness)
    project.add_inequality_constraint(InequalitySpec('a', '<', 'b', {'a': t_a}, {'b': t_b}, name='order'))
    project.add_inequality_constraint(InequalitySpec('a + b', '<', '90', {'a': t_a, 'b': t_b}, {}))

    project.violated_inequality_constraints()             # check the start point first
    project.fitter.fit_single_data_set_1d(dataset)         # penalties applied automatically

Parameters are referenced by *structural path* (see
:meth:`~easyreflectometry.Project.parameter_path`) so the constraints are
saved with the project. While a constraint is violated BUMPS skips the model
evaluation and adds a penalty growing with the violation, steering the
optimizer back into the feasible region; the ``Bumps_lm`` method spreads the
penalty over the residuals instead and enforces inequalities more weakly.

API reference
-------------

.. automodule:: easyreflectometry.constraints
   :members:

.. automodule:: easyreflectometry.inequality_constraints
   :members:
