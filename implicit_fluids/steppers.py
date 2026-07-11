from irksome import (
    ContinuousPetrovGalerkinScheme,
    GalerkinCollocationScheme,
    )
from irksome import MeshConstant, TimeStepper


def get_stepper(model, opts):
    timestepper = opts.getString('time_method', 'galerkin')
    if timestepper == 'galerkin':
        if opts.hasName('time_basis_type'):
            basis_type = opts.getString('time_basis_type')
        else:
            basis_type = None
        if opts.hasName('time_quadrature_degree'):
            quadrature_degree = opts.getInt('time_quadrature_degree')
        else:
            quadrature_degree = None
        if opts.hasName('time_quadrature_scheme'):
            quadrature_scheme = opts.getString('time_quadrature_scheme')
        else:
            quadrature_scheme = None
        time_variant = opts.getString('time_variant', 'cPG')
        time_order = opts.getInt('time_order', 1)
        if time_variant == 'cPG':
            method = ContinuousPetrovGalerkinScheme(
                order=time_order,
                basis_type=basis_type,
                quadrature_degree=quadrature_degree,
                quadrature_scheme=quadrature_scheme)
        elif time_variant == 'collocation':
            print(time_order, quadrature_degree, quadrature_scheme)
            method = GalerkinCollocationScheme(
                order=time_order,
                stage_type="deriv",
                quadrature_degree=quadrature_degree,
                quadrature_scheme=quadrature_scheme,
                max_quadrature_degree=quadrature_degree)
        else:
            raise NotImplementedError('time_variant '+time_variant)
    else:
        raise NotImplementedError('timestepper '+timestepper)

    MC = MeshConstant(model.mesh)
    dT = MC.Constant(1.)
    t = MC.Constant(0.)
    U0 = model.U0()
    eqn = model.eqn()
    return TimeStepper(eqn, method, t, dT, U0,
                       options_prefix="stepper"), dT, t
