from firedrake.petsc import PETSc
from irksome import ContinuousPetrovGalerkinScheme
from irksome import MeshConstant, TimeStepper

def get_stepper(model, opts):
    timestepper = opts.getString('time_method', 'galerkin')
    if timestepper == 'galerkin':
        if opts.hasName('basis_type'):
            basis_type = opts.getString('basis_type')
        else:
            basis_type = None
        if opts.hasName('quadrature_degree'):
            quadrature_degree = opts.getInt('quadrature_degree')
        else:
            quadrature_degree = None
        quadrature_scheme = opts.getString('quadrature_scheme',
                                           'default')
        time_variant = opts.getString('time_variant', 'cPG')
        time_order = opts.getInt('time_order', 1)
        if time_variant == 'cPG':
            method = ContinuousPetrovGalerkinScheme(
                order=time_order,
                basis_type=basis_type,
                quadrature_degree=quadrature_degree,
                quadrature_scheme=quadrature_scheme)
        else:
            raise NotImplementedError('time_variant '+time_variant)
    else:
        raise NotImplementedError('timestepper '+timestepper)

    dt = opts.getScalar('dt', 100.0)
    MC = MeshConstant(model.mesh)
    dT = MC.Constant(dt)
    t = MC.Constant(0.)
    U0 = model.U0()
    eqn = model.eqn()
    return TimeStepper(eqn, method, t, dT, U0,
                       options_prefix="stepper"), dt, t
