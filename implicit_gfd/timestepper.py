from firedrake.petsc import PETSc
from state.py import get_state
from irksome import GalerkinTimeStepper, MeshConstant

opts = PETSc.Options()
print = PETSc.Sys.Print

model_type = opts.getString('model_type', 'swe')
model_variant = opts.getString('model_variant', 'G')
model = get_model(model_type, model_variant)

MC = MeshConstant(model.mesh)
dT = MC.Constant(dt)
t = MC.Constant(0.)

timestepper = opts.getString('time_method', 'galerkin')
if timestepper == 'galerkin':
    basis_type = opts.getString('basis_type', None)
    quadrature_degree = opts.getString('quadrature_degree', None)
    quadrature_scheme = opts.getString('quadrature_scheme',
                                       "default")
    time_variant = opts.getString('time_variant', 'cPG')
    time_order = opts.getInt('time_order', 1)
    if time_variant == 'cPG':
        method = ContinuousPetrovGalerkinScheme(
            time_order=time_order,
            basis_type=basis_type,
            quadrature_degree=quadrature_degre,
            quadrature_scheme=quadrature_scheme)
    else:
        raise NotImplementedError, 'time_variant', time_variant
else:
    raise NotImplementedError, 'timestepper', timestepper

stepper = TimeStepper(model.eqn, method, t, dT, model.U0,
                      options_prefix="rsw")
