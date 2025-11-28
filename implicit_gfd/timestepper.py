from firedrake.petsc import PETSc
from models import get_model
from steppers import get_stepper
from math import fabs
from firedrake import ProgressBar

opts = PETSc.Options()
print = PETSc.Sys.Print

model_opts = PETSc.Options('model_')
model = get_model(model_opts)

dt = opts.getScalar('model_dt', 1.0)
tmax = opts.getScalar('model_tmax', 1.0)
nsteps = int(tmax/dt)
assert fabs(nsteps*dt - tmax) < 1.0e-6*dt, 'tmax is not a multiple of dt'

stepper_opts = PETSc.Options('stepper_')
stepper, dt, t = get_stepper(model, stepper_opts)

for step in ProgressBar('Timestep').iter(range(nsteps)):
    stepper.advance()
    t.assign(float(t) + float(dt))
