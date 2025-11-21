from firedrake.petsc import PETSc
from models import get_model
from irksome import GalerkinTimeStepper
from steppers import get_stepper

opts = PETSc.Options()
print = PETSc.Sys.Print

model_opts = PETSc.Options("model_")
model = get_model(model_opts)

MC = MeshConstant(model.mesh)
dt = opts.getScalar('model_dt', 1.0)
dT = MC.Constant(dt)
t = MC.Constant(0.)

stepper_opts = PETSc.Options("stepper_")
stepper = get_stepper(stepper_opts)
