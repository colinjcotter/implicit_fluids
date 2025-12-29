from firedrake.petsc import PETSc
from models import get_model
from steppers import get_stepper
from math import fabs
from firedrake import ProgressBar, VTKFile, CheckpointFile

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

filename = opts.hasName("filename")
if filename:
    filename = opts.getString("filename")
    vtkfreq = opts.getInt("vtkfreq", -999)
    chkptfreq = opts.getInt("chkptfreq", -999)
else:
    vtkfreq = -999
    chkptfreq= -999

vtk_count = 0
chkpt_count = 0
nchk = 0

if filename and vtkfreq >= 0:
    vtkfile = VTKFile(filename+".pvd")
    fields = model.output()
    vtkfile.write(*fields)

if filename and chkpt_count >=0:
    with CheckpointFile(filename+".h5", 'w') as cfile:
        fields = model.output()
        for field in fields:
            cfile.save_function(field, idx=0)

for step in ProgressBar('Timestep').iter(range(nsteps)):
    stepper.advance()
    t.assign(float(t) + float(dt))

    vtk_count += 1
    chkpt_count += 1

    if vtk_count == vtkfreq:
        fields = model.output()
        vtkfile.write(*fields)
        vtk_count = 0

    if chkpt_count == chkptfreq:
        nchk += 1
        with CheckpointFile(filename+".h5", 'w') as cfile:
            fields = model.output()
            for field in fields:
                cfile.save_function(field, idx=nchk)
        chkpt_count = 0
