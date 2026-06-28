from firedrake.petsc import PETSc
from implicit_fluids.models import get_model
from implicit_fluids.steppers import get_stepper
from math import fabs
from firedrake import ProgressBar, VTKFile, CheckpointFile, assemble
import pandas as pd

print = PETSc.Sys.Print


def run(model):
    opts = PETSc.Options()
    dt = opts.getScalar('model_dt', 1.0)
    tmax = opts.getScalar('model_tmax', 1.0)
    nsteps = int(tmax/dt)
    assert fabs(nsteps*dt - tmax) < 1.0e-6*dt, 'tmax is not a multiple of dt'

    stepper_opts = PETSc.Options('stepper_')
    stepper, dT, t = get_stepper(model, stepper_opts)
    dT.assign(dt)

    filename = opts.hasName("filename")
    if filename:
        filename = opts.getString("filename")
        vtkfreq = opts.getInt("vtkfreq", -999)
        chkptfreq = opts.getInt("chkptfreq", nsteps)
    else:
        vtkfreq = -999
        chkptfreq = -999

    vtk_count = 0
    chkpt_count = 0
    nchk = 0

    if filename and vtkfreq >= 0:
        vtkfile = VTKFile(filename+".pvd")
        fields = model.output()
        vtkfile.write(*fields)

    if filename and chkpt_count >= 0:
        with CheckpointFile(filename+".h5", 'w') as cfile:
            fields = model.output()
            for field in fields:
                cfile.save_function(field, idx=0)

    # diagnostics setup
    diagnostics0 = model.diagnostics()
    diagnostics = {}
    for key, value in diagnostics0.items():
        diagnostics[key] = [value]

    for step in ProgressBar('Timestep').iter(range(nsteps)):
        # set absolute tolerance automatically for ew
        if opts.hasName("stepper_snes_ksp_ew"):
            F = stepper.solver._problem.F
            with assemble(F).dat.vec_ro as vec:
                res0 = vec.norm()
            snes_rtol = stepper.solver.snes.rtol
            stepper.solver.snes.ksp.atol = 0.1*snes_rtol*res0

        stepper.advance()
        t.assign(float(t) + float(dt))

        # diagnostics
        diagnostics0 = model.diagnostics()
        for key, value in diagnostics0.items():
            diagnostics[key].append(value)

        # VTK
        vtk_count += 1
        if vtk_count == vtkfreq:
            fields = model.output()
            vtkfile.write(*fields)
            vtk_count = 0

        # checkpointing
        chkpt_count += 1
        if chkpt_count == chkptfreq:
            nchk += 1
            with CheckpointFile(filename+".h5", 'a') as cfile:
                fields = model.output()
                for field in fields:
                    cfile.save_function(field, idx=nchk)
            chkpt_count = 0

    # save diagnostics
    if filename:
        df = pd.DataFrame(diagnostics)
        df.to_csv(filename+'.csv')
    return diagnostics


if __name__ == "__main__":
    run()
