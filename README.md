# Implicit Fluids

A collection of models for fluid dynamics (mostly geophysical fluid
dynamics) using Firedrake and Irksome. The system is entirely driven by
PETSc command line options.

## Guide to options

`filename`: Filename prefix to use (suffixes such as .h5 and .pvd will be added)

`vtkfreq`: Frequency to output vtk files at

`chkptfreq`: Frequency to output checkpoint files at

### Specifying the model

These options all have the prefix `model_`

`model_type`: Select the model. `swe` - Shallow water equations (default).

`model_variant`: Select the model variant. `G` - The G formulation of the shallow water equations, where
D = H - div(G) (default).

`model_functionspaces_family`: Select the family for the Hdiv velocity
space, e.g. `BDM` (default), `RT`, etc.

`model_functionspaces_degree`: Select the degree for the family
(integer). 1 is the default.

`model_outputs_vorticity`: Compute the relative vorticity for the output if present.

`model_projection`: For the G variant, use G_t for u*D in the tendency. In a cPG
time-Galerkin formulation, this implements the projection of dH/du into the test
space, as required for discrete time energy conservation. If not present, u is used
in the advection terms for a non energy conserving formulation.

`model_centred`: Use a centred flux for the velocity advection term. If not present
then the upwind flux is used.

`model_dt`: The time stepsize (float)
`model_tmax`: final time

### Specifying the testcases

These options all have the prefix `testcase_`

`testcase`: select the testcase. `w6` - SWE testcase 6 from the
Williamson et al case (Rossby Haurwitz wave).

`testcase_mesh_nrefs`: Specify the number of mesh refinements for a
testcase using the icosahedral sphere mesh.

`testcase_mesh_starpatch`: Set distribution parameters in the mesh as
needed for vertex star ASM patches.

### Specifying the timestepper and its solver options

These options all have the prefix `stepper_`

`stepper_time_method`: `galerkin` - Galerkin in time.
`stepper_time_variant`: `cPG` - Continuous Petrov Galerkin scheme
'stepper_time_order': order of the scheme, integer

Options to be passed to `irksome.scheme`:
`stepper_basis_type`: string
`stepper_quadrature_degree`: integer
`stepper_quadrature_scheme`: string

The PETSc solver for the implicit timestepper also has the prefix `stepper_`

## Serving suggestions

Sample options for rotating shallow water equations using G formulation.

`python implicit_fluids/timestepper.py -model_dt 400 -model_tmax 3600 -stepper_ksp_type fgmres -stepper_pc_type lu -stepper_pc_factor_mat_solver_type mumps -stepper_snes_ksp_ew -stepper_snes_atol 0. -stepper_snes_stol 0. -stepper_snes_rtol 1.0e-8 -stepper_snes_lag_preconditioner 200 -stepper_snes_lag_preconditioner_persists -stepper_snes_monitor ascii:monitor.dat -stepper_ksp_monitor ascii:monitor.dat -filename w6 -vtkfreq 1 -chkptfreq 0`
