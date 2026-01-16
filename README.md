# Implicit Fluids

A collection of models for fluid dynamics (mostly geophysical fluid
dynamics) using Firedrake and Irksome. The system is entirely driven by
PETSc command line options.

See serving_suggestions for some possible command line options to get started.

## Guide to options

### specifying the model

These options all have the prefix `model_`

`model_type`. `swe` - Shallow water equations.

`model_variant`. `G` - The G formulation of the shallow water equations, where
D = H - div(G).
