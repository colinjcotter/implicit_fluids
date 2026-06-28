from implicit_fluids.timestepper import run
from implicit_fluids.testcases import get_testcase
from implicit_fluids.models import GSWEModel
from firedrake.petsc import PETSc


def gswe(options_dictionary={}):
    opts = PETSc.Options()
    for k, v in options_dictionary.items():
        opts[k] = v

    testcase = get_testcase(opts)
    model_opts = PETSc.Options('model_')
    model = GSWEModel(testcase, model_opts)
    diagnostics = run(model)

    if options_dictionary:
        return diagnostics
