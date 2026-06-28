from implicit_fluids.timestepper import run
from implicit_fluids.testcases import get_testcase
from implicit_fluids.models import GSWEModel
from firedrake.petsc import PETSc

def gswe(options_dictionary={}):
    opts = PETSc.Options()
    for k, v in options_dictionary.items():
        opts[k] = v

    model_opts = PETSc.Options('model_')
    testcase = get_testcase(model_opts)
    model = GSWEModel(testcase, model_opts)
    run(model)
