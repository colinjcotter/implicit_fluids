from implicit_fluids import gswe
from math import fabs


def test_energy():
    opts = {"model_dt": 1800,
            "model_tmax": 18000,
            "stepper_ksp_type": "fgmres",
            "stepper_pc_type": "lu",
            "stepper_pc_factor_mat_solver_type": "mumps",
            "stepper_snes_ksp_ew": None,
            "stepper_snes_atol": 0.,
            "stepper_snes_stol": 0.,
            "stepper_snes_rtol": 1.0e-12,
            "stepper_snes_lag_preconditioner": 200,
            "stepper_snes_lag_preconditioner_persists": None,
            "model_projection": None,
            "testcase_mesh_nrefs": 3,
            "testcase": "w6"
            }
    diagnostics = gswe(opts)
    energy = diagnostics["energy"]
    energy_error = (energy[-1]-energy[0])/energy[0]
    assert fabs(energy_error) < 1.0e-4  # larger than expected energy error
