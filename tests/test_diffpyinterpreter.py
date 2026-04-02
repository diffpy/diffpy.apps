from pathlib import Path

import numpy
from helper import make_cmi_recipe
from scipy.optimize import least_squares

from diffpy.apps.diffpy_interpreter import DiffpyInterpreter


def test_meta_model():
    # C1: Run the same fit with pdfadapter and diffpy_cmi
    #   Expect the refined parameters to be the same within 1e-5
    # diffpy_cmi fitting
    structure_path = Path(__file__).parent / "data" / "Ni.cif"
    profile_path = Path(__file__).parent / "data" / "Ni.gr"
    initial_pv_dict = {
        "s0": 0.4,
        "qdamp": 0.04,
        "qbroad": 0.02,
        "G1_a": 3.52,
        "G1_delta2": 2,
        "G1_Uiso_0": 0.005,
    }
    variables_to_refine = [
        "G1_a",
        "s0",
        "G1_Uiso_0",
        "G1_delta2",
        "qdamp",
        "qbroad",
    ]
    diffpycmi_recipe = make_cmi_recipe(
        str(structure_path), str(profile_path), initial_pv_dict
    )
    structure_path = Path(__file__).parent / "data" / "Ni.cif"
    profile_path = Path(__file__).parent / "data" / "Ni.gr"
    diffpycmi_recipe = make_cmi_recipe(
        str(structure_path), str(profile_path), initial_pv_dict
    )
    diffpycmi_recipe.fithooks[0].verbose = 0
    diffpycmi_recipe.fix("all")
    for var_name in variables_to_refine:
        diffpycmi_recipe.free(var_name)
        least_squares(
            diffpycmi_recipe.residual,
            diffpycmi_recipe.values,
            x_scale="jac",
        )
    diffpy_pv_dict = {}
    for pname, parameter in diffpycmi_recipe._parameters.items():
        diffpy_pv_dict[pname] = parameter.value

    diffpy_dsl = f"""
load structure G1 from "{str(structure_path)}"
load profile exp_ni from "{str(profile_path)}"

set G1 spacegroup as auto
set exp_ni q_range as 0.1 25
set exp_ni calculation_range as 1.5 50 0.01
create equation variables s0
set equation as "s0*G1"

variables:
---
- G1_a: 3.52
- s0: 0.4
- G1_Uiso_0: 0.005
- G1_delta2: 2
- qdamp: 0.04
- qbroad: 0.02
---
"""
    diffpy_interpreter = DiffpyInterpreter()
    diffpy_interpreter.interpret(diffpy_dsl)
    diffpy_interpreter.configure_adapter()
    interpreter_results = diffpy_interpreter.run()
    for var_name in variables_to_refine:
        diffpy_value = diffpy_pv_dict[var_name]
        interpreter_value = interpreter_results["variables"][var_name]["value"]
        assert numpy.isclose(diffpy_value, interpreter_value, atol=1e-5)
