from pathlib import Path

import numpy
from scipy.optimize import least_squares

from diffpy.apps.pdfadapter import PDFAdapter


def test_pdfadapter(ni_refined_parameters):
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
    adapter = PDFAdapter()
    adapter.initialize_profile(
        str(profile_path), q_range=(0.1, 25), calculation_range=(1.5, 50, 0.01)
    )
    adapter.initialize_structures([str(structure_path)])
    adapter.initialize_contribution(equation=["s0*G1"])
    adapter.initialize_recipe()
    adapter.recipe.addVar(list(adapter.recipe._contributions.values())[0].s0)
    adapter.set_initial_variable_values(initial_pv_dict)
    adapter.recipe.fix("all")
    for var in variables_to_refine:
        adapter.recipe.free(var)
        least_squares(adapter.recipe.residual, adapter.recipe.values)
    pdfadapter_pv_dict = {}
    for pname, parameter in adapter.recipe._parameters.items():
        pdfadapter_pv_dict[pname] = parameter.value
    for var_name in variables_to_refine:
        pdfadapter_value = pdfadapter_pv_dict[var_name]
        diffpy_value = ni_refined_parameters[var_name]
        assert numpy.isclose(diffpy_value, pdfadapter_value, atol=1e-5)
