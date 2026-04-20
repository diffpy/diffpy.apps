import re
from pathlib import Path

import numpy
import pytest
from helper import make_cmi_recipe
from scipy.optimize import least_squares

from diffpy.apps.app_runmacro import MacroParser

_STRUCTURE_PATH = str(Path(__file__).parent / "data" / "Ni.cif")
_PROFILE_PATH = str(Path(__file__).parent / "data" / "Ni.gr")


def test_meta_model():
    # C1: Run the same fit with pdfadapter and diffpy_cmi
    #   Expect the refined parameters to be the same within 1e-5
    # diffpy_cmi fitting
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
        _STRUCTURE_PATH, _PROFILE_PATH, initial_pv_dict
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
load structure G1 from "{_STRUCTURE_PATH}"
load profile exp_ni from "{_PROFILE_PATH}"

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
    diffpy_interpreter = MacroParser()
    diffpy_interpreter.parse(diffpy_dsl)
    diffpy_interpreter.preprocess()
    interpreter_results = diffpy_interpreter.run()
    for var_name in variables_to_refine:
        diffpy_value = diffpy_pv_dict[var_name]
        interpreter_value = interpreter_results["variables"][var_name]["value"]
        assert numpy.isclose(diffpy_value, interpreter_value, atol=1e-5)


@pytest.mark.parametrize(
    "command_string, expected_inputs, expected_variables",
    [
        # C1: load structure G1 from "path/to/structure.cif"
        # Expect inputs and variables to be set correctly
        (
            f'load structure G1 from "{_STRUCTURE_PATH}"',
            {
                "initialize_structures.structure_paths": _STRUCTURE_PATH,
                "initialize_structures.names": ["G1"],
            },
            {"G1": "structure"},
        ),
        # C2: load profile exp_ni from "path/to/profile.gr"
        # Expect inputs and variables to be set correctly
        (
            f'load profile exp_ni from "{_PROFILE_PATH}"',
            {"initialize_profile.profile_path": _PROFILE_PATH},
            {"exp_ni": "profile"},
        ),
        # C3: create equation variables s0
        # Expect variable names to be stored
        (
            "create equation variables s0",
            {"add_contribution_variables.variable_names": ["s0"]},
            {},
        ),
        # C4: save to "results.json"
        # Expect result path to be stored
        (
            'save to "results.json"',
            {"save_results.result_path": "results.json"},
            {},
        ),
        # C5: set equation as "s0*G1"
        # Expect equation to be stored
        (
            'set equation as "s0*G1"',
            {"initialize_contribution.equation": ["s0*G1"]},
            {},
        ),
        # C6: set exp_ni q_range as 0.1 25
        # Expect q_range to be stored
        (
            f'load profile exp_ni from "{_PROFILE_PATH}"\n'
            "set exp_ni q_range as 0.1 25",
            {"initialize_profile.q_range": [0.1, 25]},
            {"exp_ni": "profile"},
        ),
        # C7: set exp_ni calculation_range as 1.5 50 0.01
        # Expect calculation_range to be stored
        (
            f'load profile exp_ni from "{_PROFILE_PATH}"\n'
            "set exp_ni calculation_range as 1.5 50 0.01",
            {"initialize_profile.calculation_range": [1.5, 50, 0.01]},
            {"exp_ni": "profile"},
        ),
        # C8: set G1 spacegroup as auto
        # Expect spacegroup to be stored
        (
            f'load structure G1 from "{_STRUCTURE_PATH}"\n'
            "set G1 spacegroup as auto",
            {"initialize_structures.spacegroups": ["auto"]},
            {"G1": "structure"},
        ),
        # C9: variables section with multiple variables
        # Expect variable names and values to be stored
        (
            """
variables:
---
- G1_a: 3.52
- s0
- G1_Uiso_0: 0.005
---
""",
            {
                "set_initial_variable_values.variable_name_to_value": {
                    "G1_a": 3.52,
                    "G1_Uiso_0": 0.005,
                },
                "refine_variables.variable_names": ["G1_a", "s0", "G1_Uiso_0"],
            },
            {},
        ),
        # C10: comments. Expect no error is raised by textx
        (
            f"""
# This is a comment
load structure G1 from "{_STRUCTURE_PATH}"
""",
            {
                "initialize_structures.structure_paths": _STRUCTURE_PATH,
                "initialize_structures.names": ["G1"],
            },
            {"G1": "structure"},
        ),
    ],
)
def test_command_processor(
    command_string, expected_inputs, expected_variables
):
    parser = MacroParser()
    parser.parse(command_string)
    for key, value in expected_inputs.items():
        assert parser.inputs[key] == value
    assert dict(parser.variables) == expected_variables


@pytest.mark.parametrize(
    "command_string, expected_exception, match",
    [
        (
            f'load unknown foo from "{_STRUCTURE_PATH}"',
            ValueError,
            "Unknown component type: unknown "
            "Please use 'structure' or 'profile'.",
        ),
        (
            'load structure foo from "/nonexistent/path.cif"',
            FileNotFoundError,
            f"structure {Path('/nonexistent/path.cif')} not found. "
            "Please ensure the path is correct and the file exists.",
        ),
        (
            f'load structure G1 from "{_STRUCTURE_PATH}"\n'
            f'load structure G2 from "{_STRUCTURE_PATH}"',
            ValueError,
            "Multiple structures are not supported by `runmacro` yet. "
            "Please use python script instead.",
        ),
        (
            f'load profile p1 from "{_PROFILE_PATH}"\n'
            f'load profile p2 from "{_PROFILE_PATH}"',
            ValueError,
            "Multiple profiles are not supported by `runmacro`. "
            "Please use python script instead.",
        ),
    ],
)
def test_load_command_processor_bad(command_string, expected_exception, match):
    parser = MacroParser()
    with pytest.raises(expected_exception, match=re.escape(match)):
        parser.parse(command_string)
