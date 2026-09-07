import json
from pathlib import Path

import numpy
import pytest
from helper import make_cmi_recipe
from scipy.optimize import least_squares

from diffpy.srfit.fitbase import (
    Profile,
)


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path


@pytest.fixture
def nested_sine_model():
    from diffpy.apps.refinebase.parametric_model import ParametricModelEquation

    submodel = ParametricModelEquation("sub", "a*x")
    model = ParametricModelEquation("main", "A*sin(u)")
    model.register_submodel(symbol="u", submodel=submodel)
    return model, submodel


@pytest.fixture
def sine_profile():
    xobs = numpy.linspace(-numpy.pi, numpy.pi, 100)
    yobs = numpy.sin(xobs) + 1e-3 * numpy.random.normal(size=xobs.shape)
    sine_profile = Profile()
    sine_profile.setObservedProfile(xobs, yobs)
    return sine_profile


@pytest.fixture(scope="session")
def ni_refined_parameters():
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
    return diffpy_pv_dict
