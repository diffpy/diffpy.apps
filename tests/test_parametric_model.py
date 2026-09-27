from pathlib import Path

import numpy
import pytest

from diffpy.apps.refinebase.parametric_model import (
    ParametricModelEquation,
    ParametricModelFunction,
    create_pdf_model_from_file,
)
from diffpy.srfit.fitbase import (
    Profile,
)
from diffpy.srfit.pdf import PDFParser

_DATA_DIR = Path(__file__).parent / "data"


def test_parametric_model_graph():
    # C1: Create a nested parametric model
    # Expect the graph to be constructed correctly
    model = ParametricModelEquation("main", "A*sin(u)")
    submodel = ParametricModelEquation("sub", "a*x")
    model.register_submodel(symbol="u", submodel=submodel)
    expected_parameters = ["main.A", "main.sub.a", "main.sub.x"]
    actual_parameters = list(model.parameters.keys())
    assert set(expected_parameters) == set(actual_parameters)
    expected_edges = [
        ("main", "main.A"),
        ("main", "main.sub"),
        ("main.sub", "main.sub.a"),
        ("main.sub", "main.sub.x"),
    ]
    actual_edges = list(model._graph.edges)
    assert set(expected_edges) == set(actual_edges)


def test_parametric_model_parameter_access(nested_sine_model):
    # C1: Create a nested sine parametric model
    # Expect to models to share the same parameter obj
    model, submodel = nested_sine_model
    assert model.parameters["main.sub.a"] is submodel.parameters["sub.a"]


@pytest.mark.parametrize(
    "A, a, x, expected",
    [
        (1.0, 1.0, numpy.pi / 2, 1.0),
        (2.0, 2.0, numpy.pi / 4, 2.0),
        (1.0, 1.0, numpy.pi, 0.0),
    ],
)
def test_parametric_model_evaluation(nested_sine_model, A, a, x, expected):
    # C1: Create a nested sine parametric model
    # Expect the model to evaluate correctly
    model, submodel = nested_sine_model
    model.parameters["main.A"].value = A
    submodel.parameters["sub.a"].value = a
    submodel.parameters["sub.x"].value = x
    actual = model.evaluate()
    assert numpy.isclose(actual, expected, rtol=1e-6)


def test_parametric_pdf_model_parameters():
    # C1: Create a ParametricModelPDF for Ni
    # Expect the model to be initialized correctly with all parameters
    profile_path = str(_DATA_DIR / "Ni.gr")
    profile = Profile()
    parser = PDFParser()
    parser.parse_file(profile_path)
    profile.load_parsed_data(parser)
    profile.set_calculation_range(xmax=20)
    pdf_model = create_pdf_model_from_file(
        "ni", structure_file_path=str(_DATA_DIR / "Ni.cif")
    )
    parameter_names = [
        "ni.phase.lattice.a",
        "ni.phase.Ni0.Uiso",
        "ni.delta1",
        "ni.delta2",
        "ni.qbroad",
        "ni.scale",
        "ni.qdamp",
        "ni.phase.Ni0.occ",
        "ni.phase.Ni1.occ",
        "ni.phase.Ni2.occ",
        "ni.phase.Ni3.occ",
    ]
    assert set(parameter_names).issubset(set(pdf_model.parameters.keys()))
    pdf_model.constrain_symmetry("Fm-3m")
    assert set(parameter_names) == set(pdf_model.independent_parameters.keys())


def test_parametric_function_model():
    # C1: Create a ParametricModelFunction with a simple function
    # Expect the model to be initialized correctly with the function
    # registered
    def sine_func(x):
        return numpy.sin(x)

    xobs = numpy.arange(100)
    expected_y = sine_func(xobs)
    profile = Profile()
    profile.set_observed_profile(xobs, expected_y)
    func_model = ParametricModelFunction("func_model", function=sine_func)
    func_model.set_profile(profile)
    actual_y = func_model.evaluate()
    assert numpy.allclose(actual_y, expected_y, rtol=1e-6)
    expected_pnames = ["func_model.x"]
    assert list(expected_pnames) == list(func_model.parameters.keys())
    # C2: Create a PrametricModelFunction with a characteristic function
    #   Expect the model to evaluate correctly with the characteristic function
    spherical_fun = ParametricModelFunction(
        "spherical_func_model", function="spherical_particle"
    )
    expected_pnames = [
        "spherical_func_model.r",
        "spherical_func_model.particle_diameter",
    ]
    actual_pnames = list(spherical_fun.parameters.keys())
    assert set(expected_pnames) == set(actual_pnames)
    xobs = numpy.arange(100)
    expected_y = sine_func(xobs)
    profile = Profile()
    profile.set_observed_profile(xobs, expected_y)
    spherical_fun.set_profile(profile, xname="r")
    spherical_fun.parameters[
        "spherical_func_model.particle_diameter"
    ].value = 10
    r = xobs
    particle_diameter = 10
    # copied from diffpy.srfit.pdf.characteristic_functions.spherical_particle
    # start
    characteristic_function = numpy.zeros(numpy.shape(r), dtype=float)
    scaled_r = numpy.array(r, dtype=float) / particle_diameter
    inside = scaled_r < 1.0
    scaled_r_inside = scaled_r[inside]
    characteristic_function[inside] = (
        1.0 - 1.5 * scaled_r_inside + 0.5 * scaled_r_inside**3
    )
    # end
    expected_y = characteristic_function
    assert numpy.allclose(spherical_fun.evaluate(), expected_y, rtol=1e-6)
