import numpy
import pytest

from diffpy.apps.refinebase.parametric_model import (
    ParametricModelEquation,
    ParametricModelPDF,
)
from diffpy.srfit.fitbase import (
    Profile,
)
from diffpy.srfit.pdf import PDFParser
from diffpy.structure import Structure


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
    profile_path = "tests/data/Ni.gr"
    profile = Profile()
    parser = PDFParser()
    parser.parse_file(profile_path)
    profile.load_parsed_data(parser)
    profile.set_calculation_range(xmax=20)
    stru = Structure()
    structure_path = "tests/data/Ni.cif"
    stru.read(structure_path)
    pdf_model = ParametricModelPDF("ni", structure=stru)
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
