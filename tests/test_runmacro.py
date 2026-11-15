from pathlib import Path
from types import SimpleNamespace

import numpy
from helper import run_ni_example

from diffpy.apps.parametric_refinement.parametric_macro import (
    MacroRunner,
    runmacro,
)

_DATA_DIR = Path(__file__).parent / "data"


def test_runmacro(tmp_path):
    # C1: Refine a nested sine model from a macro file with the same
    #  calls as test_refine_sine in test_refinement_session.py
    #  Expect the refined parameters to be close to the true values
    xarray = numpy.linspace(-numpy.pi, numpy.pi, 100)
    yarray = numpy.sin(xarray)
    macro_file = tmp_path / "sine.dp-in"
    macro_file.write_text(f"""
# nested sine model
add_profile_from_arrays(
    xarray={xarray.tolist()},
    yarray={yarray.tolist()},
    profile_name="sine_profile",
)
add_equation_model(model_name="sub", equation_string="a*x")
add_equation_model(model_name="main", equation_string="A*sin(sub)")
combine_models(parent_model_name="main", child_model_names=["sub"])
set_model_profile(model_name="main", profile_name="sine_profile")
set_variables_value(name_value_dict={{"main.A": 0.8, "main.sub.a": 0.5}})
solve(
    profile_names=["sine_profile"],
    model_names=["main"],
    variable_names=["main.A", "main.sub.a"],
)
""")
    runner = MacroRunner()
    runner.run_file(macro_file)
    for name in ["main.A", "main.sub.a"]:
        assert numpy.isclose(
            runner.session.get_variable(name)["value"], 1.0, rtol=1e-2
        )
    # C2: Run the same macro file through the `runmacro` application entry
    #  Expect no error
    runmacro(SimpleNamespace(file=str(macro_file)))
    # C3: Run a macro that calls a private session method
    #  Expect a ValueError naming the method
    try:
        runner.run('_get_model("main")')
    except ValueError as error:
        assert "_get_model" in str(error)
    else:
        raise AssertionError("ValueError not raised")


def test_runmacro_ni():
    # C1: Refine Ni example using only a PDF model from a macro
    #  Expect the refined parameters are close to the ones
    #  obtained using diffpy.srfit script
    runner = MacroRunner()
    runner.run(f"""
add_profile_from_file(
    profile_name="ni_profile", profile_path="{_DATA_DIR / "Ni.gr"}"
)
set_profile_calculation_range(
    profile_name="ni_profile", xmin=1.5, xmax=20, dx=0.01
)
update_profile_meta(profile_name="ni_profile", meta={{"qmin": 0.1}})
add_pdf_model(
    model_name="pdf",
    structure_file_path="{_DATA_DIR / "Ni.cif"}",
    finite=False,
)
constrain_pdf_model_space_group_symmetry(
    model_name="pdf", space_group="Fm-3m"
)
set_variables_value(
    name_value_dict={{
        "pdf.phase.lattice.a": 3.52,
        "pdf.scale": 0.4,
        "pdf.phase.Ni0.Uiso": 0.005,
        "pdf.delta2": 2,
        "pdf.qdamp": 0.04,
        "pdf.qbroad": 0.02,
    }}
)
solve(
    profile_names=["ni_profile"],
    model_names=["pdf"],
    variable_names=[
        "pdf.phase.lattice.a",
        "pdf.scale",
        "pdf.phase.Ni0.Uiso",
        "pdf.delta2",
        "pdf.qdamp",
        "pdf.qbroad",
    ],
)
""")
    name_to_cmi_name = {
        "pdf.scale": "s0",
        "pdf.phase.lattice.a": "G1_a",
        "pdf.phase.Ni0.Uiso": "G1_Uiso_0",
        "pdf.delta2": "G1_delta2",
        "pdf.qdamp": "qdamp",
        "pdf.qbroad": "qbroad",
    }
    ni_refined_parameters = run_ni_example()
    for name, cmi_name in name_to_cmi_name.items():
        assert numpy.isclose(
            runner.session.get_variable(name)["value"],
            ni_refined_parameters[cmi_name],
            rtol=1e-2,
        )
