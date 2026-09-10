from pathlib import Path

import numpy
from helper import run_multi_contribution_example, run_ni_example

from diffpy.apps.refinebase.refinement_session import RefinementSession

_DATA_DIR = Path(__file__).parent / "data"


def test_refine_sine(sine_profile):
    # C1: Refinement session without additional calculator or functions
    session = RefinementSession()
    session.add_equation_model(model_name="sub", equation_str="a*x")
    session.add_equation_model(model_name="main", equation_str="A*sin(u)")
    session.combine_models(
        parent_model_name="main", child_model_names=["sub"], symbol="u"
    )
    session.set_variables_value(
        name_value_dict={
            "main.A": 0.8,
            "main.sub.a": 0.5,
        },
    )
    session._solve(
        name="sine",
        profiles=[sine_profile],
        models=[session.models_dict["main"]],
        variable_names=["main.A", "main.sub.a"],
    )
    assert numpy.isclose(
        session.get_variable("main.A")["value"],
        1.0,
        rtol=1e-2,
    )
    assert numpy.isclose(
        session.get_variable("main.sub.a")["value"],
        1.0,
        rtol=1e-2,
    )


def test_refine_ni():
    # C1: Refine Ni example using only a PDF model
    #  Expect the refined parameters are close to the ones
    #  obtained using diffpy.srfit script
    session = RefinementSession()
    session.add_profile_from_file(
        profile_name="ni_profile", profile_path=str(_DATA_DIR / "Ni.gr")
    )
    session.set_profile_calculation_range(
        profile_name="ni_profile",
        xmin=1.5,
        xmax=20,
        dx=0.01,
    )
    session.update_profile_meta(
        profile_name="ni_profile",
        meta={"qmin": 0.1},
    )
    session.add_pdf_model(
        model_name="pdf",
        structure_file_path=str(_DATA_DIR / "Ni.cif"),
    )
    session.constrain_pdf_model_space_group_symmetry(
        model_name="pdf", space_group="Fm-3m"
    )
    session.set_variables_value(
        name_value_dict={
            "pdf.phase.lattice.a": 3.52,
            "pdf.scale": 0.4,
            "pdf.phase.Ni0.Uiso": 0.005,
            "pdf.delta2": 2,
            "pdf.qdamp": 0.04,
            "pdf.qbroad": 0.02,
        },
    )
    session.solve(
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
            session.get_variable(name)["value"],
            ni_refined_parameters[cmi_name],
            rtol=1e-2,
        )
    # C2: Refine Ni example using a PDF model and a equation model to
    #  add the scale factor.
    session.set_variables_value({"pdf.scale": 1})
    session.add_equation_model(
        model_name="ni_model",
        equation_str="s*pdf",
    )
    session.combine_models(
        parent_model_name="ni_model", child_model_names=["pdf"]
    )
    session.set_variables_value(
        name_value_dict={
            "pdf.phase.lattice.a": 3.52,
            "ni_model.s": 0.4,
            "pdf.phase.Ni0.Uiso": 0.005,
            "pdf.delta2": 2,
            "pdf.qdamp": 0.04,
            "pdf.qbroad": 0.02,
        },
    )
    session.solve(
        profile_names=["ni_profile"],
        model_names=["ni_model"],
        variable_names=[
            "pdf.phase.lattice.a",
            "ni_model.s",
            "pdf.phase.Ni0.Uiso",
            "pdf.delta2",
            "pdf.qdamp",
            "pdf.qbroad",
        ],
    )
    name_to_cmi_name = {
        "ni_model.s": "s0",
        "ni_model.pdf.phase.lattice.a": "G1_a",
        "ni_model.pdf.phase.Ni0.Uiso": "G1_Uiso_0",
        "ni_model.pdf.delta2": "G1_delta2",
        "ni_model.pdf.qdamp": "qdamp",
        "ni_model.pdf.qbroad": "qbroad",
    }
    for name, cmi_name in name_to_cmi_name.items():
        assert numpy.isclose(
            session.get_variable(name)["value"],
            ni_refined_parameters[cmi_name],
            rtol=1e-2,
        )


def test_refine_multi_contribution():
    session = RefinementSession()
    session.add_profile_from_file(
        profile_path=str(_DATA_DIR / "ni-q27r60-xray.gr"),
        profile_name="ni_xray",
    )
    session.add_profile_from_file(
        profile_path=str(_DATA_DIR / "ni-q27r100-neutron.gr"),
        profile_name="ni_neutron",
    )
    session.add_profile_from_file(
        profile_path=str(_DATA_DIR / "si-q27r60-xray.gr"),
        profile_name="si_xray",
    )
    session.add_profile_from_file(
        profile_path=str(_DATA_DIR / "si90ni10-q27r60-xray.gr"),
        profile_name="total_xray",
    )
    session.set_profile_calculation_range(profile_name="ni_xray", xmax=20)
    session.set_profile_calculation_range(profile_name="ni_neutron", xmax=20)
    session.set_profile_calculation_range(profile_name="si_xray", xmax=20)
    session.set_profile_calculation_range(profile_name="total_xray", xmax=20)
    session.add_pdf_model(
        structure_file_path=str(_DATA_DIR / "Ni.cif"),
        model_name="pdf_ni",
    )
    session.constrain_pdf_model_space_group_symmetry("pdf_ni")
    session.add_pdf_model(
        from_model_name="pdf_ni",
        model_name="pdf_ni_neutron",
    )
    session.add_pdf_model(
        from_model_name="pdf_ni",
        model_name="pdf_ni_partial",
    )
    session.add_pdf_model(
        structure_file_path=str(_DATA_DIR / "si.cif"),
        model_name="pdf_si",
        structure_lib="PyObjcryst",
    )
    session.constrain_pdf_model_space_group_symmetry("pdf_si")
    session.add_pdf_model(
        from_model_name="pdf_si",
        model_name="pdf_si_partial",
    )
    session.add_equation_model(
        model_name="main",
        equation_str="scale * (pdf_ni_partial + pdf_si_partial)",
    )
    session.combine_models(
        parent_model_name="main",
        child_model_names=["pdf_ni_partial", "pdf_si_partial"],
    )
    session.set_variables_value(
        {
            "pdf_ni.qdamp": 0.055,
            "pdf_ni_neutron.qdamp": 0.030,
            "pdf_ni_partial.qdamp": 0.052,
            "pdf_si.qdamp": 0.051,
            "pdf_si_partial.qdamp": 0.052,
            "main.scale": 1.0,
            "pdf_si.scale": 1.0,
            "pdf_ni.scale": 1.0,
        }
    )
    session.solve(
        profile_names=["ni_xray", "ni_neutron", "si_xray", "total_xray"],
        model_names=["pdf_ni", "pdf_ni_neutron", "pdf_si", "main"],
        residual_equations=[
            "resv",
            "resv",
            "resv",
            "resv",
        ],
        constraints=[
            {"ni_delta2": 2.5, "si_delta2": 2.5, "pscale": 0.8},
            {
                "pdf_ni.delta2": "ni_delta2",
                "pdf_ni_neutron.delta2": "ni_delta2",
                "main.pdf_ni_partial.delta2": "ni_delta2",
                "pdf_si.delta2": "si_delta2",
                "main.pdf_si_partial.delta2": "si_delta2",
                "main.pdf_si_partial.scale": "1 - pscale",
                "main.pdf_ni_partial.scale": "pscale",
            },
        ],
        variable_names=[
            "pdf_ni.scale",
            "pdf_si.scale",
            "pdf_ni_neutron.scale",
            "main.scale",
            "pscale",
            "pdf_ni.phase.lattice.a",
            "pdf_ni.phase.Ni0.Uiso",
            "pdf_si.phase.a",
            "pdf_si.phase.Si.Biso",
            "ni_delta2",
            "si_delta2",
        ],
    )
    name_to_cmi_name = {
        "pdf_ni.scale": "xscale_ni",
        "pdf_si.scale": "xscale_si",
        "pdf_ni_neutron.scale": "nscale_ni",
        "main.scale": "xscale_sini",
        "pscale": "pscale_sini_ni",
        "pdf_ni.phase.lattice.a": "a_ni",
        "pdf_si.phase.a": "a_si",
        "ni_delta2": "delta2_ni",
        "si_delta2": "delta2_si",
    }
    multi_contribution_refined_parameters = run_multi_contribution_example()
    for name, cmi_name in name_to_cmi_name.items():
        assert numpy.isclose(
            session.get_variable(name)["value"],
            multi_contribution_refined_parameters[cmi_name],
            rtol=1e-2,
        )
    assert numpy.isclose(
        session.get_variable("pdf_ni.phase.Ni0.Uiso")["value"]
        * 8
        * numpy.pi**2,
        multi_contribution_refined_parameters["Biso_0_ni"],
        rtol=1e-2,
    )
    assert numpy.isclose(
        session.get_variable("pdf_si.phase.Si.Biso")["value"],
        multi_contribution_refined_parameters["Biso_0_si"],
        rtol=1e-2,
    )
