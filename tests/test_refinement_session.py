from pathlib import Path

import numpy
from helper import (
    run_c60_example,
    run_multi_contribution_example,
    run_nanoparticle_example,
    run_ni_example,
)

from diffpy.apps.refinebase.refinement_session import RefinementSession

_DATA_DIR = Path(__file__).parent / "data"


def test_refine_sine(sine_profile):
    # C1: Refinement session without additional calculator or functions
    session = RefinementSession()
    session.add_equation_model(model_name="sub", equation_str="a*x")
    session.add_equation_model(model_name="main", equation_str="A*sin(sub)")
    session.combine_models(parent_model_name="main", child_model_names=["sub"])
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
            {
                "pscale": 0.8,
                "ni_delta2": 2.5,
                "si_delta2": 2.5,
            },
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


def test_refine_nanoparticle_example():
    session = RefinementSession()
    session.add_profile_from_file(
        profile_path=str(_DATA_DIR / "pb_100_qmin1.gr"),
        profile_name="pb",
    )
    session.set_profile_calculation_range(
        profile_name="pb", xmin=0.1, xmax=20.0
    )
    session.update_profile_meta(profile_name="pb", meta={"qmax": 30.0})
    # default argname is ['r', 'particle_diameter']
    # xname must match the one in profile
    # default xname in profile is "x"
    session.add_function_model(
        function="spherical_particle", model_name="f", argnames=["x", "psize"]
    )
    session.add_pdf_model(
        model_name="pdf",
        structure_file_path=str(_DATA_DIR / "pb.cif"),
    )
    session.constrain_pdf_model_space_group_symmetry(model_name="pdf")
    session.add_equation_model(model_name="main", equation_str="f*pdf")
    session.combine_models(
        parent_model_name="main", child_model_names=["f", "pdf"]
    )
    session.set_variables_value(
        {
            "f.psize": 20,
            "pdf.scale": 1,
            "pdf.delta2": 0,
            "pdf.phase.Pb0.Uiso": 0.5 / 8 / numpy.pi**2,
        }
    )
    session.solve(
        profile_names=["pb"],
        model_names=["main"],
        variable_names=[
            "f.psize",
            "pdf.scale",
            "pdf.delta2",
        ],
        include_sgpars=True,
    )
    refined_parameters = run_nanoparticle_example()
    name_to_cmi_name = {
        "pdf.phase.lattice.a": "a",
        "f.psize": "particle_diameter",
        "pdf.scale": "scale",
        "pdf.delta2": "delta2",
    }
    for name, cmi_name in name_to_cmi_name.items():
        assert numpy.isclose(
            session.get_variable(name)["value"],
            refined_parameters[cmi_name],
            rtol=1e-2,
        )

    assert numpy.isclose(
        session.get_variable("pdf.phase.Pb0.Uiso")["value"]
        * (8 * numpy.pi**2),
        refined_parameters["Biso_0"],
        rtol=1e-2,
    )


def test_refine_c60_example():
    make_c60_py = (_DATA_DIR / "make_c60.txt").read_text()
    session = RefinementSession()
    session.add_profile_from_file(
        profile_path=str(_DATA_DIR / "C60.gr"), profile_name="c60", xname="r"
    )
    session.set_profile_calculation_range(profile_name="c60", xmin=1.2, xmax=8)
    session.update_profile_meta(
        profile_name="c60", meta={"qmin": 0.68, "qmax": 22.0}
    )
    session.add_pdf_model(
        model_name="pdf",
        code=make_c60_py,
        local_structure_name="molecule",  # The structure name in 'code'
        global_namespace={"c60xyz_path": str(_DATA_DIR / "c60xyz.txt")},
        finite=True,
    )
    for i in range(1, 61):
        # Agent can do iteration outside of the MCP server
        session.add_pdf_bond_length_parameter(
            model_name="pdf",
            par_name=f"rad{i}",
            obj1_name="pdf.phase.center",
            obj2_name=f"pdf.phase.C{i}",
        )

    session.models_dict["pdf"].independent_parameters["pdf.phase.C1.x"]
    initial_radius = session.get_variable("pdf.phase.rad1")["value"]
    initial_biso = session.get_variable("pdf.phase.C1.Biso")["value"]
    biso_constraints = {f"pdf.phase.C{i}.Biso": "biso" for i in range(1, 61)}
    radius_constraints = {f"pdf.phase.rad{i}": "radius" for i in range(1, 61)}
    variable_constraints = {**radius_constraints, **biso_constraints}
    session.set_variables_value({"pdf.delta2": 2, "pdf.scale": 1.3e4})
    session.solve(
        profile_names=["c60"],
        model_names=["pdf"],
        constraints=[
            {"radius": initial_radius, "biso": initial_biso},
            variable_constraints,
        ],
        variable_names=["radius", "biso", "pdf.delta2", "pdf.scale"],
    )
    refined_parameters = run_c60_example()
    name_to_cmi_name = {
        "pdf.delta2": "delta2",
        "pdf.scale": "scale",
        "radius": "radius",
        "biso": "Biso",
    }
    for name, cmi_name in name_to_cmi_name.items():
        assert numpy.isclose(
            session.get_variable(name)["value"],
            refined_parameters[cmi_name],
            rtol=1e-2,
        )
