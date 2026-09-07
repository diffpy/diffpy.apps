import numpy

from diffpy.apps.refinebase.refinement_session import RefinementSession


def test_refine_sine(nested_sine_model, sine_profile):
    # C1: Refinement session without additional calculator or functions
    sine_model, submodel = nested_sine_model
    session = RefinementSession()
    session._solve(
        profiles=[sine_profile],
        models=[sine_model],
        variables=[
            sine_model.parameters["main.A"],
            sine_model.parameters["main.sub.a"],
        ],
        initial_values=[0.8, 0.5],
    )
    assert numpy.isclose(
        sine_model.parameters["main.A"].value,
        1.0,
        rtol=1e-2,
    )
    assert numpy.isclose(
        sine_model.parameters["main.sub.a"].value,
        1.0,
        rtol=1e-2,
    )


def test_refine_ni(ni_refined_parameters):
    # C1: Refine Ni example using only a PDF model
    #  Expect the refined parameters are close to the ones
    #  obtained using diffpy.srfit script
    session = RefinementSession()
    session.add_profile_from_file(
        profile_name="ni_profile", profile_path="tests/data/Ni.gr"
    )
    session.set_profile_calculation_range(
        profile_name="ni_profile",
        xmin=1.5,
        xmax=50,
        dx=0.01,
    )
    session.update_profile_meta(
        profile_name="ni_profile",
        meta={"qmin": 0.1},
    )
    session.add_model_from_structure_file(
        model_name="pdf",
        structure_file_path="tests/data/Ni.cif",
    )
    session.constrain_pdf_model_space_group_symmetry(
        model_name="pdf", space_group="Fm-3m"
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
        initial_values=[
            3.52,
            0.4,
            0.005,
            2,
            0.04,
            0.02,
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
    for name, cmi_name in name_to_cmi_name.items():
        assert numpy.isclose(
            session.get_variable(name)["value"],
            ni_refined_parameters[cmi_name],
            rtol=1e-2,
        )
    # C2: Refine Ni example using a PDF model and a equation model to
    #  add the scale factor.
    session.set_variable_value("pdf.scale", 1)
    session.add_model_from_equation(
        model_name="ni_model",
        equation_str="s*pdf",
    )
    session.combine_models(
        parent_model_name="ni_model", child_model_name="pdf"
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
        initial_values=[
            3.52,
            0.4,
            0.005,
            2,
            0.04,
            0.02,
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
