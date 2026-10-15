from pathlib import Path

import numpy
import pytest
from helper import (
    run_multi_contribution_example,
    run_nanoparticle_example,
    run_ni_example,
)
from mcp import Client

from diffpy.apps.parametric_refinement.refinement_server import mcp

_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clear_session():
    # The MCP server uses a module-level singleton session; reset it so
    # models/profiles from one test don't leak into the next.
    from diffpy.apps.parametric_refinement.refinement_server import session

    session.clear()
    yield
    session.clear()


@pytest.mark.anyio
async def test_refine_sine():
    # C1: Set up the MCP client and do a sine refinement
    #   Expect all objs are created and refined successfully
    from diffpy.apps.parametric_refinement.refinement_server import session

    async with Client(mcp, raise_exceptions=True) as mcp_client:
        await mcp_client.call_tool(
            "add_profile_from_file",
            {
                "profile_name": "sine_profile",
                "profile_path": str(_DATA_DIR / "sine.dat"),
            },
        )
        assert "sine_profile" in session.profiles_dict
        await mcp_client.call_tool(
            "add_equation_model",
            {
                "model_name": "sine_model",
                "equation_string": "A*sin(x)",
            },
        )
        assert "sine_model" in session.models_dict
        await mcp_client.call_tool(
            "set_variables_value",
            {
                "name_value_dict": {"sine_model.A": 0.8},
            },
        )
        expected_value = 0.8
        actual_value = session.get_variable("sine_model.A")["value"]
        assert actual_value == expected_value
        await mcp_client.call_tool(
            "solve",
            {
                "profile_names": ["sine_profile"],
                "model_names": ["sine_model"],
                "variable_names": ["sine_model.A"],
            },
        )
        expected_value = 1.0  # The expected value of A after refinement
        actual_value = session.get_variable("sine_model.A")["value"]
        assert numpy.isclose(actual_value, expected_value, rtol=0.2)


@pytest.mark.anyio
async def test_refine_ni():
    from diffpy.apps.parametric_refinement.refinement_server import session

    ni_refined_parameters = run_ni_example()

    async with Client(mcp, raise_exceptions=True) as mcp_client:
        await mcp_client.call_tool(
            "add_profile_from_file",
            {
                "profile_name": "ni_profile",
                "profile_path": str(_DATA_DIR / "Ni.gr"),
            },
        )
        await mcp_client.call_tool(
            "set_profile_calculation_range",
            {
                "profile_name": "ni_profile",
                "xmin": 1.5,
                "xmax": 20,
                "dx": 0.01,
            },
        )
        await mcp_client.call_tool(
            "update_profile_meta",
            {
                "profile_name": "ni_profile",
                "meta": {"qmin": 0.1},
            },
        )
        await mcp_client.call_tool(
            "add_pdf_model",
            {
                "model_name": "pdf",
                "structure_file_path": str(_DATA_DIR / "Ni.cif"),
                "finite": False,
            },
        )
        await mcp_client.call_tool(
            "constrain_pdf_model_space_group_symmetry",
            {
                "model_name": "pdf",
                "space_group": "Fm-3m",
            },
        )
        await mcp_client.call_tool(
            "add_equation_model",
            {
                "model_name": "ni_model",
                "equation_string": "s*pdf",
            },
        )
        await mcp_client.call_tool(
            "combine_models",
            {
                "parent_model_name": "ni_model",
                "child_model_names": ["pdf"],
            },
        )
        variable_names = [
            "pdf.phase.lattice.a",
            "ni_model.s",
            "pdf.phase.Ni0.Uiso",
            "pdf.delta2",
            "pdf.qdamp",
            "pdf.qbroad",
        ]
        await mcp_client.call_tool(
            "set_variables_value",
            {
                "name_value_dict": dict(
                    zip(variable_names, [3.52, 0.4, 0.005, 2, 0.04, 0.02])
                ),
            },
        )
        await mcp_client.call_tool(
            "solve",
            {
                "profile_names": ["ni_profile"],
                "model_names": ["ni_model"],
                "variable_names": variable_names,
            },
        )
        name_to_cmi_name = {
            "ni_model.s": "s0",
            "pdf.phase.lattice.a": "G1_a",
            "pdf.phase.Ni0.Uiso": "G1_Uiso_0",
            "pdf.delta2": "G1_delta2",
            "pdf.qdamp": "qdamp",
            "pdf.qbroad": "qbroad",
        }
        for name, cmi_name in name_to_cmi_name.items():
            actual_value = session.get_variable(name)["value"]
            assert numpy.isclose(
                actual_value,
                ni_refined_parameters[cmi_name],
                rtol=1e-2,
            )


@pytest.mark.anyio
async def test_refine_multi_contribution():
    from diffpy.apps.parametric_refinement.refinement_server import session

    multi_contribution_refined_parameters = run_multi_contribution_example()

    async with Client(mcp, raise_exceptions=True) as mcp_client:
        await mcp_client.call_tool(
            "add_profile_from_file",
            {
                "profile_path": str(_DATA_DIR / "ni-q27r60-xray.gr"),
                "profile_name": "ni_xray",
            },
        )
        await mcp_client.call_tool(
            "add_profile_from_file",
            {
                "profile_path": str(_DATA_DIR / "ni-q27r100-neutron.gr"),
                "profile_name": "ni_neutron",
            },
        )
        await mcp_client.call_tool(
            "add_profile_from_file",
            {
                "profile_path": str(_DATA_DIR / "si-q27r60-xray.gr"),
                "profile_name": "si_xray",
            },
        )
        await mcp_client.call_tool(
            "add_profile_from_file",
            {
                "profile_path": str(_DATA_DIR / "si90ni10-q27r60-xray.gr"),
                "profile_name": "total_xray",
            },
        )
        for profile_name in [
            "ni_xray",
            "ni_neutron",
            "si_xray",
            "total_xray",
        ]:
            await mcp_client.call_tool(
                "set_profile_calculation_range",
                {"profile_name": profile_name, "xmax": 20},
            )
        await mcp_client.call_tool(
            "add_pdf_model",
            {
                "structure_file_path": str(_DATA_DIR / "Ni.cif"),
                "model_name": "pdf_ni",
                "finite": False,
            },
        )
        await mcp_client.call_tool(
            "constrain_pdf_model_space_group_symmetry",
            {"model_name": "pdf_ni"},
        )
        await mcp_client.call_tool(
            "add_pdf_model",
            {
                "from_model_name": "pdf_ni",
                "model_name": "pdf_ni_neutron",
                "finite": False,
            },
        )
        await mcp_client.call_tool(
            "add_pdf_model",
            {
                "from_model_name": "pdf_ni",
                "model_name": "pdf_ni_partial",
                "finite": False,
            },
        )
        await mcp_client.call_tool(
            "add_pdf_model",
            {
                "structure_file_path": str(_DATA_DIR / "si.cif"),
                "model_name": "pdf_si",
                "finite": False,
            },
        )
        await mcp_client.call_tool(
            "constrain_pdf_model_space_group_symmetry",
            {"model_name": "pdf_si"},
        )
        await mcp_client.call_tool(
            "add_pdf_model",
            {
                "from_model_name": "pdf_si",
                "model_name": "pdf_si_partial",
                "finite": False,
            },
        )
        await mcp_client.call_tool(
            "add_equation_model",
            {
                "model_name": "main",
                "equation_string": "scale * (pdf_ni_partial + pdf_si_partial)",
            },
        )
        await mcp_client.call_tool(
            "combine_models",
            {
                "parent_model_name": "main",
                "child_model_names": ["pdf_ni_partial", "pdf_si_partial"],
            },
        )
        await mcp_client.call_tool(
            "set_variables_value",
            {
                "name_value_dict": {
                    "pdf_ni.qdamp": 0.055,
                    "pdf_ni_neutron.qdamp": 0.030,
                    "pdf_ni_partial.qdamp": 0.052,
                    "pdf_si.qdamp": 0.051,
                    "pdf_si_partial.qdamp": 0.052,
                    "main.scale": 1.0,
                    "pdf_si.scale": 1.0,
                    "pdf_ni.scale": 1.0,
                },
            },
        )
        await mcp_client.call_tool(
            "solve",
            {
                "profile_names": [
                    "ni_xray",
                    "ni_neutron",
                    "si_xray",
                    "total_xray",
                ],
                "model_names": ["pdf_ni", "pdf_ni_neutron", "pdf_si", "main"],
                "residual_equations": ["resv", "resv", "resv", "resv"],
                "constraints": [
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
                "variable_names": [
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
            },
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


@pytest.mark.anyio
async def test_refine_nanoparticle_example():
    from diffpy.apps.parametric_refinement.refinement_server import session

    refined_parameters = run_nanoparticle_example()

    async with Client(mcp, raise_exceptions=True) as mcp_client:
        await mcp_client.call_tool(
            "add_profile_from_file",
            {
                "profile_path": str(_DATA_DIR / "pb_100_qmin1.gr"),
                "profile_name": "pb",
            },
        )
        await mcp_client.call_tool(
            "set_profile_calculation_range",
            {"profile_name": "pb", "xmin": 0.1, "xmax": 20.0},
        )
        await mcp_client.call_tool(
            "update_profile_meta",
            {"profile_name": "pb", "meta": {"qmax": 30.0}},
        )
        await mcp_client.call_tool(
            "add_function_model",
            {
                "model_name": "f",
                "function": "spherical_particle",
                "argnames": ["x", "psize"],
            },
        )
        await mcp_client.call_tool(
            "add_pdf_model",
            {
                "model_name": "pdf",
                "structure_file_path": str(_DATA_DIR / "pb.cif"),
                "finite": False,
            },
        )
        await mcp_client.call_tool(
            "constrain_pdf_model_space_group_symmetry",
            {"model_name": "pdf"},
        )
        await mcp_client.call_tool(
            "add_equation_model",
            {"model_name": "main", "equation_string": "f*pdf"},
        )
        await mcp_client.call_tool(
            "combine_models",
            {
                "parent_model_name": "main",
                "child_model_names": ["f", "pdf"],
            },
        )
        await mcp_client.call_tool(
            "set_variables_value",
            {
                "name_value_dict": {
                    "f.psize": 20,
                    "pdf.scale": 1,
                    "pdf.delta2": 0,
                    "pdf.phase.Pb0.Uiso": 0.5 / 8 / numpy.pi**2,
                },
            },
        )
        await mcp_client.call_tool(
            "solve",
            {
                "profile_names": ["pb"],
                "model_names": ["main"],
                "variable_names": [
                    "f.psize",
                    "pdf.scale",
                    "pdf.delta2",
                ],
                "include_sgpars": True,
            },
        )
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
