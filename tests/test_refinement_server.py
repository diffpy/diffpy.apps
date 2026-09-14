from pathlib import Path

import numpy
import pytest
from helper import run_ni_example
from mcp import Client

from diffpy.apps.refinebase.refinement_server import mcp

_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_refine_sine():
    # C1: Set up the MCP client and do a sine refinement
    #   Expect all objs are created and refined successfully
    from diffpy.apps.refinebase.refinement_server import session

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
                "equation_str": "A*sin(x)",
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
    from diffpy.apps.refinebase.refinement_server import session

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
                "equation_str": "s*pdf",
            },
        )
        await mcp_client.call_tool(
            "combine_models",
            {
                "parent_model_name": "ni_model",
                "child_model_names": ["pdf"],
                "symbol": "pdf",
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
