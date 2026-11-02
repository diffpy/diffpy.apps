"""MCP tool tests for stateless cluster screening."""

from pathlib import Path

import pytest
from mcp import Client

from diffpy.apps.parametric_refinement.refinement_server import mcp


@pytest.fixture
def anyio_backend():
    """The asyncio backend for tests involving the MCP client."""
    return "asyncio"


@pytest.mark.anyio
async def test_cluster_library_mcp_roundtrip(monkeypatch, tmp_path):
    """Preview in MCP and screen the same candidates without XYZ
    files."""
    from diffpy.apps.screening import cluster_screen

    # This test covers MCP registration and preview->screen transport.
    # Scientific refinement was validated in a separate real-Pd pilot.
    def fake_refine(candidate, *_):
        return {
            "rw": 0.25 if candidate["motif"] == "ico" else 0.4,
            "refined_parameters": {},
            "nfev": 2,
        }

    monkeypatch.setattr(cluster_screen, "_refine_one", fake_refine)
    pdf_path = tmp_path / "measured.gr"
    pdf_path.write_text("placeholder", encoding="utf-8")
    settings = {
        "r_range": [1.5, 20.0, 0.01],
        "q_range": [2.0, 23.0],
        "qdamp": 0.04,
        "qbroad": 0.01,
        "initial_values": {
            "scale": 0.2,
            "zoomscale": 1.01,
            "uiso": 0.016,
            "delta2": 4.5,
        },
    }
    async with Client(mcp, raise_exceptions=True) as client:
        preview_result = await client.call_tool(
            "generate_cluster_library",
            {
                "element": "Pd",
                "min_atoms": 2,
                "max_atoms": 1200,
                "max_clusters": 2,
                "candidate_specs": [
                    {"motif": "ico", "noshells": 3},
                    {"motif": "oct", "length": 5, "cutoff": 2},
                ],
            },
        )
        assert not preview_result.is_error
        preview = preview_result.structured_content
        assert preview is not None
        assert len(preview["models"]) == 2
        assert all("atoms" not in item for item in preview["models"])

        screen_result = await client.call_tool(
            "screen_cluster_library",
            {
                "preview": preview,
                "pdf_path": str(pdf_path),
                "refinement": settings,
            },
        )
        assert not screen_result.is_error
        screened = screen_result.structured_content
        assert screened is not None
        assert [r["rw"] for r in screened["results"]] == [0.25, 0.4]
        assert len(screened["results"]) == 2
        assert not list(Path(tmp_path).glob("*.xyz"))


@pytest.mark.anyio
async def test_preview_budget_rejected():
    """Reject a candidate list that exceeds the stated model budget."""

    async with Client(mcp, raise_exceptions=True) as client:
        result = await client.call_tool(
            "generate_cluster_library",
            {
                "element": "Pd",
                "min_atoms": 2,
                "max_atoms": 1200,
                "max_clusters": 1,
                "candidate_specs": [
                    {"motif": "ico", "noshells": 2},
                    {"motif": "ico", "noshells": 3},
                ],
            },
        )
        assert result.is_error


@pytest.mark.anyio
async def test_reject_changed_preview(tmp_path):
    """Do not fit a preview whose structure metadata was changed."""

    async with Client(mcp, raise_exceptions=True) as client:
        generated = await client.call_tool(
            "generate_cluster_library",
            {
                "element": "Pd",
                "min_atoms": 2,
                "max_atoms": 1200,
                "max_clusters": 1,
                "candidate_specs": [{"motif": "ico", "noshells": 3}],
            },
        )
        assert not generated.is_error
        preview = generated.structured_content
        preview["models"][0]["n_atoms"] = 999
        pdf_path = tmp_path / "unused.gr"
        pdf_path.write_text("placeholder", encoding="utf-8")
        result = await client.call_tool(
            "screen_cluster_library",
            {
                "preview": preview,
                "pdf_path": str(pdf_path),
                "refinement": {},
            },
        )
        assert result.is_error
