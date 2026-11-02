"""Lightweight unit tests for ranking and compute-budget enforcement."""

import pytest

from diffpy.apps.screening.cluster import generate_cluster_library
from diffpy.apps.screening.cluster_screen import screen_cluster_library


@pytest.fixture
def two_clusters():
    return generate_cluster_library(
        "Pd",
        min_atoms=2,
        max_atoms=1200,
        max_clusters=2,
        candidate_specs=[
            {"motif": "ico", "noshells": 2},
            {"motif": "ico", "noshells": 3},
        ],
    )


def test_rank_and_keep_failures(two_clusters, tmp_path, monkeypatch):
    """Do not silently omit failed candidates from the ranked output."""
    import diffpy.apps.screening.cluster_screen as cluster_screen

    def fake_fit(candidate, *_):
        if candidate["n_atoms"] == 13:
            raise ValueError("simulated solver failure")
        return {"rw": 0.2, "refined_parameters": {}, "nfev": 3}

    monkeypatch.setattr(cluster_screen, "_refine_one", fake_fit)
    pdf = tmp_path / "dummy.gr"
    pdf.write_text("dummy")
    result = screen_cluster_library(
        two_clusters,
        pdf,
        config={
            "initial_values": {
                "scale": 0.2,
                "zoomscale": 1.0,
                "uiso": 0.01,
                "delta2": 2.0,
            }
        },
        max_clusters=2,
    )
    rows = result["results"]
    assert [row["status"] for row in rows] == ["converged", "failed"]
    assert [row["rank"] for row in rows] == [1, None]
    assert "simulated solver failure" in rows[1]["error"]
    assert all("atoms" not in row for row in rows)


def test_reject_excessive_library(two_clusters, tmp_path):
    """Refuse screens that exceed the number of permitted candidates."""
    with pytest.raises(ValueError, match="max_clusters"):
        screen_cluster_library(
            two_clusters, tmp_path / "missing.gr", {}, max_clusters=1
        )


def test_pd55_experimental_pdf_regression():
    """Check a Pd55 refinement against a measured PDF."""
    from pathlib import Path

    pdf_path = Path(__file__).parent / "data" / "Pd_cluster.gr"
    library = generate_cluster_library(
        "Pd",
        min_atoms=55,
        max_atoms=55,
        max_clusters=1,
        candidate_specs=[{"motif": "ico", "noshells": 3}],
    )
    config = {
        "r_range": [1.5, 20.0, 0.01],
        "q_range": [2.0, 23.0],
        "qdamp": 0.04,
        "qbroad": 0.01,
        "initial_values": {
            "scale": 1.0,
            "zoomscale": 1.01,
            "uiso": 0.01,
            "delta2": 1.0,
        },
        "ncores": 1,
    }

    result = screen_cluster_library(library, pdf_path, config, max_clusters=1)
    row = result["results"][0]

    assert row["status"] == "converged"
    assert row["n_atoms"] == 55
    assert row["rw"] == pytest.approx(0.5881, abs=0.005)
