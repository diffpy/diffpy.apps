"""Tests of small, in-memory, reproducible ASE candidate libraries."""

import pytest

from diffpy.apps.screening.cluster import generate_cluster_library


@pytest.mark.parametrize(
    "requested, expected",
    [
        (
            # One from each motif, no dependency on an XYZ file.
            {
                "candidate_specs": [
                    {"motif": "ico", "noshells": 3},
                    {"motif": "oct", "length": 5, "cutoff": 2},
                    {"motif": "deca", "p": 3, "q": 2, "r": 0},
                ]
            },
            {"ico", "oct", "deca"},
        ),
        (
            # A deliberately small Cartesian grid.
            {"motif_ranges": {"ico": {"noshells": [2, 3]}}},
            {"ico"},
        ),
    ],
)
def test_generate_in_memory(requested, expected, tmp_path, monkeypatch):
    """Accept explicit or gridded selections without writing files."""
    monkeypatch.chdir(tmp_path)
    library = generate_cluster_library(
        "Pd",
        min_atoms=2,
        max_atoms=1200,
        max_clusters=20,
        **requested,
    )
    assert {model["motif"] for model in library["models"]} == expected
    assert len(library["models"]) <= 20
    for model in library["models"]:
        assert model["n_atoms"] == len(model["atoms"])
        assert model["model_id"]
        assert model["morphology"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "requested, message",
    [
        (
            # The maximum applies before the Cartesian product is built.
            {
                "motif_ranges": {
                    "deca": {"p": list(range(1, 7)), "q": [1, 2], "r": [0, 1]}
                }
            },
            "exceeding max_clusters=20",
        ),
        (
            # An explicit candidate list also must respect the budget.
            {
                "candidate_specs": [
                    {"motif": "ico", "noshells": n} for n in range(2, 23)
                ]
            },
            "exceeding max_clusters=20",
        ),
        (
            # No ambiguity about which builder parameters to use.
            {"candidate_specs": [{"motif": "deca", "p": 3, "q": 2}]},
            "requires exactly",
        ),
        (
            # Duplicate work should never enter a screen.
            {
                "candidate_specs": [
                    {"motif": "ico", "noshells": 3},
                    {"motif": "ico", "noshells": 3},
                ]
            },
            "Duplicate candidate",
        ),
    ],
)
def test_generation_rejects_invalid(requested, message):
    """Fail explicitly rather than silently clipping or duplicating."""
    with pytest.raises(ValueError, match=message):
        generate_cluster_library(
            "Pd",
            min_atoms=2,
            max_atoms=1200,
            max_clusters=20,
            **requested,
        )


def test_morphology_labels():
    """Retain subfamily distinctions needed for screening analysis."""
    lib = generate_cluster_library(
        "Pd",
        min_atoms=2,
        max_atoms=1200,
        candidate_specs=[
            {"motif": "oct", "length": 5, "cutoff": 2},
            {"motif": "deca", "p": 3, "q": 1, "r": 0},
            {"motif": "deca", "p": 3, "q": 2, "r": 0},
        ],
    )
    assert [m["morphology"] for m in lib["models"]] == [
        "cuboctahedron",
        "regular decahedron",
        "Ino-type decahedron",
    ]
