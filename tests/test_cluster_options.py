"""Configuration and process-pool lifecycle tests for cluster
screening."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from diffpy.apps.screening import cluster_screen


def settings():
    """Return the minimal parameter settings used by these unit
    tests."""
    return {
        "initial_values": {
            "scale": 0.2,
            "zoomscale": 1.01,
            "uiso": 0.016,
            "delta2": 4.5,
        }
    }


def test_default_and_native_variable_selection():
    config = settings()
    free, bounds = cluster_screen._validate_options(config)
    assert free == ["cluster.scale", "zoomscale", "uiso", "cluster.delta2"]
    assert bounds == {}
    config.update(
        {
            "extra_variables": {"stretch_z": 1.0},
            "parameter_values": {"cluster.delta1": 0.0},
            "free_parameters": ["zoomscale", "stretch_z", "cluster.delta1"],
            "constraints": {"cluster.phase.lattice.c": "stretch_z"},
            "soft_bounds": {
                "stretch_z": {
                    "lower_bound": 0.9,
                    "upper_bound": 1.1,
                    "uncertainty": 0.01,
                }
            },
        }
    )
    free, bounds = cluster_screen._validate_options(config)
    assert free == ["zoomscale", "stretch_z", "cluster.delta1"]
    assert bounds["stretch_z"]["use_soft_bounds"] is True


def test_axial_override_preserves_other_ties():
    config = settings()
    config["extra_variables"] = {"stretch_z": 1.0}
    config["constraints"] = {"cluster.phase.lattice.c": "stretch_z"}
    names = [f"cluster.phase.lattice.{a}" for a in ("a", "b", "c")]
    names.extend(["cluster.phase.Pd0.Uiso", "cluster.phase.Pd1.Uiso"])
    model = SimpleNamespace(parameters=dict.fromkeys(names))
    helpers, ties = cluster_screen._constraints_for_model(
        model, config["initial_values"], config
    )
    assert helpers["stretch_z"] == 1.0
    assert ties["cluster.phase.lattice.a"] == "zoomscale"
    assert ties["cluster.phase.lattice.b"] == "zoomscale"
    assert ties["cluster.phase.lattice.c"] == "stretch_z"
    assert ties["cluster.phase.Pd0.Uiso"] == "uiso"
    config["constraints"]["cluster.phase.lattice.c"] = None
    _, ties = cluster_screen._constraints_for_model(
        model, config["initial_values"], config
    )
    assert "cluster.phase.lattice.c" not in ties


@pytest.mark.parametrize(
    "update, message",
    [
        ({"ncores": 0}, "ncores"),
        ({"ncores": True}, "ncores"),
        ({"ncores": 1.5}, "ncores"),
        ({"free_parameters": ["not_a_parameter"]}, "Unknown parameter"),
        ({"free_parameters": ["scale", "cluster.scale"]}, "Repeated"),
        ({"free_parameters": []}, "nonempty"),
        ({"extra_variables": {"uiso": 0.01}}, "new identifier"),
        ({"extra_variables": {"strain.z": 1.0}}, "new identifier"),
        ({"extra_variables": {"strain_z": float("nan")}}, "not finite"),
        ({"parameter_values": {"delta1": 0.0}}, "native"),
        ({"constraints": {"cluster.phase.lattice.c": 1.0}}, "equations"),
        ({"soft_bounds": {"uiso": {"use_soft_bounds": False}}}, "soft bounds"),
        ({"soft_bounds": {"uiso": {"uncertainty": 0.0}}}, "positive"),
        ({"optimizer": {"unsupported_option": 1}}, "optimizer"),
        ({"misspelled_option": 1}, "Unknown refinement"),
    ],
)
def test_invalid_options_raise(update, message):
    config = settings()
    config.update(update)
    with pytest.raises(ValueError, match=message):
        cluster_screen._validate_options(config)


def test_unknown_constraint_target_raises():
    config = settings()
    config["constraints"] = {"cluster.phase.lattice.not_an_axis": "zoomscale"}
    with pytest.raises(ValueError, match="Unknown constraint target"):
        cluster_screen._constraints_for_model(
            SimpleNamespace(parameters={}), config["initial_values"], config
        )


def test_serial_does_not_create_pool(tmp_path, monkeypatch):
    pdf = tmp_path / "unused.gr"
    pdf.write_text("unused", encoding="utf-8")
    library = {"models": [{"model_id": "one", "atoms": object()}]}
    original = settings()
    before = deepcopy(original)

    def no_pool(*args):
        raise AssertionError("Serial screening must not start a process pool")

    def fit(candidate, path, config, free, bounds, pool):
        assert pool is None
        return {"rw": 0.2, "refined_parameters": {}, "nfev": 1}

    monkeypatch.setattr(cluster_screen, "get_context", no_pool)
    monkeypatch.setattr(cluster_screen, "_refine_one", fit)
    result = cluster_screen.screen_cluster_library(library, pdf, original)
    assert result["execution"]["ncores"] == 1
    assert result["execution"]["start_method"] is None
    assert original == before
    assert result["results"][0]["elapsed_seconds"] >= 0
    assert "atoms" not in result["results"][0]


@pytest.mark.parametrize("interrupt", [False, True])
def test_one_pool_reused_and_closed(tmp_path, monkeypatch, interrupt):
    pdf = tmp_path / "unused.gr"
    pdf.write_text("unused", encoding="utf-8")
    library = {"models": [{"model_id": "one"}, {"model_id": "two"}]}
    events = []

    class Pool:
        def __init__(self, processes):
            events.append(("create", processes))

        def __enter__(self):
            events.append(("enter", self))
            return self

        def __exit__(self, *args):
            events.append(("exit", self))

    def context(method):
        assert method == "spawn"
        return SimpleNamespace(Pool=Pool)

    seen = []

    def fit(candidate, path, config, free, bounds, pool):
        seen.append(pool)
        if candidate["model_id"] == "two":
            if interrupt:
                raise KeyboardInterrupt()
            raise ValueError("deliberate candidate failure")
        return {"rw": 0.2, "refined_parameters": {}, "nfev": 1}

    monkeypatch.setattr(cluster_screen, "get_context", context)
    monkeypatch.setattr(cluster_screen, "_refine_one", fit)
    config = {**settings(), "ncores": 4}
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            cluster_screen.screen_cluster_library(library, pdf, config)
    else:
        result = cluster_screen.screen_cluster_library(library, pdf, config)
        assert [row["status"] for row in result["results"]] == [
            "converged",
            "failed",
        ]
    assert [event[0] for event in events] == ["create", "enter", "exit"]
    assert len(seen) == 2 and seen[0] is seen[1]


@pytest.mark.parametrize("solver_fails", [False, True])
def test_refine_one_forwards_options_and_detaches_pool(
    monkeypatch, solver_fails
):
    """Check solver forwarding, not PDF numerics, with a controlled
    session."""
    import numpy as np

    calls = []
    state = {}

    class Generator:
        def parallel(self, ncpu, mapfunc=None):
            calls.append((ncpu, mapfunc))

    class Session:
        def __init__(self):
            self.values = {}
            self.models_dict = {}
            self.profiles_dict = {}
            self.recipes_dict = {}

        def add_profile_from_file(self, path, profile_name):
            self.profiles_dict[profile_name] = SimpleNamespace(
                y=np.array([1.0, 2.0, 1.0]),
                ycalc=np.array([1.1, 1.9, 1.0]),
                dy=np.ones(3),
            )

        def set_profile_calculation_range(self, *args):
            pass

        def update_profile_meta(self, *args):
            pass

        def add_pdf_model(self, **kwargs):
            assert kwargs["run_parallel"] is False
            names = [f"cluster.phase.lattice.{a}" for a in ("a", "b", "c")]
            names += ["cluster.phase.Pd0.Uiso", "cluster.phase.Pd1.Uiso"]
            names += [
                "cluster.scale",
                "cluster.delta1",
                "cluster.delta2",
                "cluster.qdamp",
                "cluster.qbroad",
            ]
            self.models_dict["cluster"] = SimpleNamespace(
                parameters=dict.fromkeys(names), calc_obj=Generator()
            )

        def set_variables_value(self, values):
            self.values.update(values)

        def solve(self, **kwargs):
            state.update(kwargs)
            if solver_fails:
                raise RuntimeError("controlled solver failure")
            self.values.update(kwargs["constraints"][0])
            self.recipes_dict["cluster_fit"] = SimpleNamespace(
                get_values=lambda: np.zeros(len(kwargs["variable_names"])),
                residual=lambda values: None,
            )
            return {"success": True, "nfev": 3, "message": "test"}

        def get_variable(self, name):
            return {"value": self.values[name]}

    class Atoms:
        def __len__(self):
            return 2

        def get_chemical_symbols(self):
            return ["Pd", "Pd"]

        def get_positions(self):
            return np.array([[0, 0, 0], [2, 0, 0]])

    monkeypatch.setattr(cluster_screen, "RefinementSession", Session)
    config = settings()
    config.update(
        {
            "r_range": [1.5, 20.0, 0.01],
            "q_range": [2.0, 23.0],
            "qdamp": 0.04,
            "qbroad": 0.01,
            "ncores": 4,
            "extra_variables": {"stretch_z": 1.0},
            "constraints": {"cluster.phase.lattice.c": "stretch_z"},
            "parameter_values": {"cluster.delta1": 0.1},
            "free_parameters": ["zoomscale", "stretch_z", "cluster.delta1"],
        }
    )
    free, bounds = cluster_screen._validate_options(config)
    pool = SimpleNamespace(map=map)
    candidate = {"atoms": Atoms()}
    if solver_fails:
        with pytest.raises(RuntimeError, match="controlled solver failure"):
            cluster_screen._refine_one(
                candidate, "unused.gr", config, free, bounds, pool
            )
    else:
        row = cluster_screen._refine_one(
            candidate, "unused.gr", config, free, bounds, pool
        )
        assert row["n_parameters"] == 3
        assert row["refined_parameters"]["cluster.delta1"] == 0.1
        assert state["variable_names"] == free
        assert (
            state["constraints"][1]["cluster.phase.lattice.c"] == "stretch_z"
        )
    assert calls == [(4, map), (1, None)]
