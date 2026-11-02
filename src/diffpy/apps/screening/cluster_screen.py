"""Single-stage PDF screening of finite, in-memory cluster libraries."""

import re
from contextlib import nullcontext
from copy import deepcopy
from keyword import iskeyword
from math import isfinite
from multiprocessing import get_context
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np

from diffpy.apps.parametric_refinement.refinement_session import (
    RefinementSession,
)
from diffpy.structure import Atom, Structure

# Shared definitions

_VARIABLES = {
    "scale": "cluster.scale",
    "zoomscale": "zoomscale",
    "uiso": "uiso",
    "delta2": "cluster.delta2",
}


# Main cluster-screening function: refine, compare, and rank candidates


def screen_cluster_library(library, pdf_path, config, *, max_clusters=20):
    """Refine and rank a bounded library with one consistent
    configuration.

    Parameters
    ----------
    library : dict
        The in-memory output of ``generate_cluster_library``.
    pdf_path : str or pathlib.Path
        The measured PDF accepted by ``RefinementSession``.
    config : dict
        The r_range, q_range, qdamp, qbroad, and initial_values for the
        four aliases scale, zoomscale, uiso, and delta2. Optional ncores
        selects PDF-calculator worker processes (default 1). Optional
        extra_variables and constraints extend or override preset ties.
        Native cluster. names are accepted in free_parameters,
        parameter_values, and soft_bounds. Optional optimizer settings
        are forwarded to the existing solver. There is one fit stage.
    max_clusters : int, optional
        The maximum number of candidate refinements (default 20).

    Returns
    -------
    dict
        The ranked results, failures, settings, and wall-clock timings.

    Notes
    -----
    No structures are written to disk. Candidates are refined sequentially;
    ncores > 1 parallelizes each PDF calculation with a single reused spawn
    pool. Run multiprocessing scripts from a file with a __main__ guard.
    Global coordinate-axis scaling is not domain-resolved crystal strain.
    """
    if type(max_clusters) is not int or max_clusters < 1:
        raise ValueError("max_clusters must be a positive integer.")
    models = library["models"]
    if not 0 < len(models) <= max_clusters:
        raise ValueError("Library is empty or exceeds max_clusters.")
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"Experimental PDF not found: {pdf_path}")
    config = deepcopy(config)
    free, bounds = _validate_options(config)
    ncores = config.get("ncores", 1)
    started = perf_counter()
    manager = (
        get_context("spawn").Pool(processes=ncores)
        if ncores > 1
        else nullcontext(None)
    )
    rows = []
    with manager as pool:
        for candidate in models:
            meta = {
                key: val for key, val in candidate.items() if key != "atoms"
            }
            t0 = perf_counter()
            try:
                row = _refine_one(
                    candidate, pdf_path, config, free, bounds, pool
                )
                rows.append({**meta, **row, "status": "converged"})
            except Exception as exc:
                rows.append(
                    {
                        **meta,
                        "status": "failed",
                        "rw": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            rows[-1]["elapsed_seconds"] = perf_counter() - t0
    rows.sort(
        key=lambda row: (
            row["rw"] is None,
            row["rw"] if row["rw"] is not None else float("inf"),
        )
    )
    for i, row in enumerate(rows, 1):
        row["rank"] = i if row["rw"] is not None else None
    return {
        "settings": config,
        "max_clusters": max_clusters,
        "results": rows,
        "skipped_generation": library.get("skipped", []),
        "execution": {
            "ncores": ncores,
            "parallelism": "within_pdf",
            "start_method": "spawn" if ncores > 1 else None,
            "elapsed_seconds": perf_counter() - started,
        },
    }


# Internal helper functions for parameter validation, adding non-preset
# constraints, multiprocessing, and uniform result formatting.


def _finite_number(value, name):
    """Validate a numerical parameter without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number; provide a number.")
    if not isfinite(value):
        raise ValueError(f"{name} is not finite; provide a finite number.")
    return float(value)


def _parameter_name(name, extra):
    """Resolve a preset alias, helper variable, or native model name."""
    if not isinstance(name, str):
        raise ValueError("Parameter names must be strings.")
    if name in _VARIABLES:
        return _VARIABLES[name]
    if name in extra or name.startswith("cluster."):
        return name
    raise ValueError(
        f"Unknown parameter {name!r}; use a preset alias, an extra_variable "
        "name, or a registered 'cluster.' parameter."
    )


def _validate_options(config):
    """Validate optional execution settings and refinement
    extensions."""
    allowed = {
        "r_range",
        "q_range",
        "qdamp",
        "qbroad",
        "initial_values",
        "free_parameters",
        "soft_bounds",
        "ncores",
        "extra_variables",
        "parameter_values",
        "constraints",
        "optimizer",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown refinement settings; check the config keys.")
    initial = config.get("initial_values", {})
    if not isinstance(initial, dict) or set(initial) != set(_VARIABLES):
        raise ValueError(f"initial_values must contain {list(_VARIABLES)}.")
    for name, value in initial.items():
        _finite_number(value, name)
    ncores = config.get("ncores", 1)
    if type(ncores) is not int or ncores < 1:
        raise ValueError(
            "ncores must be a positive integer; use 1 for serial."
        )
    extra = config.get("extra_variables", {})
    values = config.get("parameter_values", {})
    overrides = config.get("constraints", {})
    for label, mapping in (
        ("extra_variables", extra),
        ("parameter_values", values),
        ("constraints", overrides),
    ):
        if not isinstance(mapping, dict):
            raise ValueError(f"{label} must be a dictionary.")
    for name, value in extra.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or iskeyword(name)
            or name in _VARIABLES
        ):
            raise ValueError(
                "extra_variables require new identifier names; do not "
                "replace the four preset aliases."
            )
        _finite_number(value, name)
    for name, value in values.items():
        if not isinstance(name, str) or not name.startswith("cluster."):
            raise ValueError("parameter_values require native cluster. names.")
        _finite_number(value, name)
    for target, expression in overrides.items():
        if (
            not isinstance(target, str)
            or not target.startswith("cluster.")
            or (
                expression is not None
                and (not isinstance(expression, str) or not expression)
            )
        ):
            raise ValueError(
                "constraints map native cluster. names to equations; "
                "use null to remove a preset tie."
            )
    selected = config.get("free_parameters", list(_VARIABLES))
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError("free_parameters must be a nonempty list.")
    free = [_parameter_name(name, extra) for name in selected]
    if len(set(free)) != len(free):
        raise ValueError("Repeated free parameters; select each only once.")
    soft = config.get("soft_bounds", {})
    if not isinstance(soft, dict):
        raise ValueError("soft_bounds must be a dictionary.")
    bounds = {}
    for name, settings in soft.items():
        native = _parameter_name(name, extra)
        if (
            not isinstance(settings, dict)
            or settings.get("use_soft_bounds", True) is not True
        ):
            raise ValueError("Only native soft bounds are supported.")
        if set(settings) - {
            "lower_bound",
            "upper_bound",
            "uncertainty",
            "scaled",
            "use_soft_bounds",
        }:
            raise ValueError("Unknown soft-bound option; check its spelling.")
        if native in bounds:
            raise ValueError("Repeated bounds; use one name per parameter.")
        if "uncertainty" in settings:
            sigma = _finite_number(settings["uncertainty"], "uncertainty")
            if sigma <= 0:
                raise ValueError("Soft-bound uncertainty must be positive.")
        bounds[native] = {**settings, "use_soft_bounds": True}
    optimizer = config.get("optimizer", {})
    if not isinstance(optimizer, dict) or set(optimizer) - {
        "method",
        "x_scale",
        "max_nfev",
        "ftol",
        "xtol",
    }:
        raise ValueError("Use optimizer options supported by session.solve.")
    return free, bounds


def _constraints_for_model(model, initial, config):
    """Compose the preset constraints with explicit user overrides."""
    helpers = {"zoomscale": initial["zoomscale"], "uiso": initial["uiso"]}
    helpers.update(config.get("extra_variables", {}))
    ties = {
        f"cluster.phase.lattice.{axis}": "zoomscale"
        for axis in ("a", "b", "c")
    }
    ties.update(
        {key: "uiso" for key in model.parameters if key.endswith(".Uiso")}
    )
    for target, expression in config.get("constraints", {}).items():
        if target not in model.parameters:
            raise ValueError(
                f"Unknown constraint target {target!r}; inspect the model "
                "parameters before requesting this constraint."
            )
        if expression is None:
            ties.pop(target, None)
        else:
            ties[target] = expression
    return helpers, ties


def _refine_one(candidate, pdf_path, config, free, bounds, pool=None):
    """Refine one candidate with an isolated, native
    RefinementSession."""
    atoms = candidate["atoms"]
    # A unit orthogonal cell preserves Cartesian coordinates during the
    # conversion. Changing its lengths then applies global axis scaling.
    structure = Structure(
        [
            Atom(symbol, xyz)
            for symbol, xyz in zip(
                atoms.get_chemical_symbols(),
                atoms.get_positions(),
                strict=True,
            )
        ]
    )
    session = RefinementSession()
    session.add_profile_from_file(str(pdf_path), profile_name="observed")
    session.set_profile_calculation_range("observed", *config["r_range"])
    qmin, qmax = config["q_range"]
    session.update_profile_meta("observed", {"qmin": qmin, "qmax": qmax})
    # Use the existing in-memory entry point with a constant assignment.
    session.add_pdf_model(
        model_name="cluster",
        finite=True,
        run_parallel=False,
        code="structure = supplied_structure\nspacegroup_symbol = 'P1'",
        global_namespace={"supplied_structure": structure},
    )
    initial = config["initial_values"]
    values = {
        "cluster.scale": initial["scale"],
        "cluster.delta2": initial["delta2"],
        "cluster.qdamp": config["qdamp"],
        "cluster.qbroad": config["qbroad"],
    }
    values.update(config.get("parameter_values", {}))
    model = session.models_dict["cluster"]
    helpers, ties = _constraints_for_model(model, initial, config)
    if sum(key.endswith(".Uiso") for key in model.parameters) != len(atoms):
        raise ValueError("Expected one Uiso parameter per cluster atom.")
    for name in free:
        if name in ties:
            raise ValueError(
                f"{name} is constrained; free its helper variable instead "
                "or explicitly remove that tie with a null override."
            )
        if name not in helpers and name not in model.parameters:
            raise ValueError(f"Unknown model parameter {name!r}.")
        if name in helpers and not any(
            re.search(rf"\b{re.escape(name)}\b", equation)
            for equation in ties.values()
        ):
            raise ValueError(
                f"Helper {name!r} is unused; reference it in a constraint "
                "or remove it from free_parameters."
            )
    for name in config.get("parameter_values", {}):
        if name in ties:
            raise ValueError(
                f"{name} is constrained; set the helper's initial value "
                "or remove its tie before assigning a native value."
            )
    session.set_variables_value(values)
    generator = model.calc_obj
    if pool is not None:
        generator.parallel(config.get("ncores", 1), mapfunc=pool.map)
    try:
        optimizer = {"method": "trf", "x_scale": 1.0}
        optimizer.update(config.get("optimizer", {}))
        fit = session.solve(
            profile_names=["observed"],
            model_names=["cluster"],
            variable_names=free,
            constraints=[helpers, ties],
            residual_equations=["chiv"],
            bounds=bounds,
            name="cluster_fit",
            **optimizer,
        )
        if not fit["success"]:
            raise RuntimeError(
                f"Refinement did not converge: {fit['message']}"
            )
        recipe = session.recipes_dict["cluster_fit"]
        recipe.residual(recipe.get_values())
        profile = session.profiles_dict["observed"]
        rw = np.linalg.norm((profile.y - profile.ycalc) / profile.dy)
        rw /= np.linalg.norm(profile.y / profile.dy)
        if not np.isfinite(rw):
            raise ValueError("Refinement produced a nonfinite Rw.")
        reported = dict(_VARIABLES)
        reported.update({name: name for name in helpers})
        reported.update(
            {name: name for name in config.get("parameter_values", {})}
        )
        for name in config.get("free_parameters", list(_VARIABLES)):
            reported[name] = _parameter_name(name, helpers)
        return {
            "rw": float(rw),
            "refined_parameters": {
                alias: float(session.get_variable(name)["value"])
                for alias, name in reported.items()
            },
            "n_parameters": len(recipe.get_values()),
            "nfev": int(fit["nfev"]),
            "message": str(fit["message"]),
        }
    finally:
        # The screen owns the pool; no calculator may keep using it after
        # the screen exits, including when an individual fit fails.
        if pool is not None:
            generator.parallel(1)
