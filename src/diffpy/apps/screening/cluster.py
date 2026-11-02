"""Build small, explicitly selected libraries of finite metal clusters.

This module generates ASE ``Atoms`` objects in memory, with reproducible
builder metadata. It does not decide how to sample a structural search
space, perform PDF refinements, or write structure files.
"""

from __future__ import annotations

import itertools
import math
from numbers import Real

from ase import __version__ as ase_version
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.data import atomic_numbers, reference_states

# Shared definitions

_MOTIF_FIELDS = {
    "ico": ("noshells",),
    "oct": ("length", "cutoff"),
    "deca": ("p", "q", "r"),
}


# Main cluster-generation function


def generate_cluster_library(
    element: str,
    *,
    min_atoms: int,
    max_atoms: int,
    max_clusters: int = 20,
    candidate_specs: list[dict] | None = None,
    motif_ranges: dict | None = None,
    lattice_constant: float | None = None,
) -> dict:
    """Generate a strictly capped, in-memory library of ASE clusters.

    Parameters
    ----------
    element : str
        The chemical symbol of the monatomic metal, e.g. ``"Pd"``.
    min_atoms : int
        The inclusive lower atom-count limit for accepted structures.
    max_atoms : int
        The inclusive upper atom-count limit for accepted structures.
    max_clusters : int, optional
        The maximum permitted number of candidate builder combinations.
        This is a safety budget, not a sampling strategy; default 20.
    candidate_specs : list[dict], optional
        The explicitly selected builders, e.g.
        ``[{"motif": "deca", "p": 5, "q": 1, "r": 0}]``.
    motif_ranges : dict, optional
        The per-motif ASE integer grids, e.g.
        ``{"ico": {"noshells": [2, 3]}}``. Their Cartesian-product
        size must fit within ``max_clusters``.
    lattice_constant : float, optional
        The reference fcc lattice constant in angstroms; uses the ASE
        reference for fcc elements when omitted.

    Returns
    -------
    dict
        The ``models`` list of ASE ``Atoms`` and corresponding metadata,
        ``skipped`` combinations with reasons, and the exact requested
        specifications. These Python objects are not directly MCP/JSON
        serializable; the MCP wrapper returns a metadata-only preview
        and regenerates the structures when screening.

    Notes
    -----
    This function does not refine, rank, write XYZ files, or choose a
    morphology-aware sampling strategy. The caller or agent skill is
    responsible for supplying a sparse, scientifically motivated candidate
    selection. A cap on the *number* of clusters alone does not bound PDF
    compute time: Debye calculations also become substantially more costly
    as the number of atoms grows.
    """
    if not isinstance(element, str):
        raise ValueError("element must be a valid chemical symbol, e.g. 'Pd'.")
    element = element.strip().capitalize()
    if element not in atomic_numbers:
        raise ValueError("element must be a valid chemical symbol, e.g. 'Pd'.")
    if (
        type(min_atoms) is not int
        or type(max_atoms) is not int
        or min_atoms < 1
        or max_atoms < min_atoms
    ):
        raise ValueError(
            "Require integer atom-count limits: 1 <= min_atoms <= max_atoms."
        )
    if type(max_clusters) is not int or max_clusters < 1:
        raise ValueError("max_clusters must be a positive integer.")

    specs = _select_candidates(candidate_specs, motif_ranges, max_clusters)
    a = _lattice_constant(element, lattice_constant)
    models, skipped = [], []
    for spec in specs:
        motif = spec["motif"]
        params = spec["parameters"]
        try:
            atoms = _make_atoms(element, motif, params, a)
        except ValueError as exc:
            skipped.append({**spec, "reason": str(exc)})
            continue
        n_atoms = len(atoms)
        if not min_atoms <= n_atoms <= max_atoms:
            skipped.append(
                {
                    **spec,
                    "reason": (
                        f"{n_atoms} atoms outside [{min_atoms}, {max_atoms}]"
                    ),
                }
            )
            continue
        model_id = "_".join(
            [
                motif,
                element,
                *(f"{k}{v}" for k, v in params.items()),
                f"N{n_atoms}",
            ]
        )
        models.append(
            {
                "model_id": model_id,
                "motif": motif,
                "morphology": _morphology(motif, params),
                "parameters": params,
                "element": element,
                "n_atoms": n_atoms,
                "atoms": atoms,
            }
        )
    if not models:
        raise ValueError(
            f"No structures fell within [{min_atoms}, {max_atoms}] atoms. "
            f"Of {len(specs)} requested candidates, {len(skipped)} were "
            "invalid or outside the size range. Adjust the input selection."
        )
    return {
        "element": element,
        "ase_version": ase_version,
        "lattice_constant_angstrom": a,
        "min_atoms": min_atoms,
        "max_atoms": max_atoms,
        "max_clusters": max_clusters,
        "requested_specs": specs,
        "models": models,
        "skipped": skipped,
    }


# Internal helper functions


def _lattice_constant(element, value):
    """Resolve the fcc lattice parameter for ideal cluster
    construction."""
    if value is None:
        ref = reference_states[atomic_numbers[element]]
        if not ref or ref.get("symmetry", "").lower() != "fcc":
            raise ValueError(
                f"No ASE fcc reference lattice constant for {element}. "
                "Provide lattice_constant explicitly."
            )
        value = ref.get("a")
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("lattice_constant must be a positive finite number.")
    return float(value)


def _make_atoms(element, motif, params, lattice_constant):
    """Construct one finite ASE cluster from its canonical
    specification."""
    if motif == "ico":
        return Icosahedron(
            element, params["noshells"], latticeconstant=lattice_constant
        )
    if motif == "oct":
        return Octahedron(
            element,
            params["length"],
            params["cutoff"],
            latticeconstant=lattice_constant,
        )
    return Decahedron(
        element,
        params["p"],
        params["q"],
        params["r"],
        latticeconstant=lattice_constant,
    )


def _morphology(motif, params):
    """Classify common ASE builders by their idealized morphologies."""
    if motif == "ico":
        return "Mackay icosahedron"
    if motif == "oct":
        length, cutoff = params["length"], params["cutoff"]
        if cutoff == 0:
            return "regular octahedron"
        if length == 2 * cutoff + 1:
            return "cuboctahedron"
        if length == 3 * cutoff + 1:
            return "regular truncated octahedron"
        return "truncated octahedron"
    q, r = params["q"], params["r"]
    if r > 0 and q > 1:
        return "Ino-truncated Marks decahedron"
    if r > 0:
        return "Marks decahedron"
    if q > 1:
        return "Ino-type decahedron"
    return "regular decahedron"


# Validate parameter grid options, giving agents using the cluster-mining
# diffpy skill flexibility to adjust the cluster library based on user goals
# and computational budgets.


def _validate_builder_integer(motif, field, value):
    """Validate one integer parameter passed to an ASE cluster
    builder."""
    minimum = 0 if field in ("cutoff", "r") else 1
    if type(value) is not int or value < minimum:
        raise ValueError(f"{motif}.{field} must be an integer >= {minimum}.")
    return value


def _validate_parameters(motif, parameters):
    """Validate one set of native ASE builder parameters.

    Parameters
    ----------
    motif : str
        The ASE cluster-builder family.
    parameters : dict
        The mapping of builder parameter names to integer values.

    Returns
    -------
    dict
        The validated parameters in a consistent key order.
    """
    if not isinstance(motif, str) or motif not in _MOTIF_FIELDS:
        raise ValueError(
            f"Unknown motif {motif!r}. Choose from {list(_MOTIF_FIELDS)}."
        )
    fields = _MOTIF_FIELDS[motif]
    if not isinstance(parameters, dict) or set(parameters) != set(fields):
        raise ValueError(
            f"{motif} requires exactly these parameters: {', '.join(fields)}."
        )
    normalized = {}
    for field in fields:
        normalized[field] = _validate_builder_integer(
            motif, field, parameters[field]
        )
    return normalized


def _select_candidates(candidate_specs, motif_ranges, max_clusters):
    """Expand candidate specifications without silently subsampling.

    Parameters
    ----------
    candidate_specs : list[dict] or None
        The explicit list of individual builder specifications.
    motif_ranges : dict or None
        The mapping of motif families to parameter grids.
    max_clusters : int
        The maximum number of builder combinations to consider.

    Returns
    -------
    list[dict]
        The normalized individual candidate specifications.

    Notes
    -----
    A parameter grid is a Cartesian product, not a morphology-aware sampler.
    Reject grids over the budget *before* building any clusters, even if
    filtering by atom count would later remove some combinations. The skill
    or caller must select a sparser grid; silently taking the first N would
    bias which structural families and sizes get tested.
    """
    if (candidate_specs is None) == (motif_ranges is None):
        raise ValueError(
            "Supply exactly one of candidate_specs or motif_ranges."
        )
    requested = []
    if candidate_specs is not None:
        if (
            not isinstance(candidate_specs, (list, tuple))
            or not candidate_specs
        ):
            raise ValueError("candidate_specs must be a nonempty list.")
        if len(candidate_specs) > max_clusters:
            raise ValueError(
                f"Requested {len(candidate_specs)} candidates, exceeding "
                f"max_clusters={max_clusters}. Choose a sparser selection."
            )
        for spec in candidate_specs:
            if not isinstance(spec, dict) or "motif" not in spec:
                raise ValueError(
                    "Each candidate must be a dict with a 'motif' field."
                )
            motif = spec["motif"]
            params = {
                key: value for key, value in spec.items() if key != "motif"
            }
            requested.append(
                {
                    "motif": motif,
                    "parameters": _validate_parameters(motif, params),
                }
            )
    else:
        if not isinstance(motif_ranges, dict) or not motif_ranges:
            raise ValueError("motif_ranges must be a nonempty dictionary.")
        unknown = set(motif_ranges) - set(_MOTIF_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown motifs: {sorted(unknown)}. "
                f"Choose from {list(_MOTIF_FIELDS)}."
            )
        # Calculate the Cartesian-product size first, without generating
        # structures or any potentially large list of combinations.
        n_requested = 0
        for motif, grid in motif_ranges.items():
            fields = _MOTIF_FIELDS[motif]
            if not isinstance(grid, dict) or set(grid) != set(fields):
                raise ValueError(
                    f"{motif} requires ranges for: {', '.join(fields)}."
                )
            for field in fields:
                values = grid[field]
                if not isinstance(values, (list, tuple, range)) or not values:
                    raise ValueError(
                        f"{motif}.{field} must be a nonempty integer sequence."
                    )
                for value in values:
                    _validate_builder_integer(motif, field, value)
                if len(set(values)) != len(values):
                    raise ValueError(
                        f"{motif}.{field} contains duplicate values."
                    )
            n_requested += math.prod(len(grid[f]) for f in fields)
        if n_requested > max_clusters:
            raise ValueError(
                f"Requested grid has {n_requested} combinations, exceeding "
                f"max_clusters={max_clusters}. Reduce the ranges or supply "
                "specific candidate_specs; no structures were generated."
            )
        for motif in _MOTIF_FIELDS:
            if motif not in motif_ranges:
                continue
            fields = _MOTIF_FIELDS[motif]
            for values in itertools.product(
                *(motif_ranges[motif][f] for f in fields)
            ):
                requested.append(
                    {
                        "motif": motif,
                        "parameters": _validate_parameters(
                            motif, dict(zip(fields, values))
                        ),
                    }
                )

    identities = [
        (spec["motif"], tuple(spec["parameters"].values()))
        for spec in requested
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "Duplicate candidate specifications found. "
            "Remove duplicates to avoid repeating refinements."
        )
    return requested
