from __future__ import annotations

import os
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter


def download_mp_cifs(
    query: str,
    start_index: int,
    end_index: int,
    output_dir: str | Path = "mp_cifs",
    api_key: str | None = None,
) -> list[Path]:
    """
    Download a consecutive range of Materials Project structures as CIF files.

    Parameters
    ----------
    query
        One of:
        - Materials Project ID: "mp-149"
        - Formula: "SiO2" or "LiNbO3"
        - Chemical system: "Li-Ti-Nb-O"
    start_index, end_index
        Zero-based inclusive indices in the returned MP result list.
    output_dir
        Directory in which CIF files will be written.
    api_key
        Materials Project API key. If omitted, reads MP_API_KEY from
        the environment.

    Returns
    -------
    list[Path]
        Paths of the CIF files written.
    """
    if start_index < 0 or end_index < start_index:
        raise ValueError("Require 0 <= start_index <= end_index.")

    api_key = api_key or os.environ["MP_API_KEY"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with MPRester(api_key) as mpr:
        if query.startswith("mp-"):
            docs = mpr.materials.summary.search(
                material_ids=[query],
                fields=["material_id", "structure"],
            )
        elif "-" in query:
            # Example: "Li-Ti-Nb-O"
            docs = mpr.materials.summary.search(
                chemsys=query,
                fields=["material_id", "structure"],
            )
        else:
            # Example: "SiO2" or "LiNbO3"
            docs = mpr.materials.summary.search(
                formula=[query],
                fields=["material_id", "structure"],
            )

    if end_index >= len(docs):
        raise IndexError(
            f"Requested index {end_index}, but MP returned only "
            f"{len(docs)} structure(s)."
        )

    written_files = []

    for index in range(start_index, end_index + 1):
        doc = docs[index]
        filename = output_dir / f"{index:04d}_{doc.material_id}.cif"
        CifWriter(doc.structure).write_file(filename)
        written_files.append(filename)

    return written_files
