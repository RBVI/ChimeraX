# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

import csv
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from chimerax.core.errors import UserError

from .install import managed_fasthydromap_executable

ATTR_NAME = "fasthydromap_score"
DEFAULT_PALETTE = "^lipophilicity"
DEFAULT_RANGE = "4,6.5"


def fasthydromap(
    session,
    structures=None,
    *,
    color=True,
    target="acs",
    show_atoms=None,
    install_location=None,
):
    from chimerax.atomic import Residue, all_atomic_structures

    if structures is None:
        structures = list(all_atomic_structures(session))
    else:
        structures = list(structures)

    if not structures:
        raise UserError("fasthydromap: no atomic structures specified")

    Residue.register_attr(session, ATTR_NAME, "FastHydroMap", attr_type=float)

    colored = 0
    for structure in structures:
        assigned = _assign_fasthydromap_scores(
            session,
            structure,
            install_location=install_location,
        )
        if assigned == 0:
            session.logger.warning(f"FastHydroMap assigned no residues for {structure}")
            continue
        colored += 1
        if color:
            _color_structure(session, structure, target=target, show_atoms=show_atoms)

    if colored == 0:
        raise UserError("fasthydromap: no structures produced usable FastHydroMap predictions")


def _assign_fasthydromap_scores(session, structure, *, install_location):
    scores = _predict_scores(
        session,
        structure,
        install_location=install_location,
    )
    if not scores:
        return 0

    no_chain_lookup = {}
    for residue in structure.residues:
        label = _residue_label(residue, include_chain=False)
        if label is not None and label not in no_chain_lookup:
            no_chain_lookup[label] = residue

    assigned = 0
    for residue_label, score in scores.items():
        residue = _find_structure_residue(structure, residue_label, no_chain_lookup)
        if residue is None:
            session.logger.warning(
                f"FastHydroMap could not map prediction {residue_label} onto {structure}"
            )
            continue
        setattr(residue, ATTR_NAME, float(score))
        assigned += 1

    session.logger.info(f"FastHydroMap assigned {assigned} residues for {structure}")
    return assigned


def _predict_scores(session, structure, *, install_location):
    with TemporaryDirectory(prefix="fasthydromap_") as temp_dir:
        temp_path = Path(temp_dir)
        pdb_path = temp_path / "model.pdb"
        outroot = temp_path / "fasthydromap"
        csv_path = Path(f"{outroot}.csv")

        from chimerax.pdb import save_pdb
        save_pdb(session, str(pdb_path), models=[structure])

        command = _fasthydromap_command(
            session,
            pdb_path,
            outroot,
            install_location=install_location,
        )
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        except FileNotFoundError as err:
            raise UserError(
                "fasthydromap: could not find the FastHydroMap executable. "
                "Run 'fasthydromap install' or set FASTHYDROMAP_EXE."
            ) from err
        except subprocess.CalledProcessError as err:
            stderr = (err.stderr or "").strip()
            stdout = (err.stdout or "").strip()
            detail = stderr or stdout or str(err)
            raise UserError(f"fasthydromap failed for {structure}: {detail}") from err

        if not csv_path.exists():
            detail = (proc.stderr or proc.stdout or "").strip()
            raise UserError(
                f"fasthydromap did not produce {csv_path.name} for {structure}: {detail}"
            )
        return _read_single_structure_scores(csv_path)


def _fasthydromap_command(session, pdb_path, outroot, *, install_location):
    exe_override = os.environ.get("FASTHYDROMAP_EXE")
    if exe_override:
        return [exe_override, "predict", str(pdb_path), "-o", str(outroot)]

    managed_exe = managed_fasthydromap_executable(session, install_location=install_location)
    if managed_exe:
        return [managed_exe, "predict", str(pdb_path), "-o", str(outroot)]

    raise UserError(
        "FastHydroMap is not installed in ChimeraX yet. "
        "Run 'fasthydromap install' first, or set FASTHYDROMAP_EXE."
    )


def _read_single_structure_scores(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise UserError(f"fasthydromap output {csv_path} is missing prediction rows")

    header = rows[0]
    if header and header[0] == "frame":
        values = rows[1]
        if len(header) != len(values):
            raise UserError(f"fasthydromap output {csv_path} has inconsistent column counts")
        return {label: float(score) for label, score in zip(header[1:], values[1:])}

    if "residue" in header and "Fdewet" in header:
        residue_index = header.index("residue")
        score_index = header.index("Fdewet")
        scores = {}
        for row in rows[1:]:
            if len(row) <= max(residue_index, score_index):
                raise UserError(f"fasthydromap output {csv_path} has inconsistent column counts")
            scores[row[residue_index]] = float(row[score_index])
        return scores

    raise UserError(f"fasthydromap output {csv_path} has an unexpected header")


def _color_structure(session, structure, *, target, show_atoms):
    from chimerax.core.commands import run

    spec = structure.atomspec
    if "s" in target:
        run(session, f"show {spec} surface")
    if "c" in target:
        run(session, f"show {spec} cartoon")
    if show_atoms is None:
        show_atoms = "s" not in target
    if show_atoms:
        run(session, f"show {spec} atoms")
        run(session, f"show {spec} bonds")
    else:
        run(session, f"hide {spec} atoms")
    run(
        session,
        f"color byattribute r:{ATTR_NAME} {spec} target {target} palette {DEFAULT_PALETTE}"
            f" range {DEFAULT_RANGE}"
            f" novalue gray",
    )


def _find_structure_residue(structure, residue_label, no_chain_lookup):
    chain_id, resid, insertion_code = _parse_residue_label(residue_label)
    if chain_id is None:
        return no_chain_lookup.get(f"{resid}{insertion_code}")
    insert = insertion_code or " "
    return structure.find_residue(chain_id, resid, insert=insert)


def _parse_residue_label(label):
    if ":" in label:
        chain_id, base = label.split(":", 1)
        chain_id = chain_id if chain_id != "_" else " "
    else:
        chain_id = None
        base = label
    match = re.match(r"(-?\d+)(.*)$", base)
    if match is None:
        raise UserError(f"fasthydromap returned an unrecognized residue label: {label}")
    resid = int(match.group(1))
    insertion_code = match.group(2).strip()
    return chain_id, resid, insertion_code


def _residue_label(residue, *, include_chain):
    if residue.polymer_type != residue.PT_AMINO:
        return None
    atom_names = {a.name for a in residue.atoms}
    if not {"N", "CA", "C"}.issubset(atom_names):
        return None
    base = f"{residue.number}{residue.insertion_code.strip()}"
    if include_chain:
        chain_id = residue.chain_id.strip() or "_"
        return f"{chain_id}:{base}"
    return base


def register_fasthydromap_command(logger):
    from chimerax.core.commands import (
        BoolArg,
        CmdDesc,
        Or,
        EmptyArg,
        SaveFolderNameArg,
        register,
    )
    from chimerax.atomic import AtomicStructuresArg
    from chimerax.std_commands.color import TargetArg

    desc = CmdDesc(
        required=[("structures", Or(AtomicStructuresArg, EmptyArg))],
        keyword=[
            ("color", BoolArg),
            ("target", TargetArg),
            ("show_atoms", BoolArg),
            ("install_location", SaveFolderNameArg),
        ],
        synopsis="Predict per-residue Fdewet with FastHydroMap and optionally color the result",
    )
    register("fasthydromap", desc, fasthydromap, logger=logger)
