# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from typing import Optional, Union
from chimerax.core.commands import (
    StringArg,
    BoolArg,
    FloatArg,
    IntArg,
    EnumOf,
    Or,
    CmdDesc,
    AtomSpecArg,
    atomspec,
)
from chimerax.atomic import SequenceArg, Sequence, Chain
from chimerax.core.errors import UserError
from chimerax.seqalign import AlignSeqPairArg

from .data_model import AvailableDBs, AvailableMatrices
from .job import BlastProteinJob


# Use camel-case variable names for displaying keywords in help/usage
def blastprotein(
    session,
    # TODO: Figure out what types are included in Or(AtomSpecArg, AlignSeqPairArg, SequenceArg)
    # Or is a union, clearly
    atoms=None,
    database="pdb",
    cutoff: float = 1.0e-3,
    matrix: str = "BLOSUM62",
    maxSeqs: int = 100,
    version: Optional[int] = None,
    showResultsTable: bool = True,
    loadStructures: bool = False,
    showSequenceAlignment: bool = False,
    onlyBest: bool = False,
    log=None,
    *,
    name=None
):
    """Search PDB/NR using BLAST"""
    str_chain = None
    if isinstance(atoms, tuple):
        # Must be alignment:seq
        alignment, chain = atoms
        str_chain = None
        for sc, c in sorted(alignment.associations.items()):
            if c is chain:
                str_chain = sc
                break
    elif isinstance(atoms, Sequence):
        chain = atoms
    else:
        if atoms is None:
            atoms = atomspec.everything(session)
        results = atoms.evaluate(session)
        chains = results.atoms.residues.unique_chains
        if len(chains) == 0:
            raise UserError(
                "Cannot start BLAST job: no chain was specified or no model is open."
            )
        elif len(chains) > 1:
            raise UserError(
                "Cannot start BLAST job: please choose exactly one chain (%d were specified)"
                % len(chains)
            )
        str_chain = chain = chains[0]
    if not str_chain:
        chain_spec = None
    else:
        chain_spec = str_chain.atomspec
        if chain_spec[0] == "/":
            # Make sure we have a structure spec in there so
            # the atomspec remains unique when we load structures later
            chain_spec = str_chain.structure.atomspec + chain_spec
    job = BlastProteinJob(
        session,
        chain.ungapped(),
        chain_spec,
        database=database,
        cutoff=cutoff,
        matrix=matrix,
        max_seqs=maxSeqs,
        version=version,
        log=log,
        tool_inst_name=name,
        show_gui=showResultsTable,
        load_structures=loadStructures,
        load_sequences=showSequenceAlignment,
        only_best=onlyBest,
    )
    job.start()

    return job

blastprotein_desc = CmdDesc(
    required=[("atoms", Or(AtomSpecArg, AlignSeqPairArg, SequenceArg))],
    keyword=[
        ("database", EnumOf(AvailableDBs)),
        ("cutoff", FloatArg),
        ("matrix", EnumOf(AvailableMatrices)),
        ("maxSeqs", IntArg),
        ("version", StringArg),
        ("log", BoolArg),
        ("name", StringArg),
        ("showResultsTable", BoolArg),
        ("loadStructures", BoolArg),
        ("showSequenceAlignment", BoolArg),
        ("onlyBest", BoolArg),
    ],
    synopsis=blastprotein.__doc__.split("\n")[0].strip(),
)
