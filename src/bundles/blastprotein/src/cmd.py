# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.commands import CmdDesc, AtomSpecArg
from chimerax.core.commands import StringArg, BoolArg, FloatArg, IntArg, EnumOf, Or
from chimerax.core.errors import UserError
from chimerax.seqalign import AlignSeqPairArg

from .data_model import AvailableDBs, AvailableMatrices
from .job import BlastProteinJob, manually_pull_blast_job
from .ui import find_match

# Use camel-case variable names for displaying keywords in help/usage
def blastprotein(session, atoms=None, database="pdb", cutoff=1.0e-3,
                 matrix="BLOSUM62", maxSeqs=100, log=None, *, name=None):
    if isinstance(atoms, tuple):
        # Must be alignment:seq
        alignment, chain = atoms
        str_chain = None
        for sc, c in sorted(alignment.associations.items()):
            if c is chain:
                str_chain = sc
                break
    else:
        if atoms is None:
            atoms = atomspec.everything(session)
        results = atoms.evaluate(session)
        chains = results.atoms.residues.unique_chains
        if len(chains) == 0:
            raise UserError("Cannot start BLAST job: no chain was specified or no model is open.")
        elif len(chains) > 1:
            raise UserError("Cannot start BLAST job: please choose exactly one chain (%d were specified)" %
                            len(chains))
        str_chain = chain = chains[0]
    if not str_chain:
        chain_spec = None
    else:
        chain_spec = str_chain.atomspec
        if chain_spec[0] == '/':
            # Make sure we have a structure spec in there so
            # the atomspec remains unique when we load structures later
            chain_spec = str_chain.structure.atomspec + chain_spec
    BlastProteinJob(session, chain.ungapped(), chain_spec,
                    database=database, cutoff=cutoff, matrix=matrix,
                    max_seqs=maxSeqs, log=log, tool_inst_name=name)


blastprotein_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg,
                                                   AlignSeqPairArg))],
                        keyword=[("database", EnumOf(AvailableDBs)),
                                 ("cutoff", FloatArg),
                                 ("matrix", EnumOf(AvailableMatrices)),
                                 ("maxSeqs", IntArg),
                                 ("log", BoolArg),
                                 ("name", StringArg),
                                 ],
                        synopsis="Search PDB/NR using BLAST")

def blastprotein_pull(session, jobid, log=None):
    manually_pull_blast_job(session, jobid, log)

blastprotein_pull_desc = CmdDesc(required=[("jobid", StringArg)],
        keyword=[("log", BoolArg)])
