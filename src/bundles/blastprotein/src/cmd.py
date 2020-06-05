# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
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
from chimerax.seqalign import AlignSeqPairArg

DBs = ["pdb", "nr"]
Matrices = ["BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90",
            "PAM30", "PAM70", "PAM250", "IDENTITY"]

# Use camel-case variable names for displaying keywords in help/usage

def blastprotein(session, atoms=None, database="pdb", cutoff=1.0e-3,
                 matrix="BLOSUM62", maxSeqs=500, log=None, *, name=None):
    from .job import BlastProteinJob
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
            from chimerax.core.errors import UserError
            raise UserError("no chain was specified")
        elif len(chains) > 1:
            from chimerax.core.errors import UserError
            raise UserError("please choose exactly one chain (%d were specified)" %
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
                    database, cutoff, matrix, maxSeqs, log, name)
blastprotein_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg,
                                                   AlignSeqPairArg))],
                        keyword=[("database", EnumOf(DBs)),
                                 ("cutoff", FloatArg),
                                 ("matrix", EnumOf(Matrices)),
                                 ("maxSeqs", IntArg),
                                 ("log", BoolArg),
                                 ("name", StringArg),
                                 ],
                        synopsis="Search PDB/NR using BLAST")


def blastprotein_mav(session, name=None, selected=True):
    from . import tool
    tool.find_match(name).show_mav_cmd(selected)
blastprotein_mav_desc = CmdDesc(optional=[("name", StringArg)],
                                keyword=[("selected", BoolArg)])


def ccd(session, name):
    from .job import CCDJob
    CCDJob(session, name)
ccd_desc = CmdDesc(required=[("name", StringArg),],
                   synopsis="Get Chemical Component Dictionary template")
