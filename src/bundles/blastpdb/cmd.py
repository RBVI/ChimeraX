# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.commands import CmdDesc, AtomSpecArg
from chimerax.core.commands import StringArg, BoolArg, FloatArg, IntArg, EnumOf

DBs = ["pdb", "nr"]
Matrices = ["BLOSUM45", "BLOSUM62", "BLOSUM80", "BLOSUM90", "BLOSUM100",
            "PAM30", "PAM70"]

def blastpdb(session, spec=None, database="pdb", cutoff=1.0e-3,
             matrix="BLOSUM62", max_hits=500,
             finish_callback=None, fail_callback=None):
    from .job import BlastPDBJob
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    chains = results.atoms.residues.unique_chains
    if len(chains) == 0:
        from chimerax.core.errors import UserError
        raise UserError("no chain was specified")
    elif len(chains) > 1:
        from chimerax.core.errors import UserError
        raise UserError("please choose exactly one chain (%d were specified)" %
                        len(chains))
    BlastPDBJob(session, chains[0].characters,
                database, cutoff, matrix, max_hits,
                finish_callback, fail_callback)
blastpdb_desc = CmdDesc(required=[("spec", AtomSpecArg),],
                        keyword=[("database", EnumOf(DBs)),
                                 ("cutoff", FloatArg),
                                 ("matrix", EnumOf(Matrices)),
                                 ("max_hits", IntArg),
                                 ],
                        synopsis="Search PDB/NR using BLAST")

def ccd(session, name):
    from .job import CCDJob
    CCDJob(session, name)
ccd_desc = CmdDesc(required=[("name", StringArg),],
                   synopsis="Get Chemical Component Dictionary template")
