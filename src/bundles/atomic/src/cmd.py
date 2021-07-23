# vim: set expandtab ts=4 sw=4:

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

def log_chains(session, structures=None):
    if structures is None:
        from chimerax.atomic import AtomicStructure
        structures = [s for s in session.models if isinstance(s, AtomicStructure)]
    for s in structures:
        s._report_chain_descriptions(session)

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, Or, NoneArg
    from .args import AtomicStructuresArg
    chains_desc = CmdDesc(
                        optional = [('structures', Or(AtomicStructuresArg, NoneArg))],
                        synopsis = 'Add structure chains table to the log'
    )
    register('log chains', chains_desc, log_chains, logger=logger)
