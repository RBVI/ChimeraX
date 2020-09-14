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

#
# Compute/assign secondary structure using Kabsch and Sander algorithm
#
def compute_ss(session, structures=None, *,
        min_helix_len=3, min_strand_len=3, energy_cutoff=-0.5, report=False):
    from chimerax.atomic import Structure
    from chimerax.dssp import compute_ss
    if structures is None:
        structures = [m for m in session.models.list() if isinstance(m, Structure)]
    elif isinstance(structures, Structure):
        structures = [structures]

    if len(structures) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No structures specified')

    for struct in structures:
        compute_ss(struct._c_pointer.value, energy_cutoff, min_helix_len, min_strand_len, report)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, BoolArg, IntArg
    from chimerax.atomic import StructuresArg

    desc = CmdDesc(
        optional=[('structures', StructuresArg)],
        keyword=[('min_helix_len', IntArg),
                   ('min_strand_len', IntArg),
                   ('energy_cutoff', FloatArg),
                   ('report', BoolArg)],
        synopsis="compute/assign secondary structure using Kabsch & Sander DSSP algorithm"
    )
    register('dssp', desc, compute_ss, logger=logger)
