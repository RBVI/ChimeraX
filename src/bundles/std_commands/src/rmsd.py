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


def rmsd(session, atoms, to=None):
    """Compute RMSD between 'atoms' and 'with_atoms"""
    from chimerax.core.errors import UserError
    atoms1, atoms2 = atoms, to
    if len(atoms1) != len(atoms2):
        raise UserError("Number of atoms from first atom spec (%d) differs from number in second (%d)"
            % (len(atoms1), len(atoms2)))
    if len(atoms1) == 0:
        raise UserError("Given atom specs don't contain any atoms")

    diff = atoms1.scene_coords - atoms2.scene_coords
    from math import sqrt
    import numpy
    val = sqrt(numpy.sum(diff * diff)/len(atoms1))
    session.logger.info("RMSD between %d atom pairs is %.3f" % (len(atoms1), val))
    return val

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.atomic import OrderedAtomsArg
    desc = CmdDesc(required = [('atoms', OrderedAtomsArg)],
                   keyword = [('to', OrderedAtomsArg),],
                   required_arguments = ['to'],
                   synopsis = 'Compute RMSD between two sets of atoms')
    register('rmsd', desc, rmsd, logger=logger)
