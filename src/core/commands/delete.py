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

def delete(session, atoms):
    '''Delete atoms.

    Parameters
    ----------
    atoms : Atoms collection
        Delete these atoms.  If all atoms of a model are closed then the model is closed.
    '''
    atoms.delete()


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(required=[('atoms', cli.AtomsArg)],
                       synopsis='delete atoms')
    cli.register('delete', desc, delete)
