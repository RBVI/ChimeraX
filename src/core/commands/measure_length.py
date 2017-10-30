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

def measure_length(session, bonds):
    '''
    Compute sum of length of bonds.
    '''
    if len(bonds) > 0:
        a1, a2 = bonds.atoms
        v = a1.scene_coords - a2.scene_coords
        from numpy import sqrt
        length = sqrt((v*v).sum(axis=1)).sum()
    else:
        length = 0
    msg = 'Total length of %d bonds = %.5g' % (len(bonds), length)
    session.logger.status(msg, log = True)

def register_command(session):
    from . import CmdDesc, register, BondsArg
    desc = CmdDesc(
        required = [('bonds', BondsArg)],
        synopsis = 'compute sum of lengths of bonds')
    register('measure length', desc, measure_length, logger=session.logger)
