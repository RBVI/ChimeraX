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

from chimerax.core.graphics.windowsize import window_size

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, PositiveIntArg
    desc = CmdDesc(optional=[('width', PositiveIntArg),
                             ('height', PositiveIntArg)],
                   synopsis='report or set window size')
    register('windowsize', desc, window_size, logger=logger)
