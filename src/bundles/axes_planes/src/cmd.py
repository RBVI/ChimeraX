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

from .settings import defaults

def cmd_define_plane(session, atoms, *, thickness=defaults["plane_thickness"], padding=0.0, color=None,
        radius=None, name="plane"):
    pass


def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, FloatArg, ColorArg, PositiveFloatArg, \
    from chimerax.core.commands import StringArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('thickness', PositiveFloatArg), ('padding', FloatArg), ('color', ColorArg),
            ('radius', PositiveFloatArg), ('name', StringArg)],
        synopsis = 'Create plane'
    )
    register('define plane', desc, cmd_define_plane, logger=logger)
