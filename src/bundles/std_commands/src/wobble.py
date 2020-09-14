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

from chimerax.core.commands import Axis

def wobble(session, axis=Axis((0,1,0)), angle=30, frames="forever", cycle=136,
           aspect=0.3, center=None, coordinate_system=None, models=None, atoms=None):
    '''
    Wobble the scene back and forth.  Same as the turn command with infinite frames argument
    and angle 15 degrees and wobble (number of frames per cycle) equivalent to cycle option
    (default 136), and wobble_aspect equal to the aspect option (default 0.3).
    See turn documentation of other parameters.
    '''
    from .turn import turn
    turn(session, axis=axis, angle=angle, frames=frames, wobble=cycle, wobble_aspect=aspect,
         center=center, coordinate_system=coordinate_system, models=models, atoms=atoms)


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    from chimerax.core.commands import CenterArg, CoordSysArg, TopModelsArg
    from chimerax.atomic import AtomsArg
    from .turn import FramesArg
    desc = CmdDesc(
        optional= [('axis', AxisArg),
                   ('angle', FloatArg),
                   ('frames', FramesArg)],
        keyword = [('center', CenterArg),
                   ('coordinate_system', CoordSysArg),
                   ('cycle', PositiveIntArg),
                   ('aspect', FloatArg),
                   ('models', TopModelsArg),
                   ('atoms', AtomsArg)],
        synopsis='move models in figure 8 motion'
    )
    register('wobble', desc, wobble, logger=logger)
