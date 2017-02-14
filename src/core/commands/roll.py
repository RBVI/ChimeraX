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

from .motion import CallForNFrames

from .cli import Axis

def roll(session, axis=Axis((0,1,0)), angle=1, frames=CallForNFrames.Infinite, rock=None,
         center=None, coordinate_system=None, models=None, atoms=None):
    '''Rotate the scene.  Same as the turn command with infinite frames argument and angle step 1 degree.

    Parameters
    ----------
    axis : Axis
       Defines the axis to rotate about.
    angle : float
       Rotation angle in degrees.
    frames : integer
       Repeat the rotation for N frames, typically used in recording movies.
    rock : integer
       Repeat the rotation reversing the direction every N/2 frames.  The first reversal
       occurs at N/4 frames so that the rocking motion is centered at the current orientation.
    center : Center
       Specifies the center of rotation. If not specified, then the current
       center of rotation is used.
    coordinate_system : Place
       The coordinate system for the axis and optional center point.
       If no coordinate system is specified then scene coordinates are used.
    models : list of Models
       Move the coordinate systems of these models.  Camera is not moved.
    atoms : Atoms
       Change the coordinates of these atoms.  Camera is not moved.
    '''
    from .turn import turn
    turn(session, axis=axis, angle=angle, frames=frames, rock=rock, center=center,
         coordinate_system=coordinate_system, models=models, atoms=atoms)


def register_command(session):
    from .cli import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    from .cli import CenterArg, CoordSysArg, TopModelsArg, AtomsArg
    desc = CmdDesc(
        optional= [('axis', AxisArg),
                   ('angle', FloatArg),
                   ('frames', PositiveIntArg)],
        keyword = [('center', CenterArg),
                   ('coordinate_system', CoordSysArg),
                   ('rock', PositiveIntArg),
                   ('models', TopModelsArg),
                   ('atoms', AtomsArg)],
        synopsis='rotate models continuously'
    )
    register('roll', desc, roll, logger=session.logger)
