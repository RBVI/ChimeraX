# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.commands import Axis

def roll(session, axis=Axis((0,1,0)), angle=1, frames='forever',
         rock=None, wobble=None, wobble_aspect=0.3,
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
    wobble : integer
       Like rock only move in a figure 8 pattern.
    wobble_aspect : float
       Ratio of wobble angle amplitude to rocking angle amplitude.  Default 0.3.
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
    turn(session, axis=axis, angle=angle, frames=frames,
         rock=rock, wobble=wobble, wobble_aspect=wobble_aspect, center=center,
         coordinate_system=coordinate_system, models=models, atoms=atoms)


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    from chimerax.core.commands import CenterArg, CoordSysArg, TopModelsArg, Or, EnumOf
    from chimerax.atomic import AtomsArg
    from .turn import FramesArg
    desc = CmdDesc(
        optional= [('axis', AxisArg),
                   ('angle', FloatArg),
                   ('frames', FramesArg)],
        keyword = [('center', CenterArg),
                   ('coordinate_system', CoordSysArg),
                   ('rock', PositiveIntArg),
                   ('wobble', PositiveIntArg),
                   ('wobble_aspect', FloatArg),
                   ('models', TopModelsArg),
                   ('atoms', AtomsArg)],
        synopsis='rotate models continuously'
    )
    register('roll', desc, roll, logger=logger)
