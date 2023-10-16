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

from chimerax.core.commands.motion import CallForNFrames

from chimerax.core.commands import Axis

def rock(session, axis=Axis((0,1,0)), angle=30, frames=CallForNFrames.Infinite, cycle=136,
         center=None, coordinate_system=None, models=None, atoms=None):
    '''
    Rock the scene back and forth.  Same as the turn command with infinite frames argument
    and angle 15 degrees and rock (number of frames per cycle) equivalent to cycle option
    (default 136).

    Parameters
    ----------
    axis : Axis
       Defines the axis to rotate about.
    angle : float
       Full range rotation angle in degrees.
    frames : integer
       Repeat the rotation for N frames, typically used in recording movies.
    cycle : integer
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
    turn(session, axis=axis, angle=angle, frames=frames, rock=cycle, center=center,
         coordinate_system=coordinate_system, models=models, atoms=atoms)


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
                   ('models', TopModelsArg),
                   ('atoms', AtomsArg)],
        synopsis='rock models back and forth'
    )
    register('rock', desc, rock, logger=logger)
