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

def turn(session, axis=Axis((0,1,0)), angle=90, frames=None,
         rock=None, wobble=None, wobble_aspect=0.3,
         center=None, coordinate_system=None, models=None, atoms=None):
    '''
    Rotate the scene.  Actually the camera is rotated about the scene center of rotation
    unless the models argument is specified in which case the model coordinate systems
    are moved, or if atoms is specifed the atom coordinates are moved.

    Parameters
    ----------
    axis : Axis
       Defines the axis to rotate about.
    angle : float
       Rotation angle about axis in degrees.
    frames : "forever" or an integer
       Repeat the rotation for N frames, typically used in recording movies.
    rock : integer
       Rotate +/- angle/2 degrees repeating, one cycle every specified number of frames.
       The rocking steps are small at the ends of the rock, using sine modulation.
       If the frames option is not given the rocking continues until the stop command
       is run.
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

    if ((rock or wobble) and frames is None) or frames == 'forever':
        frames = -1	# Continue motion indefinitely.

    v = session.main_view
    camera = v.camera
    with session.undo.block():
        saxis = axis.scene_coordinates(coordinate_system, camera)	# Scene coords
        if center is None:
            ab = axis.base_point()
            center = v.center_of_rotation if ab is None else ab
        else:
            center = center.scene_coordinates(coordinate_system)

    turner = Turner(axis=saxis, center=center, angle=angle, rock=rock,
                    wobble=wobble, wobble_aspect=wobble_aspect,
                    camera=camera, models=models, atoms=atoms)

    if frames is not None:
        from .move import multiframe_motion
        multiframe_motion("turn", turner.turn_step, frames, session)
    else:
        from .view import UndoView
        undo = UndoView("move", session, models, frames=frames)
        with session.undo.block():
            turner.turn()
        undo.finish(session, models)
        session.undo.register(undo)

class Turner:
    def __init__(self, axis=(0,1,0), center=(0,0,0), angle=90,
                 rock=None, wobble=None, wobble_aspect=0.3, wobble_axis=None,
                 camera=None, models=None, atoms=None):
        self._axis = axis
        self._center = center
        self._full_angle = angle
        self._rock = rock
        self._wobble = wobble
        if wobble is not None and wobble_axis is None:
            from chimerax.geometry import cross_product, normalize_vector
            wobble_axis = normalize_vector(cross_product(camera.view_direction(), axis))
        self._wobble_axis = wobble_axis
        self._wobble_aspect = wobble_aspect
        self._wobble_axis = wobble_axis
        self._camera = camera
        self._models = models
        self._atoms = atoms
        
    def turn(self):
        r = self._rotation(self._full_angle)
        self._move(r)
        
    def _rotation(self, angle):
        from chimerax.geometry import rotation
        r = rotation(self._axis, angle, self._center)
        return r

    def _move(self, rotation, invert=False):
        r = rotation.inverse() if invert else rotation
        models = self._models
        atoms = self._atoms
        move_camera = (models is None and atoms is None)
        if move_camera:
            camera = self._camera
            camera.position = r.inverse() * camera.position
        else:
            if models is not None:
                for m in models:
                    if not m.deleted:
                        m.positions = r * m.positions
            if atoms is not None:
                atoms.scene_coords = r * atoms.scene_coords

    def turn_step(self, session, frame, undo=None):
        with session.undo.block():
            if self._rock:
                r = self._rock_motion(frame)
            elif self._wobble:
                r = self._wobble_motion(frame)
            else:
                r = self._rotation(self._full_angle)
            self._move(r, invert=undo)

    def _rock_motion(self, frame):
        n = self._rock
        from math import pi, sin
        f = sin(2*pi*((frame+1)%n)/n) - sin(2*pi*(frame%n)/n)
        r = self._rotation(f * self._full_angle/2)
        return r

    def _wobble_motion(self, frame):
        n = self._wobble
        w0 = self._wobble_position((frame%n)/n)
        w1 = self._wobble_position(((frame+1)%n)/n)
        move_camera = (self._models is None and self._atoms is None)
        r = w0 * w1.inverse() if move_camera else w1 * w0.inverse()
        return r

    def _wobble_position(self, f):
        amax = 0.5 * self._full_angle
        from math import pi, sin
        a = sin(2*pi*f) * amax
        wa = sin(4*pi*f) * amax * self._wobble_aspect
        from chimerax.geometry import rotation
        r = rotation(self._axis, a, self._center)
        rw = rotation(self._wobble_axis, wa, self._center)
        return rw*r

from chimerax.core.commands import Or, EnumOf, PositiveIntArg
FramesArg = Or(EnumOf(['forever']), PositiveIntArg)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    from chimerax.core.commands import CenterArg, CoordSysArg, TopModelsArg, Or, EnumOf
    from chimerax.atomic import AtomsArg
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
        synopsis='rotate models'
    )
    register('turn', desc, turn, logger=logger)

