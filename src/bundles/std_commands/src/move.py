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

def move(session, axis, distance=None, frames=None, coordinate_system=None,
         models=None, atoms=None):
    '''Shift the scene.  Actually the camera is shifted and the models stay fixed
    unless the models option is specified in which case the model coordinate systems
    are moved, or if atoms is specifed the atom coordinates are moved.

    Parameters
    ----------
    axis : Axis
       Defines the axis in scene coordinates to shift along.
    distance : float
       Distance to shift in scene units.
    frames : integer
       Repeat the shift for N frames.
    coordinate_system : Place
       The coordinate system for the axis.
       If no coordinate system is specified then scene coordinates are used.
    models : list of Models
       Move the coordinate systems of these models.  Camera is not moved.
    atoms : Atoms
       Change the coordinates of these atoms.  Camera is not moved.
    '''
    if frames is not None:
        def move_step(session, frame, undo=None):
            with session.undo.block():
                d = distance if undo is None else -distance
                move(session, axis=axis, distance=d, frames=None,
                     coordinate_system=coordinate_system, models=models)
        multiframe_motion("move", move_step, frames, session)
        return

    from .view import UndoView
    undo = UndoView("move", session, models, frames=frames)
    with session.undo.block():
        c = session.main_view.camera
        normalize = (distance is not None)
        saxis = axis.scene_coordinates(coordinate_system, c, normalize)	# Scene coords
        if distance is None:
            distance = 1
        d = -distance if models is None else distance
        from chimerax.geometry import translation
        t = translation(saxis * d)
        if models is not None:
            for m in models:
                m.positions = t * m.positions
        if atoms is not None:
            atoms.scene_coords = t.inverse() * atoms.scene_coords
        if models is None and atoms is None:
            c.position = t * c.position
    undo.finish(session, models)
    session.undo.register(undo)


def multiframe_motion(name, func, frames, session):
    from chimerax.core.commands.motion import CallForNFrames
    if frames != CallForNFrames.Infinite:
        session.undo.register(UndoMotion(name, func, frames, session))
    CallForNFrames(func, frames, session)


from chimerax.core.undo import UndoAction
class UndoMotion(UndoAction):

    def __init__(self, name, func, frames, session):
        super().__init__(name, can_redo=True)
        self._func = func
        self._frames = frames
        self._session = session

    def undo(self):
        from chimerax.core.commands.motion import CallForNFrames
        def undo_func(*args, **kw):
            self._func(*args, undo=True, **kw)
        CallForNFrames(undo_func, self._frames, self._session)

    def redo(self):
        from chimerax.core.commands.motion import CallForNFrames
        CallForNFrames(self._func, self._frames, self._session)


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    from chimerax.core.commands import CoordSysArg, TopModelsArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('axis', AxisArg)],
        optional = [('distance', FloatArg),
                    ('frames', PositiveIntArg)],
        keyword = [('coordinate_system', CoordSysArg),
                   ('models', TopModelsArg),
                   ('atoms', AtomsArg)],
        synopsis='move camera, models, or atoms'
    )
    register('move', desc, move, logger=logger)
