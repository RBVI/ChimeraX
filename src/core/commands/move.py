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
        def move_step(session, frame):
            move(session, axis=axis, distance=distance, frames=None,
                 coordinate_system=coordinate_system, models=models)
        from . import motion
        motion.CallForNFrames(move_step, frames, session)
        return

    c = session.main_view.camera
    normalize = (distance is not None)
    saxis = axis.scene_coordinates(coordinate_system, c, normalize)	# Scene coords
    if distance is None:
        distance = 1
    d = -distance if models is None else distance
    from ..geometry import translation
    t = translation(saxis * d)
    if models is not None:
        for m in models:
            m.positions = t * m.positions
    if atoms is not None:
        atoms.scene_coords = t.inverse() * atoms.scene_coords
    if models is None and atoms is None:
        c.position = t * c.position

def register_command(session):
    from .cli import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    from .cli import CoordSysArg, TopModelsArg, AtomsArg
    desc = CmdDesc(
        required = [('axis', AxisArg)],
        optional = [('distance', FloatArg),
                    ('frames', PositiveIntArg)],
        keyword = [('coordinate_system', CoordSysArg),
                   ('models', TopModelsArg),
                   ('atoms', AtomsArg)],
        synopsis='move camera, models, or atoms'
    )
    register('move', desc, move, logger=session.logger)
