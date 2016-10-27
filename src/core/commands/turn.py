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

from .cli import Axis

def turn(session, axis=Axis((0,1,0)), angle=90, frames=None, rock=None,
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
       Rotation angle in degrees.
    frames : integer
       Repeat the rotation for N frames, typically used in recording movies.
    rock : integer
       Repeat the rotation reversing the direction every N/2 frames.  The first reversal
       occurs at N/4 frames so that the rocking motion is centered at the current orientation.
       If the frames option is not given the rocking continues indefinitely.
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

    if rock is not None and frames is None:
        frames = -1	# Continue motion indefinitely.

    if frames is not None:
        def turn_step(session, frame):
            rp = _rock_phase(rock, frame)
            if rp != 0:
                turn(session, axis=axis, angle=rp*angle, frames=None, rock=None, center=center,
                     coordinate_system=coordinate_system, models=models)
        from . import motion
        motion.CallForNFrames(turn_step, frames, session)
        return

    v = session.main_view
    c = v.camera
    saxis = axis.scene_coordinates(coordinate_system, c)	# Scene coords
    if center is None:
        ab = axis.base_point()
        c0 = v.center_of_rotation if ab is None else ab
    else:
        c0 = center.scene_coordinates(coordinate_system)
    a = -angle if models is None else angle
    from ..geometry import rotation
    r = rotation(saxis, a, c0)
    if models is not None:
        for m in models:
            m.positions = r * m.positions
    if atoms is not None:
        atoms.scene_coords = r.inverse() * atoms.scene_coords
    if models is None and atoms is None:
        c.position = r * c.position

def _rock_phase(rock, frame):
    if rock is None:
        return 1
    p = (frame + (rock+2)//4) % rock
    h = rock//2
    if p < h:
        return 1
    elif p < 2*h:
        return -1
    return 0

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
        synopsis='rotate models'
    )
    register('turn', desc, turn)

