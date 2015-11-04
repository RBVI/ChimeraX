# vim: set expandtab shiftwidth=4 softtabstop=4:

from .cli import Axis

def turn(session, axis=Axis((0,1,0)), angle=90, frames=None,
         center=None, coordinate_system=None, models=None):
    '''
    Rotate the scene.  Actually the camera is rotated about the scene center of rotation
    unless the models argument is specified.

    Parameters
    ----------
    axis : Axis
       Defines the axis to rotate about.
    angle : float
       Rotation angle in degrees.
    frames : integer
       Repeat the rotation for N frames, typically used in recording movies.
    center : Center
       Specifies the center of rotation. If not specified, then the current
       center of rotation is used.
    coordinate_system : Model
       The coordinate system for the axis and optional center point.
       If no model coordinate system is specified then scene coordinates are used.
    models : list of Models
       Only these models are moved.  If not specified, then the camera is moved.
    '''

    if frames is not None:
        def turn_step(session, frame):
            turn(session, axis=axis, angle=angle, frames=None, center=center,
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
    if models is None:
        c.position = r * c.position
    else:
        for m in models:
            m.positions = r * m.positions

def register_command(session):
    from .cli import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    from .cli import CenterArg, ModelArg, TopModelsArg
    desc = CmdDesc(
        optional= [('axis', AxisArg),
                   ('angle', FloatArg),
                   ('frames', PositiveIntArg)],
        keyword = [('center', CenterArg),
                   ('coordinate_system', ModelArg),
                   ('models', TopModelsArg)],
        synopsis='rotate models'
    )
    register('turn', desc, turn)
