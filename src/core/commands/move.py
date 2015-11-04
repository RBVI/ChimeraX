# vim: set expandtab shiftwidth=4 softtabstop=4:


def move(session, axis, distance=None, frames=None, coordinate_system=None, models=None):
    '''Shift the scene.  Actually the camera is shifted and the models stay fixed
    unless the models option is specified.

    Parameters
    ----------
    axis : Axis
       Defines the axis in scene coordinates to shift along.
    distance : float
       Distance to shift in scene units.
    frames : integer
       Repeat the shift for N frames.
    coordinate_system : Model
       The coordinate system for the axis.
       If no model coordinate system is specified then scene coordinates are used.
    models : list of Models
       Only these models are moved.  If not specified, then the camera is moved.
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
    if models is None:
        c.position = t * c.position
    else:
        for m in models:
            m.positions = t * m.positions

def register_command(session):
    from .cli import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    from .cli import ModelArg, TopModelsArg
    desc = CmdDesc(
        required = [('axis', AxisArg)],
        optional = [('distance', FloatArg),
                    ('frames', PositiveIntArg)],
        keyword = [('coordinate_system', ModelArg),
                   ('models', TopModelsArg)],
        synopsis='translate models'
    )
    register('move', desc, move)
