# vim: set expandtab shiftwidth=4 softtabstop=4:


def move(session, axis, distance, frames=None):
    '''Shift the scene.  Actually the camera is shifted and the models stay fixed.

    Parameters
    ----------
    axis : 3 comma-separated numbers
       Defines the axis in scene coordinates to shift along.
    distance : float
       Distance to shift in scene units.
    frames : integer
       Repeat the shift for N frames.
    '''
    c = session.main_view.camera
    cv = c.position
    saxis = cv.apply_without_translation(axis)  # Convert axis from camera to scene coordinates
    from ..geometry import translation
    t = translation(saxis * -distance)
    if frames is None:
        c.position = t * cv
    else:
        def translate(session, frame, t=t):
            c.position = t * c.position
        from . import motion
        motion.CallForNFrames(translate, frames, session)


def register_command(session):
    from .cli import CmdDesc, register, AxisArg, FloatArg, PositiveIntArg
    desc = CmdDesc(
        required=[('axis', AxisArg),
                  ('distance', FloatArg)],
        optional=[('frames', PositiveIntArg)],
        synopsis='translate models'
    )
    register('move', desc, move)
