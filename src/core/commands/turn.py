# vim: set expandtab shiftwidth=4 softtabstop=4:


def turn(session, axis=(0, 1, 0), angle=1.5, frames=None):
    '''Rotate the scene.  Actually the camera is rotated about the scene center of rotation.

    Parameters
    ----------
    axis : 3 comma-separated numbers
       Defines the axis in scene coordinates to rotate about.
    angle : float
       Rotation angle in degrees.
    frames : integer
       Repeat the rotation for N frames, typically used in recording movies.
    '''
    v = session.main_view
    c = v.camera
    cv = c.position
    saxis = cv.apply_without_translation(axis)  # Convert axis from camera to scene coordinates
    center = v.center_of_rotation
    from ..geometry import rotation
    r = rotation(saxis, -angle, center)
    if frames is None:
        c.position = r * cv
    else:
        def rotate(session, frame, r=r, c=c):
            c.position = r * c.position
        from . import motion
        motion.CallForNFrames(rotate, frames, session)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(
        required=[('axis', cli.AxisArg),
                  ('angle', cli.FloatArg)],
        optional=[('frames', cli.PositiveIntArg)],
        synopsis='rotate models'
    )
    cli.register('turn', desc, turn)
