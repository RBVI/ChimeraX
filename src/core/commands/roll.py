# vim: set expandtab shiftwidth=4 softtabstop=4:
from .motion import CallForNFrames


def roll(session, axis=(0, 1, 0), angle=1.5, frames=CallForNFrames.Infinite):
    '''Rotate the scene.  Same as the turn command with infinite frames argument.

    Parameters
    ----------
    axis : 3 comma-separated numbers
       Defines the axis in scene coordinates to rotate about.
    angle : float
       Rotation angle in degrees.
    frames : integer
       Repeat the rotation for N frames.
    '''
    from .turn import turn
    turn(session, axis, angle, frames)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(
        optional=[('axis', cli.AxisArg),
                  ('angle', cli.FloatArg),
                  ('frames', cli.PositiveIntArg)],
        synopsis='rotate models'
    )
    cli.register('roll', desc, roll)
