# vim: set expandtab shiftwidth=4 softtabstop=4:
from .motion import CallForNFrames

from .cli import Axis

def roll(session, axis=Axis((0,1,0)), angle=1, frames=CallForNFrames.Infinite,
         center=None, coordinate_system=None, models=None):
    '''Rotate the scene.  Same as the turn command with infinite frames argument and angle step 1 degree.

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
    from .turn import turn
    turn(session, axis=axis, angle=angle, frames=frames, center=center,
         coordinate_system=coordinate_system, models=models)


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
        synopsis='rotate models continuously'
    )
    register('roll', desc, roll)
