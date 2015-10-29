# vim: set expandtab shiftwidth=4 softtabstop=4:

def zoom(session, factor, frames=None):
    '''
    Zoom the scene by changing camera field of view.
    With 360 degree camera modes this has no effect.

    Parameters
    ----------
    factor : float
       Factor by which to change the width of the field of view.
    frames : integer
       Repeat the zoom N times over N frames.
    '''
    c = session.main_view.camera
    if frames is None:
        zoom_camera(c, factor)
    else:
        def zoom_cb(session, frame, c=c, f=factor):
            zoom_camera(c,f)
        from . import motion
        motion.CallForNFrames(zoom_cb, frames, session)

def zoom_camera(c, factor):
    if hasattr(c, 'field_of_view'):
        from math import tan, atan, radians, degrees
        fov = degrees(2*atan(tan(0.5*radians(c.field_of_view))/factor))
        c.field_of_view = fov
    elif hasattr(c, 'field_width'):
        c.field_width /= factor
    c.redraw_needed = True

def register_command(session):
    from .cli import CmdDesc, register, FloatArg, PositiveIntArg
    desc = CmdDesc(
        required=[('factor', FloatArg)],
        optional=[('frames', PositiveIntArg)],
        synopsis='zoom models'
    )
    register('zoom', desc, zoom)
