# vim: set expandtab shiftwidth=4 softtabstop=4:

def zoom(session, factor, frames=None):
    '''
    Move the camera toward or away from the center of rotation
    to make the objects appear bigger by a specified factor.

    Parameters
    ----------
    factor : float
       Factor by which to change apparent object size.
    frames : integer
       Repeat the zoom N times over N frames.
    '''
    v = session.main_view
    cofr = v.center_of_rotation
    c = v.camera
    if frames is None:
        zoom_camera(c, cofr, factor)
    else:
        def zoom_cb(session, frame, c=c, p=cofr, f=factor):
            zoom_camera(c,p,f)
        from . import motion
        motion.CallForNFrames(zoom_cb, frames, session)

def zoom_camera(c, point, factor):
    if hasattr(c, 'field_width'):
        # Orthographic camera
        c.field_width /= factor
    else:
        # Perspective camera
        v = c.view_direction()
        p = c.position
        from ..geometry import inner_product, translation
        delta_z = inner_product(p.origin() - point, v)
        zmove = (delta_z*(1-factor)/factor) * v
        c.position = translation(zmove) * p
    c.redraw_needed = True

def register_command(session):
    from .cli import CmdDesc, register, FloatArg, PositiveIntArg
    desc = CmdDesc(
        required=[('factor', FloatArg)],
        optional=[('frames', PositiveIntArg)],
        synopsis='zoom models'
    )
    register('zoom', desc, zoom)
