# vim: set expandtab shiftwidth=4 softtabstop=4:

def clip(session, enable=None, near=None, far=None, tilt=None, face_on=None):
    '''
    Enable or disable clip planes.

    Parameters
    ----------
    enable : bool
       Enable or disable clip planes
    near, far : float
       Distance from center of rotation for near and far clip planes.
       Positive distances are further away, negative are closer than center.
    '''
    if near is not None or far is not None or tilt is not None or face_on is not None:
        enable = True

    v = session.main_view
    clip = v.clip
    if enable is None:
        coff = clip.near_point is None and clip.far_point is None
        msg = 'Clipping ' + ('off' if coff else ('tilt mode' if clip.tilt else 'face-on'))
        log = session.logger
        log.info(msg)
        log.status(msg)
        return

    if enable:
        c = v.camera
        cofr = v.center_of_rotation
        from ..errors import UserError
        if cofr is None:
            raise UserError("Can't position clip planes with nothing displayed.")
        view_num = 0
        normal = clip.normal if clip.tilt else c.view_direction(view_num)
        if near is not None:
            np = cofr + near*normal
        elif clip.near_point is None:
            np = cofr
        else:
            np = clip.near_point

        if far is not None:
            fp = cofr + far*normal
        elif clip.far_point is None:
            b = v.drawing_bounds()
            if b is None:
                raise UserError("Can't position clip planes with nothing displayed.")
            fp = b.center() + b.radius()*normal
        else:
            fp = clip.far_point

        from ..geometry import inner_product
        if inner_product(np-fp,normal) > 0:
            raise UserError("Near clip plane is beyond far clip plane.")
        clip.near_point = np
        clip.far_point = fp
        if tilt:
            clip.set_tilt(normal)
        elif face_on:
            clip.set_face_on()
    else:
        clip.no_clipping()
    v.redraw_needed = True

def register_command(session):
    from .cli import CmdDesc, register, BoolArg, FloatArg, NoArg
    desc = CmdDesc(
        optional=[('enable', BoolArg)],
        keyword=[('near', FloatArg),
                 ('far', FloatArg),
                 ('tilt', NoArg),
                 ('face_on', NoArg)],
        synopsis='set clip planes'
    )
    register('clip', desc, clip)
