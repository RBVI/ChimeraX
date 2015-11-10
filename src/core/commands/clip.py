# vim: set expandtab shiftwidth=4 softtabstop=4:

def clip(session, enable=None, near=None, far=None, tilt=False):
    '''
    Enable or disable clip planes.

    Parameters
    ----------
    enable : bool
       Enable or disable clip planes
    near, far : float
       Distance from center of rotation for near and far clip planes.
       Positive distances are further away, negative are closer than center.
    tilt : bool
       Effect clip planes fixed in the scene instead of perpendicular to view.
    '''
    if near is not None or far is not None:
        enable = True
    if tilt is not None and enable is None:
        enable = True

    v = session.main_view
    clip = v.clip_scene if tilt else v.clip
    if enable is None:
        msg = 'Clipping is ' + ('on' if clip.enabled else 'off')
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

        if tilt and clip.enabled:
            normal = clip.normal
        else:
            normal = c.view_direction(view_num)

        if near is not None:
            np = cofr + near*normal
        elif not clip.enabled:
            np = cofr
        else:
            np = clip.near_point

        if far is not None:
            fp = cofr + far*normal
        elif not clip.enabled:
            b = v.drawing_bounds()
            if b is None:
                raise UserError("Can't position clip planes with nothing displayed.")
            fp = b.center() + b.radius()*normal
        else:
            fp = clip.far_point

        from ..geometry import inner_product
        if inner_product(np-fp,normal) > 0:
            raise UserError("Near clip plane is beyond far clip plane.")
    else:
        np = fp = normal = None
        if not clip.enabled and v.clip_scene.enabled:
            clip = v.clip_scene

    clip.near_point = np
    clip.far_point = fp
    clip.normal = normal
    clip.enabled = enable
    v.redraw_needed = True

def register_command(session):
    from .cli import CmdDesc, register, BoolArg, FloatArg, NoArg
    desc = CmdDesc(
        optional=[('enable', BoolArg)],
        keyword=[('near', FloatArg),
                 ('far', FloatArg),
                 ('tilt', NoArg)],
        synopsis='set clip planes'
    )
    register('clip', desc, clip)
