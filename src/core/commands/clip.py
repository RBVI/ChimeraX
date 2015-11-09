# vim: set expandtab shiftwidth=4 softtabstop=4:

def clip(session, enable=None, near=None, far=None):
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
    if near is not None or far is not None:
        enable = True

    v = session.main_view
    clip = v._clip
    nf = clip._near_far		# TODO: Make public
    if enable is None:
        msg = 'Clipping is ' + ('off' if nf is None else 'on')
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
        vd = c.view_direction(view_num)
        co = c.position.origin()
        from ..geometry import inner_product
        dc = inner_product(cofr - co, vd)
        if near is not None:
            n = dc + near
        elif nf is not None:
            n = nf[0]
        else:
            n = dc
        if far is not None:
            f = dc + far
        elif nf is not None:
            f = nf[1]
        else:
            b = v.drawing_bounds()
            if b is None:
                raise UserError("Can't position clip planes with nothing displayed.")
            f = inner_product(b.center() - co, vd) + b.radius()
        if n > f:
            raise UserError("Near clip distance is farther than far clip distance.")
        clip.set_near_far(n,f)
    else:
        clip.no_clipping()
    v.redraw_needed = True

def register_command(session):
    from .cli import CmdDesc, register, BoolArg, FloatArg
    desc = CmdDesc(
        optional=[('enable', BoolArg)],
        keyword=[('near', FloatArg),
                 ('far', FloatArg)],
        synopsis='set clip planes'
    )
    register('clip', desc, clip)
