# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def zoom(session, factor=None, frames=None, pixel_size=None):
    '''
    Move the camera toward or away from the center of rotation
    to make the objects appear bigger by a specified factor.

    Parameters
    ----------
    factor : float
       Factor by which to change apparent object size.
    frames : integer
       Perform the specified zoom over N frames.
    pixel_size : float or None
       Zoom so that the pixel size in physical units (Angstroms) is this value.
       For perspective camera modes the pixel size is set at the center of rotation depth.
       If factor is also given then it multiplies pixel size.
    '''
    v = session.main_view
    cofr = v.center_of_rotation
    if pixel_size is not None:
        f = v.pixel_size(cofr) / pixel_size
        factor = f if factor is None else f*factor
    elif factor is None:
        msg = 'Pixel size at center of rotation is %.3g' % v.pixel_size(cofr)
        log = session.logger
        log.status(msg)
        log.info(msg)
        return
    c = v.camera
    if frames is None or frames <= 0:
        zoom_camera(c, cofr, factor)
    else:
        import math
        ff = math.pow(factor, 1/frames)
        def zoom_cb(session, frame, c=c, p=cofr, f=ff):
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
        optional=[('factor', FloatArg),
                  ('frames', PositiveIntArg)],
        keyword=[('pixel_size', FloatArg)],
        synopsis='zoom models'
    )
    register('zoom', desc, zoom, logger=session.logger)
