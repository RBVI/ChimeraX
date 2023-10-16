# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
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
    if frames is None or frames <= 0:
        _zoom_camera(v, cofr, factor)
    else:
        import math
        ff = math.pow(factor, 1/frames)
        def zoom_cb(session, frame, v=v, p=cofr, f=ff):
            _zoom_camera(v,p,f)
        from chimerax.core.commands import motion
        motion.CallForNFrames(zoom_cb, frames, session)

def _zoom_camera(view, point, factor):
    c = view.camera
    if hasattr(c, 'field_width'):
        # Orthographic camera
        c.field_width /= factor
    else:
        # Perspective camera
        v = c.view_direction()
        p = c.position
        from chimerax.geometry import inner_product, translation
        delta_z = inner_product(p.origin() - point, v)
        zdist = (delta_z*(1-factor)/factor)
        zmove = zdist * v
        c.position = translation(zmove) * p
        _move_near_far_clip_planes(view, zdist)
    c.redraw_needed = True

def _move_near_far_clip_planes(view, zdist):
    for pname in ('near', 'far'):
        plane = view.clip_planes.find_plane(pname)
        if plane:
            print ('moved plane', pname, zdist, view.camera.view_direction())
            plane.plane_point -= zdist * view.camera.view_direction()
    
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, PositiveFloatArg, PositiveIntArg
    desc = CmdDesc(
        optional=[('factor', PositiveFloatArg),
                  ('frames', PositiveIntArg)],
        keyword=[('pixel_size', PositiveFloatArg)],
        synopsis='zoom models'
    )
    register('zoom', desc, zoom, logger=logger)
