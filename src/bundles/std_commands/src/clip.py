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

def clip(session, near=None, far=None, front=None, back=None, slab=None,
         position=None, axis=None, coordinate_system=None, cap=None):
    '''
    Enable or disable clip planes.

    Parameters
    ----------
    near, far, front, back : float or "off"
       Distance to move near, far, front or back clip planes.
       Near and far clip planes remain perpendicular to the view direction.
       Front and back planes rotate with models.  If a plane is not currently
       enabled then the offset value is from the center of rotation.
       Positive distances are further away, negative are closer.
    position : Center
       Plane offsets are relative to this point.  If not give then offsets are
       relative to current plane positions.  If plane is not enabled then offset
       is relative to center of bounding box of displayed models.
    axis : Axis
       Normal to clip plane for planes front and back.  Not used for near and far planes.
    coordinate_system : Place
       Coordinate system for axis and position, if none then screen coordinates are used.
    '''

    v = session.main_view
    planes = v.clip_planes

    have_offset = not (near is None and far is None and front is None and back is None)
    if not have_offset:
        if planes.find_plane('front'):
            return
        else:
            front = 0
            have_offset = True

    if have_offset:
        pos = None
        if position is not None:
            pos = position.scene_coordinates(coordinate_system)
        from numpy import array, float32
        z = array((0,0,1), float32)
        if near is not None and far is not None:
            adjust_slab('near', near, 'far', far, pos, None, planes, v, -z)
        elif near is not None:
            adjust_plane('near', near, pos, None, planes, v, -z)
        elif far == 'off':
            planes.remove_plane('far')
        elif far is not None:
            adjust_plane('far', -far, pos, None, planes, v, z)

        normal = None
        if axis is not None:
            normal = axis.scene_coordinates(coordinate_system, v.camera)
        if front is not None and back is not None:
            adjust_slab('front', front, 'back', back, pos, normal, planes, v)
        elif front is not None:
            adjust_plane('front', front, pos, normal, planes, v)
        elif back == 'off':
            planes.remove_plane('back')
        elif back is not None:
            adjust_plane('back', -back, pos, normal, planes, v)

        warn_on_zero_spacing(session, near, far, front, back)

        from chimerax import surface
        surface.update_clip_caps(session.main_view)

def clip_off(session):
    '''
    Turn off all clip planes.
    '''
    v = session.main_view
    planes = v.clip_planes
    planes.clear()

def clip_list(session):
    '''
    List active clip planes.
    '''
    report_clip_info(session.main_view, session.logger)

def clip_model(session, models, clipping = None):
    '''
    Allow disabling clipping for specific models.

    Parameters
    ----------
    models : list of Models
    clipping : bool
      Whether models will be clipped by scene clip planes.
      This setting does not effect clipping by near and far camera planes.
    '''
    if clipping is None:
        lines = ['Model #%s clipping %s' % (m.id_string, m.allow_clipping)
                 for m in models]
        session.logger.info('\n'.join(lines))
    else:
        for m in models:
            m.allow_clipping = clipping
        for d in _non_model_descendants(models):
            d.allow_clipping = clipping

def _non_model_descendants(models):
    dlist = []
    from chimerax.core.models import Model
    for m in models:
        for d in m.child_drawings():
            if not isinstance(d, Model):
                dlist.append(d)
                dlist.extend(_non_model_descendants([d]))
    return dlist

def plane_origin(view):
    b = view.drawing_bounds()
    if b is None:
        from chimerax.core.errors import UserError
        raise UserError("Can't position clip planes relative to center "
                        " of displayed models since nothing is displayed.")
    c0 = b.center()
    return c0

def adjust_plane(name, offset, origin, normal, planes, view = None, camera_normal = None):
    if offset == 'off':
        planes.remove_plane(name)
        return

    if camera_normal is not None and view is not None:
        cpos = view.camera.position
        normal = cpos.transform_vector(camera_normal)

    p = planes.find_plane(name)
    if p is None:
        n = normal
        if n is None:
            n = view.camera.view_direction()
            if name == 'back':
                n = -n
            face_pair = {'front':'back', 'back':'front'}
            if name in face_pair:
                pp = planes.find_plane(face_pair[name])
                if pp:
                    n = -pp.normal
        if origin is None:
            origin = plane_origin(view)
        plane_point = origin + offset * n
        if camera_normal is None or view is None:
            from chimerax.graphics import SceneClipPlane
            p = SceneClipPlane(name, n, plane_point)
        else:
            camera_plane_point = cpos.inverse() * plane_point
            from chimerax.graphics import CameraClipPlane
            p = CameraClipPlane(name, camera_normal, camera_plane_point, view)
        planes.add_plane(p)
    else:
        n = p.normal if normal is None else normal
        p.plane_point = (p.plane_point if origin is None else origin) + offset * n
        if normal is not None:
            p.normal = normal
        
    return p

def adjust_slab(name1, offset1, name2, offset2, origin, normal, planes, view,
                camera_normal = None):
    if offset1 == 'off' or offset2 == 'off':
        adjust_plane(name1, offset1, origin, normal, planes, view, camera_normal)
        adjust_plane(name2, offset2, origin, normal, planes, view, camera_normal)
        return

    if normal is None and camera_normal is None:
        # Use an existing plane normal if one exists.
        front, back = planes.find_plane(name1), planes.find_plane(name2)
        if front is not None:
            normal = front.normal
        elif back is not None:
            normal = -back.normal
        else:
            normal = view.camera.view_direction()

    adjust_plane(name1, offset1, origin, normal, planes, view, camera_normal)
    n2 = None if normal is None else -normal
    cn2 = None if camera_normal is None else -camera_normal
    adjust_plane(name2, -offset2, origin, n2, planes, view, cn2)
        
def report_clip_info(viewer, log):
    # Report current clip planes.
    planes = viewer.clip_planes.planes()
    if planes:
        b = viewer.drawing_bounds()
        c0 = b.center() if b else (0,0,0)
        pinfo = [_clip_plane_info(p, c0) for p in planes]
        msg = 'Using %d clip planes:\n%s' % (len(planes), '\n'.join(pinfo))
    else:
        msg = 'Clipping is off'
    log.info(msg)
    log.status(msg)

def _clip_plane_info(plane, center):
    offset = -plane.offset(center) if plane.name in ('far', 'back') else plane.offset(center)
    axis = '%.3f,%.3f,%.3f' % tuple(plane.normal)
    point = '%.4g,%.4g,%.4g' % tuple(plane.plane_point)
    info = '%s offset %.5g, axis %s, point %s)' % (plane.name,  offset, axis, point)
    return info

def warn_on_zero_spacing(session, near, far, front, back):
    p = session.main_view.clip_planes
    if near is not None and near != 'off' or far is not None and far != 'off':
        np, fp = p.find_plane('near'), p.find_plane('far')
        if np and fp and np.offset(fp.plane_point) >= 0:
            session.logger.warning('clip far plane is in front of near plane')
    if front is not None and front != 'off' or back is not None and back != 'off':
        np, fp = p.find_plane('front'), p.find_plane('back')
        if np and fp and np.offset(fp.plane_point) >= 0:
            session.logger.warning('clip back plane is in front of front plane')
        
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import CenterArg, CoordSysArg, Or, EnumOf, FloatArg, AxisArg
    from chimerax.core.commands import ModelsArg, BoolArg
    offset_arg = Or(EnumOf(['off']), FloatArg)
    desc = CmdDesc(
        optional=[],
        keyword=[('near', offset_arg),
                 ('far', offset_arg),
                 ('front', offset_arg),
                 ('back', offset_arg),
                 ('position', CenterArg),
                 ('axis', AxisArg),
                 ('coordinate_system', CoordSysArg)],
        synopsis='set clip planes'
    )
    register('clip', desc, clip, logger=logger)
    register('clip off', CmdDesc(synopsis = 'Turn off all clip planes'), clip_off, logger=logger)
    register('clip list', CmdDesc(synopsis = 'List active clip planes'), clip_list, logger=logger)
    model_desc = CmdDesc(
        required = [('models', ModelsArg)],
        optional = [('clipping', BoolArg)],
        synopsis="Turn off clipping for individual models.")
    register('clip model', model_desc, clip_model, logger=logger)
    create_alias('~clip', 'clip off', logger=logger)
