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
            from chimerax.core.graphics import SceneClipPlane
            p = SceneClipPlane(name, n, plane_point)
        else:
            camera_plane_point = cpos.inverse() * plane_point
            from chimerax.core.graphics import CameraClipPlane
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
        pinfo = ['%s %.5g' % (p.name,  -p.offset(c0) if p.name in ('far', 'back') else p.offset(c0))
                 for p in planes]
        msg = 'Using %d clip planes: %s' % (len(planes), ', '.join(pinfo))
    else:
        msg = 'Clipping is off'
    log.info(msg)
    log.status(msg)

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
    from chimerax.core.commands import CmdDesc, register, FloatArg, AxisArg
    from chimerax.core.commands import CenterArg, CoordSysArg, Or, EnumOf, create_alias
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
    create_alias('~clip', 'clip off', logger=logger)
