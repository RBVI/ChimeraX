# vim: set expandtab shiftwidth=4 softtabstop=4:

def clip(session, near=None, far=None, front=None, back=None, slab=None, list=None, off=None,
         position=None, axis=None, coordinate_system=None, cap=None):
    '''
    Enable or disable clip planes.

    Parameters
    ----------
    near, far, front, back : float or "off"
       Distance to move near, far, front or back clip planes.
       Near and far clip planes emain perpendicular to the view direction.
       Front and back planes rotate with models.  If a plane is not currently
       enabled then the offset value is from the center of rotation.
       Positive distances are further away, negative are closer.
    list : bool
       Report info about the current clip planes.
    off : bool
       Turn off clipping (all planes).
    position : Center
       Plane offsets are relative to this point.  If not give then offsets are
       relative to current plane positions.  If plane is not enabled then offset
       is relative to center of bounding box of displayed models.
    axis : Axis
       Normal to clip plane for planes front and back.  Not used for near and far planes.
    coordinate_system : Model
       Coordinate system for axis and position, if none then screen coordinates are used.
    '''

    v = session.main_view
    planes = v.clip_planes
    if list:
        report_clip_info(v, session.logger)
        return

    have_offset = not (near is None and far is None and front is None and back is None)
    if not have_offset and off is None:
        if planes.find_plane('front'):
            return
        else:
            front = 0
            have_offset = True

    if off:
        planes.clear()

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
        elif far is not None:
            adjust_plane('far', -far, pos, None, planes, v, z)

        normal = None
        if axis is not None:
            normal = axis.scene_coordinates(coordinate_system, v.camera)
        if front is not None and back is not None:
            adjust_slab('front', front, 'back', back, pos, normal, planes, v)
        elif front is not None:
            adjust_plane('front', front, pos, normal, planes, v)
        elif back is not None:
            adjust_plane('back', -back, pos, normal, planes, v)

def plane_origin(view):
    b = view.drawing_bounds()
    if b is None:
        from ..errors import UserError
        raise UserError("Can't position clip planes relative to center "
                        " of displayed models since nothing is displayed.")
    c0 = b.center()
    return c0

def adjust_plane(name, offset, origin, normal, planes, view = None, camera_normal = None):
    if offset == 'off':
        planes.remove_plane(name)
        return

    if camera_normal is not None:
        normal = camera.position.apply_without_translation(camera_normal)

    p = planes.find_plane(name)
    if p is None:
        n = normal
        if n is None:
            n = view.camera.view_direction()
            face_pair = {'front':'back', 'back':'front'}
            if name in face_pair:
                pp = planes.find_plane(face_pair[name])
                if pp:
                    n = -pp.normal
        if origin is None:
            origin = plane_origin(view)
        plane_point = origin + offset * n
        from ..graphics import ClipPlane
        p = ClipPlane(name, n, plane_point, camera_normal)
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
        pinfo = ['%s %.5g' % (p.name,  p.offset(c0)) for p in planes]
        msg = 'Using %d clip planes: %s' % (len(planes), ', '.join(pinfo))
    else:
        msg = 'Clipping is off'
    log.info(msg)
    log.status(msg)

def register_command(session):
    from .cli import CmdDesc, register, FloatArg, NoArg, AxisArg, ModelArg, CenterArg, Or, EnumOf, create_alias
    offset_arg = Or(EnumOf(['off']), FloatArg)
    desc = CmdDesc(
        optional=[],
        keyword=[('near', offset_arg),
                 ('far', offset_arg),
                 ('front', offset_arg),
                 ('back', offset_arg),
                 ('list', NoArg),
                 ('off', NoArg),
                 ('position', CenterArg),
                 ('axis', AxisArg),
                 ('coordinate_system', ModelArg)],
        synopsis='set clip planes'
    )
    register('clip', desc, clip)
    create_alias('~clip', 'clip off')
