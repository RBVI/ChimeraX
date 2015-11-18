# vim: set expandtab shiftwidth=4 softtabstop=4:

def clip(session, near=None, far=None, p1=None, p2=None, slab=None, list=None, off=None,
         position=None, axis=None, coordinate_system=None, cap=None):
    '''
    Enable or disable clip planes.

    Parameters
    ----------
    near, far, p1, p2 : float or "off"
       Distance from center of rotation for near and far clip planes that
       remain perpendicular to view and planes p1 and p2 that rotate with models.
       For near/far, positive distances are further away, negative are closer
       than center.
    list : bool
       Report info about the current clip planes.
    off : bool
       Turn off clipping (all planes).
    position : Center
       Plane offsets are relative to this point.  If not give then center
       of bounding box of displayed models is used.
    axis : Axis
       Normal to clip plane for planes p1 and p2.  Not used for near, far planes.
    coordinate_system : Model
       Coordinate system for axis and position, if none then screen coordinates are used.
    cap : bool
      Option for testing display of surface caps.  Will remove this later.
    '''

    v = session.main_view
    planes = v.clip_planes
    if list:
        report_clip_info(v, session.logger)

    have_offset = not (near is None and far is None and p1 is None and p2 is None)
    if not have_offset and off is None:
        if planes.find_plane('p1'):
            return
        else:
            p1 = 0
            have_offset = True

    if off:
        planes.clear()

    if have_offset:
        origin = plane_origin(v, position, coordinate_system)
        cam = v.camera
        from numpy import array, float32
        z = array((0,0,1), float32)
        if near is not None and far is not None:
            adjust_slab('near', near, 'far', far, origin, None, planes, cam, -z)
        elif near is not None:
            adjust_plane('near', near, origin, None, planes, cam, -z)
        elif far is not None:
            adjust_plane('far', -far, origin, None, planes, cam, z)

        normal = None if axis is None else axis.scene_coordinates(coordinate_system, cam)
        if p1 is not None and p2 is not None:
            adjust_slab('p1', p1, 'p2', p2, origin, normal, planes, cam)
        elif p1 is not None:
            adjust_plane('p1', p1, origin, normal, planes, cam)
        elif p2 is not None:
            adjust_plane('p2', p2, origin, normal, planes, cam)

    if cap:
        show_surface_caps(v)

def plane_origin(view, position, coordinate_system):
    if position is None:
        b = view.drawing_bounds()
        if b is None:
            raise UserError("Can't position clip planes relative to center "
                            " of displayed models since nothing is displayed.")
        c0 = b.center()
    else:
        c0 = position.scene_coordinates(coordinate_system)
    return c0

def adjust_plane(name, offset, origin, normal, planes, camera, camera_normal = None):
    if offset == 'off':
        planes.remove_plane(name)
        return

    if camera_normal is not None:
        normal = camera.position.apply_without_translation(camera_normal)

    p = planes.find_plane(name)
    if p is None:
        n = camera.view_direction() if normal is None else normal
        plane_point = origin + offset * n
        from ..graphics import ClipPlane
        p = ClipPlane(name, n, plane_point, camera_normal)
        planes.add_plane(p)
    else:
        n = p.normal if normal is None else normal
        p.plane_point = origin + offset * n
        if normal is not None:
            p.normal = normal

def adjust_slab(name1, offset1, name2, offset2, origin, normal, planes, camera,
                camera_normal = None):
    if offset1 == 'off' or offset2 == 'off':
        adjust_plane(name1, offset1, origin, normal, planes, camera, camera_normal)
        adjust_plane(name2, offset2, origin, normal, planes, camera, camera_normal)
        return

    if normal is None and camera_normal is None:
        # Use an existing plane normal if one exists.
        p1, p2 = planes.find_plane(name1), planes.find_plane(name2)
        if p1 is not None:
            normal = p1.normal
        elif p2 is not None:
            normal = -p2.normal
        else:
            normal = camera.view_direction()

    adjust_plane(name1, offset1, origin, normal, planes, camera, camera_normal)
    n2 = None if normal is None else -normal
    cn2 = None if camera_normal is None else -camera_normal
    adjust_plane(name2, -offset2, origin, n2, planes, camera, cn2)
        
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
    from .cli import CmdDesc, register, FloatArg, NoArg, AxisArg, ModelArg, CenterArg, Or, EnumOf
    offset_arg = Or(EnumOf(['off']), FloatArg)
    desc = CmdDesc(
        optional=[],
        keyword=[('near', offset_arg),
                 ('far', offset_arg),
                 ('p1', offset_arg),
                 ('p2', offset_arg),
                 ('list', NoArg),
                 ('off', NoArg),
                 ('position', CenterArg),
                 ('axis', AxisArg),
                 ('coordinate_system', ModelArg),
                 ('cap', NoArg)],
        synopsis='set clip planes'
    )
    register('clip', desc, clip)
