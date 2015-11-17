# vim: set expandtab shiftwidth=4 softtabstop=4:

def clip(session, near=None, far=None, p1=None, p2=None, off=None,
         center=None, axis=None, coordinate_system=None, cap=None):
    '''
    Enable or disable clip planes.

    Parameters
    ----------
    near, far, p1, p2 : float
       Distance from center of rotation for near and far clip planes that
       remain perpendicular to view and planes p1 and p2 that rotate with models.
       For near/far, positive distances are further away, negative are closer
       than center.
    off : no value
       Turn off clipping (all planes).
    center : Center
       Near far offsets are relative to this point.  If not give then center
       of rotation is used.
    axis : Axis
       Normal to clip plane for planes p1 and p2.  Not used for near, far planes.
    coordinate_system : Model
       Coordinate system for axis, if none then screen coordinates are used.
    cap : bool
      Option for testing display of surface caps.  Will remove this later.
    '''

    v = session.main_view
    have_offset = not (near is None and far is None and p1 is None and p2 is None)
    if not have_offset and off is None:
        if v.clip_planes:
            report_clip_info(v, session.logger)
            return
        else:
            p1 = 0
            have_offset = True

    if off:
        v.clip_planes = []

    if have_offset:
        origin = plane_origin(v, center, coordinate_system)
        cam = v.camera
        adjust_near_far_planes(near, far, origin, v.clip_planes, cam)
        normal = None if axis is None else axis.scene_coordinates(coordinate_system, cam)
        adjust_scene_planes(p1, p2, origin, normal, v.clip_planes, cam)

    v.redraw_needed = True

    if cap:
        show_surface_caps(v)

def plane_origin(view, center, origin):
    if center is None:
        b = view.drawing_bounds()
        if b is None:
            raise UserError("Can't position clip planes relative to center "
                            " of displayed models since nothing is displayed.")
        c0 = b.center()
    else:
        c0 = center.scene_coordinates(coordinate_system)
    return c0

def adjust_near_far_planes(near, far, origin, planes, camera):
    for name, offset, axis in (('near', near, (0,0,-1)), ('far', far, (0,0,1))):
        if offset is None:
            continue
        normal = camera.position.apply_without_translation(axis)
        plane_point = origin - offset * normal
        p = find_plane(planes, name)
        if p is None:
            from ..graphics import ClipPlane
            p = ClipPlane(name, normal, plane_point, axis)
            planes.append(p)
        else:
            p.plane_point = plane_point

def adjust_scene_planes(p1, p2, origin, normal, planes, camera):
    for name, offset in (('p1', p1), ('p2', p2)):
        if offset is None:
            continue
        p = find_plane(planes, name)
        if p is None:
            n = camera.view_direction() if normal is None else normal
            plane_point = origin - offset * n
            from ..graphics import ClipPlane
            p = ClipPlane(name, n, plane_point)
            p.name = name
            planes.append(p)
        else:
            n = p.normal if normal is None else normal
            p.plane_point = origin + offset * n

def find_plane(planes, name):
    np = [p for p in planes if p.name == name]
    return np[0] if len(np) >= 1 else None
        
def report_clip_info(viewer, log):
    # Report current clip planes.
    planes = viewer.clip_planes
    if planes:
        b = viewer.drawing_bounds()
        c0 = b.center() if b else (0,0,0)
        pinfo = ['%s %.5g' % (p.name,  p.offset(c0)) for p in planes]
        msg = 'Using %d clip planes: %s' % (len(planes), ', '.join(pinfo))
    else:
        msg = 'Clipping is off'
    log.info(msg)
    log.status(msg)

def show_surface_caps(view):
    drawings = view.drawing.all_drawings()
    show_surface_clip_caps(view.clip_planes, drawings)

def show_surface_clip_caps(clip_planes, drawings, offset = 0.01):
    for p in clip_planes:
        normal = p.normal
        from ..geometry import inner_product
        poffset = inner_product(normal, p.plane_point) + offset
        cap_name = 'cap ' + p.name
        for d in drawings:
            if d.triangles is not None and not d.name.startswith('cap'):
                from ..surface import compute_cap
                cvarray, ctarray = compute_cap(normal, poffset, d.vertices, d.triangles)
                mcap = [cm for cm in d.child_drawings() if cm.name == cap_name]
#                print ('showing cap for', d.name, 'triangles', len(ctarray), 'have', len(mcap),
#                       'normal', normal, 'offset', poffset)
                cm = mcap[0] if mcap else d.new_drawing(cap_name)
                cm.vertices = cvarray
                cm.triangles = ctarray
                n = cvarray.copy()
                n[:] = normal
                cm.normals = n
                cm.display = True

    cap_names = set('cap ' + p.name for p in clip_planes)
    for d in drawings:
        if d.name.startswith('cap') and d.name not in cap_names:
            d.display = False

def register_command(session):
    from .cli import CmdDesc, register, FloatArg, NoArg, AxisArg, ModelArg, CenterArg
    desc = CmdDesc(
        optional=[],
        keyword=[('near', FloatArg),
                 ('far', FloatArg),
                 ('p1', FloatArg),
                 ('p2', FloatArg),
                 ('off', NoArg),
                 ('center', CenterArg),
                 ('axis', AxisArg),
                 ('coordinate_system', ModelArg),
                 ('cap', NoArg)],
        synopsis='set clip planes'
    )
    register('clip', desc, clip)
