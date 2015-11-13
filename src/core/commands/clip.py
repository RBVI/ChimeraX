# vim: set expandtab shiftwidth=4 softtabstop=4:

def clip(session, enable=None, near=None, far=None, center=None, tilt=False,
         axis=None, coordinate_system=None, cap=None):
    '''
    Enable or disable clip planes.

    Parameters
    ----------
    enable : bool
       Enable or disable clip planes
    near, far : float
       Distance from center of rotation for near and far clip planes.
       Positive distances are further away, negative are closer than center.
    center : Center
       Near far offsets are relative to this point.  If not give then center
       of rotation is used.
    tilt : bool
       Effect clip planes fixed in the scene instead of perpendicular to view.
    axis : Axis
       Normal to clip plane in tilt mode, in screen coordinates.
    coordinate_system : Model
       Coordinate system for axis, if none then screen coordinates are used.
    cap : bool
      Option for testing display of surface caps.  Will remove this later.
    '''
    if near is not None or far is not None:
        enable = True
    if tilt and enable is None:
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
        c0 = center.scene_coordinates(coordinate_system) if center else v.center_of_rotation
        from ..errors import UserError
        if c0 is None:
            raise UserError("Can't position clip planes with nothing displayed.")
        view_num = 0

        if tilt and axis:
            normal = axis.scene_coordinates(coordinate_system, c)
        elif tilt and clip.enabled:
            normal = clip.normal
        else:
            normal = c.view_direction(view_num)

        if near is not None:
            np = c0 + near*normal
        elif not clip.enabled:
            np = c0
        else:
            np = clip.near_point

        if far is not None:
            fp = c0 + far*normal
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

    if cap:
        show_surface_caps(v)

def register_command(session):
    from .cli import CmdDesc, register, BoolArg, FloatArg, NoArg, AxisArg, ModelArg, CenterArg
    desc = CmdDesc(
        optional=[('enable', BoolArg)],
        keyword=[('near', FloatArg),
                 ('far', FloatArg),
                 ('center', CenterArg),
                 ('tilt', NoArg),
                 ('axis', AxisArg),
                 ('coordinate_system', ModelArg),
                 ('cap', BoolArg)],
        synopsis='set clip planes'
    )
    register('clip', desc, clip)

def show_surface_caps(view):
    drawings = view.drawing.all_drawings()
    show_surface_clip_caps(view.clip, drawings, 'cap')
    show_surface_clip_caps(view.clip_scene, drawings, 'cap scene')

def show_surface_clip_caps(clip, drawings, cap_name, offset = 0.01):
    if clip.enabled:
        normal = clip.normal
        from ..geometry import inner_product
        poffset = inner_product(normal, clip.near_point) + offset
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
    else:
        for d in drawings:
            if d.name == cap_name:
                d.display = False
