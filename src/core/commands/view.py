# vi: set expandtab shiftwidth=4 softtabstop=4:


def view(session, atoms=None, show=None, frames=None,
         name=None, list=False, delete=None, orient=False):
    '''
    Move camera so the displayed models fill the graphics window.

    Parameters
    ----------
    atoms : Atoms
      Move camera so the bounding box of specified atoms fills the window.
    show : string
      Restore the saved camera view having this name.
    frames : int
      Interpolate to the desired view over the specified number of frames.
      Only works when showing a named view.
    name : string
      Name the current camera view so it can be shown later with the "show" option.
    list : no value
      Print the named camera views in the log.  The names are links and clicking
      them show the corresponding view.
    delete : string
      Delete the name of a saved view.  Delete "all" deletes all named views.
    orient : no value
      Specifying the orient keyword moves the camera view point to
      look down the scene z axis with the x-axis horizontal and y-axis
      vertical.
    '''
    v = session.main_view
    if orient:
        v.initial_camera_view()
    if atoms is None:
        if name is None and show is None and list is None and delete is None:
            v.view_all()
    elif len(atoms) == 0:
        from ..errors import UserError
        raise UserError('No atoms specified.')
    else:
        from .. import geometry
        b = geometry.sphere_bounds(atoms.scene_coords, atoms.radii)
        v.view_all(b)
    if name is not None:
        save_view(name, session)
    if show is not None:
        show_view(show, frames, session)
    if list:
        list_views(session)
    if delete is not None:
        delete_view(delete, session)

def save_view(name, session):
    nv = _named_views(session)
    v = session.main_view
    nv[name] = _CameraView(v.camera, v.center_of_rotation)

def delete_view(name, session):
    nv = _named_views(session)
    if name == 'all':
        nv.clear()
    elif name in nv:
        del nv[name]

def show_view(name, frames, session):
    nv = _named_views(session)
    if name in nv:
        v = session.main_view
        if frames is None:
            nv[name].set_camera(v.camera)
        else:
            v1 = _CameraView(v.camera, v.center_of_rotation)
            v2 = nv[name]
            _InterpolateViews(v1, v2, frames, session)
    else:
        from ..errors import UserError
        raise UserError('Unknown view "%s"' % name)

def list_views(session):
    nv = _named_views(session)
    names = ['<a href="ch2cmd:view show %s">%s</a>' % (name,name) for name in sorted(nv.keys())]
    msg = 'Named views: ' + ', '.join(names)
    session.logger.info(msg, is_html = True)

def _named_views(session):
    if not hasattr(session, '_named_views'):
        session._named_views = {}
    return session._named_views

class _CameraView:
    camera_attributes = ('position', 'field_of_view', 'field_width',
                         'eye_separation_scene', 'eye_separation_pixels')
    def __init__(self, camera, look_at):
        for attr in self.camera_attributes:
            if hasattr(camera, attr):
                setattr(self, attr, getattr(camera, attr))

        # Scene point which is focus of attention used when
        # interpolating between two views so that the focus
        # of attention stays steady as camera moves and rotates.
        self.look_at = look_at

    def set_camera(self, camera):
        for attr in self.camera_attributes:
            if hasattr(self, attr):
                setattr(camera, attr, getattr(self, attr))
        camera.redraw_needed = True

class _InterpolateViews:
    def __init__(self, v1, v2, frames, session):
        self.view1 = v1
        self.view2 = v2
        self.frames = frames
        from . import motion
        motion.CallForNFrames(self.frame_cb, frames, session)

    def frame_cb(self, session, frame):
        v1, v2 = self.view1, self.view2
        c = session.main_view.camera
        if frame == self.frames:
            v2.set_camera(c)
        else:
            f = frame / self.frames
            interpolate_views(v1, v2, f, c)

def interpolate_views(v1, v2, f, camera):
    from ..geometry import interpolate_rotation, interpolate_points
    r = interpolate_rotation(v1.position, v2.position, f)
    la = interpolate_points(v1.look_at, v2.look_at, f)
    # Look-at points in camera coordinates
    cl1 = v1.position.inverse() * v1.look_at
    cl2 = v2.position.inverse() * v2.look_at
    cla = interpolate_points(cl1, cl2, f)
    # Make camera translation so that camera coordinate look-at point
    # maps to scene coordinate look-at point r*cla + t = la.
    from ..geometry import translation
    t = translation(la - r*cla)
    camera.position = t * r

    # Interpolate field of view
    if hasattr(v1, 'field_of_view') and hasattr(v2, 'field_of_view'):
        camera.field_of_view = (1-f)*v1.field_of_view + f*v1.field_of_view
    elif hasattr(v1, 'field_width') and hasattr(v2, 'field_width_view'):
        camera.field_width_view = (1-f)*v1.field_width + f*v1.field_width

    camera.redraw_needed = True
        
def register_command(session):
    from . import CmdDesc, register, AtomsArg, NoArg, EmptyArg, StringArg, PositiveIntArg, Or
    desc = CmdDesc(
        optional=[('atoms', Or(AtomsArg, EmptyArg)),
                  ('show', StringArg),
                  ('frames', PositiveIntArg)],
        keyword=[('name', StringArg),
                 ('list', NoArg),
                 ('delete', StringArg),
                 ('orient', NoArg)],
        synopsis='reset view so everything is visible in window')
    register('view', desc, view)
