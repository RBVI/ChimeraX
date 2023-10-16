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

from chimerax.core.commands import Annotation, AnnotationError
from chimerax.core.errors import UserError


def view(session, objects=None, frames=None, clip=True, cofr=True,
         orient=False, zalign=None, in_front_of=None, pad=0.05, need_undo=True):
    '''
    Move camera so the displayed models fill the graphics window.
    Also camera and model positions can be saved and restored.
    Adjust camera to view all models if objects is None.

    Parameters
    ----------
    objects : Objects
      Move camera so the bounding box of specified objects fills the window.
    frames : int
      Interpolate to the desired view over the specified number of frames.
    clip : bool
      Turn on clip planes in front and behind objects.
    cofr : bool
      Set center of rotation to center of objects.
    orient : bool
      Specifying the orient keyword moves the camera view point to
      look down the scene z axis with the x-axis horizontal and y-axis
      vertical.
    zalign : Objects
      Rotate view point so two specified atoms are aligned along the view
      axis with the first atom in front.  Exactly two atoms must be specified.
      Alternatively an AxisModel or a PlaneModel can be specified, in which
      case the axis or plane normal will be aligned.
    in_front_of : Atoms
      Used only with zalign option.  If specified the geometric center of the
      zalign atoms and the geometric center of the in_front_of atoms are aligned
      perpendicular to the screen.
    pad : float
      When making objects fit in window use a window size reduced by this fraction.
      Default value is 0.05.  Pad is ignored when restoring named views.
    need_undo : bool
      Whether to create undo action
    '''

    if need_undo:
        models = session.models.list()
        undo = UndoView("view", session, models)

    with session.undo.block():
        v = session.main_view
        if orient:
            v.initial_camera_view(set_pivot = cofr)
        if zalign:
            _z_align_view(session.main_view.camera, zalign, in_front_of)
        if objects is None:
            v.view_all(pad = pad)
            if cofr:
                v.center_of_rotation_method = 'front center'
            cp = v.clip_planes
            cp.remove_plane('near')
            cp.remove_plane('far')
        elif isinstance(objects, NamedView):
            show_view(session, objects, frames)
        else:
            view_objects(objects, v, clip, cofr, pad)

    if need_undo:
        undo.finish(session, models)
        session.undo.register(undo)


def view_objects(objects, v, clip, cofr, pad):
    if objects.empty():
        raise UserError('No objects specified.')
    disp = objects.displayed()
    # Use atoms but not whole molecular surfaces. Ticket #5663
    disp = _remove_molecular_surfaces(disp)
    b = disp.bounds()
    if b is None:
        raise UserError('No displayed objects specified.')
    v.view_all(b, pad = pad)
    c, r = b.center(), b.radius()

    cp = v.clip_planes
    if clip:
        vd = v.camera.view_direction()
        cp.set_clip_position('near', c - r * vd, v)
        cp.set_clip_position('far', c + r * vd, v)
    else:
        cp.remove_plane('near')
        cp.remove_plane('far')

    if cofr:
        v.center_of_rotation_method = 'center of view'
        if not clip:
            v.set_rotation_depth(c)

def _remove_molecular_surfaces(objects):
    '''
    Remove molecular surface models from objects if the objects include atoms
    for those surfaces.
    '''
    from chimerax.atomic import MolecularSurface
    msurfs = set([s for s in objects.models
                  if isinstance(s, MolecularSurface) and s.atoms.intersects(objects.atoms)])
    misurfs = set([s for s in objects.model_instances.keys()
                   if isinstance(s, MolecularSurface) and s.atoms.intersects(objects.atoms)])
    if len(msurfs) == 0 and len(misurfs) == 0:
        return objects
    from chimerax.core.objects import Objects
    o = Objects(atoms = objects.atoms, bonds = objects.bonds,
                pseudobonds = objects.pseudobonds,
                models = [m for m in objects.models if m not in msurfs])
    for m, minst in objects.model_instances.items():
        if m not in misurfs:
            o.add_model_instances(m, minst)
    return o
    
def _z_align_view(camera, objects, in_front_of = None):
    '''
    Rotate camera so axis/plane/two atoms is/are along view direction (if atoms, first atom in front).
    Rotation is about midpoint between the two atoms, or center of axis/plane.
    '''
    align_pts = None
    from chimerax.dist_monitor import ComplexMeasurable
    for m in objects.models:
        if isinstance(m, ComplexMeasurable):
            try:
                m_align_pts = m.alignment_points
            except NotImplemented:
                continue
            if align_pts is None:
                align_pts = m_align_pts
            else:
                raise UserError("Specify only one axis or plane to 'zalign'")

    atoms = objects.atoms
    if atoms:
        if align_pts:
            raise UserError("Must specify one axis or plane or two atoms for 'zalign'; you specified"
                            " both an axis/plane and atoms")
        if in_front_of is not None:
            if len(atoms) == 0:
                raise UserError('view: zAlign option specified no atoms')
            if len(in_front_of) == 0:
                raise UserError('view: inFrontOf option specified no atoms')
            align_pts = atoms.scene_coords.mean(axis = 0), in_front_of.scene_coords.mean(axis = 0)
        elif len(atoms) == 2:
            align_pts = atoms.scene_coords
        else:
            raise UserError('view: Must specify two atoms with zalign option, got %d' % len(atoms))

    if align_pts is None:
        raise UserError("Must specify one axis or plane or two atoms for 'zalign' option")

    xyz_front, xyz_back = align_pts
    new_view_direction = xyz_back - xyz_front
    center = 0.5*(xyz_front + xyz_back) - camera.position.origin()
    from chimerax.geometry import vector_rotation, translation
    r = (translation(-center)
         * vector_rotation(camera.view_direction(), new_view_direction)
         * translation(center))
    camera.position = r * camera.position
    
def view_name(session, name):
    """Save current view as given name.

    Parameters
    ----------
    name : string
      Name the current camera view and model positions so they can be shown
      later with the "show" option.
    """
    reserved = ('clip', 'cofr', 'delete', 'frames', 'initial',
                'list', 'matrix', 'orient', 'zalign', 'pad', 'position')
    matches = [r for r in reserved if r.startswith(name)]
    if matches:
        raise UserError('view name "%s" conflicts with "%s" view option.\n' % (name, matches[0]) +
                        'Names cannot be option names or their abbreviations:\n %s'
                        % ', '.join('"%s"' % n for n in reserved))
    
    nv = _named_views(session).views
    v = session.main_view
    models = session.models.list()
    nv[name] = NamedView(v, v.center_of_rotation, models)


def view_delete(session, name):
    """Delete named saved view.

    Parameters
    ----------
    name : string
      Name of the view.  "all" deletes all named views.
    """
    nv = _named_views(session).views
    if name == 'all':
        nv.clear()
    elif name in nv:
        del nv[name]


def show_view(session, v2, frames=None):
    """Restore the saved camera view and model positions having this name.

    Parameters
    ----------
    v2 : string
      The view to show.
    frames : int
      Interpolate to the desired view over the specified number of frames.
    """
    if frames is None:
        frames = 1
    v = session.main_view
    models = session.models.list()
    v1 = NamedView(v, v.center_of_rotation, models)
    v2.remove_deleted_models()
    _InterpolateViews(v1, v2, frames, session)

    from chimerax import surface
    surface.update_clip_caps(v)

def view_list(session):
    """Print the named camera views in the log.

    The names are links and clicking them show the corresponding view.
    """
    nv = _named_views(session).views
    names = ['<a href="cxcmd:view %s">%s</a>' % (name, name) for name in sorted(nv.keys())]
    if names:
        msg = 'Named views: ' + ', '.join(names)
    else:
        msg = 'No named views.'
    session.logger.info(msg, is_html=True)


def _named_views(session):
    # Returns dictionary mapping name to NamedView.
    if not hasattr(session, '_named_views'):
        session._named_views = nvs = NamedViews()
        session.add_state_manager('named views', nvs)
    return session._named_views


from chimerax.core.state import State, StateManager
class NamedView(State):
    camera_attributes = ('position', 'field_of_view', 'field_width',
                         'eye_separation_scene', 'eye_separation_pixels')

    def __init__(self, view, look_at, models):
        camera = view.camera
        self.camera = {attr: getattr(camera, attr)
                       for attr in self.camera_attributes if hasattr(camera, attr)}
        self.clip_planes = [p.copy() for p in view.clip_planes.planes()]

        # Scene point which is focus of attention used when
        # interpolating between two views so that the focus
        # of attention stays steady as camera moves and rotates.
        self.look_at = look_at

        # Save model positions
        self.positions = pos = {}
        for m in models:
            pos[m] = m.positions

    def set_view(self, view, models):
        # Set camera
        for attr, value in self.camera.items():
            setattr(view.camera, attr, value)

        # Set clip planes.
        view.clip_planes.replace_planes([p.copy() for p in self.clip_planes])

        # Set model positions
        pos = self.positions
        for m in models:
            if m in pos:
                p = pos[m]
                if m.positions is not p:
                    m.positions = p

    def remove_deleted_models(self):
        pos = self.positions
        for m in tuple(pos.keys()):
            if m.deleted:
                del pos[m]
                
    # Session saving for a named view.
    version = 1
    save_attrs = [
        'camera',
        'clip_planes',
        'look_at',
        'positions',
    ]
    def take_snapshot(self, session, flags):
        self.remove_deleted_models()
        vattrs = {a:getattr(self,a) for a in self.save_attrs}
        vattrs['positions'] = {m:p for m,p in self.positions.items() if m.SESSION_SAVE}
        data = {'view attrs': vattrs,
                'version': self.version}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        nv = NamedView.__new__(NamedView)
        for k,v in data['view attrs'].items():
            setattr(nv, k, v)
        # Fix view where a model was deleted in old sessions. Bug #1829.
        if None in nv.positions:
            del nv.positions[None]
        return nv

class NamedViews(StateManager):
    def __init__(self):
        self._views = {}	# Maps name to NamedView

    @property
    def views(self):
        return self._views

    def clear(self):
        self._views.clear()
    
    # Session saving for named views.
    version = 1
    def take_snapshot(self, session, flags):
        data = {'views': self.views,
                'version': self.version}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        nvs = _named_views(session)	# Get singleton NamedViews object
        nvs._views = data['views']
        return nvs

    def reset_state(self, session):
        nvs = _named_views(session)
        nvs.clear()

class _InterpolateViews:
    def __init__(self, v1, v2, frames, session):
        self.view1 = v1
        self.view2 = v2
        self.frames = frames
        self.centers = _model_motion_centers(v1.positions, v2.positions)
        if frames == 1:
            self.frame_cb(session, 0)
        else:
            from chimerax.core.commands import motion
            motion.CallForNFrames(self.frame_cb, frames, session)

    def frame_cb(self, session, frame):
        v1, v2 = self.view1, self.view2
        v = session.main_view
        if frame == self.frames - 1:
            models = session.models.list()
            v2.set_view(v, models)
        else:
            f = frame / self.frames
            _interpolate_views(v1, v2, f, v, self.centers)


def _interpolate_views(v1, v2, f, view, centers):
    _interpolate_camera(v1, v2, f, view.camera)
    _interpolate_clip_planes(v1, v2, f, view)
    _interpolate_model_positions(v1, v2, centers, f)


def _interpolate_camera(v1, v2, f, camera):
    c1, c2 = v1.camera, v2.camera

    # Interpolate camera position
    from chimerax.geometry import interpolate_rotation, interpolate_points
    p1, p2 = c1['position'], c2['position']
    r = interpolate_rotation(p1, p2, f)
    la = interpolate_points(v1.look_at, v2.look_at, f)
    # Look-at points in camera coordinates
    cl1 = p1.inverse() * v1.look_at
    cl2 = p2.inverse() * v2.look_at
    cla = interpolate_points(cl1, cl2, f)
    # Make camera translation so that camera coordinate look-at point
    # maps to scene coordinate look-at point r*cla + t = la.
    from chimerax.geometry import translation
    t = translation(la - r * cla)
    camera.position = t * r

    # Interpolate field of view
    if 'field_of_view' in c1 and 'field_of_view' in c2:
        camera.field_of_view = (1 - f) * c1['field_of_view'] + f * c2['field_of_view']
    elif 'field_width' in c1 and 'field_width' in c2:
        camera.field_width = (1 - f) * c1['field_width'] + f * c2['field_width']

    camera.redraw_needed = True


def _interpolate_clip_planes(v1, v2, f, view):
    # Currently interpolate only if both states have clipping enabled and
    # clip plane scene normal is identical.
    p1 = {p.name: p for p in v1.clip_planes}
    p2 = {p.name: p for p in v2.clip_planes}
    pv = {p.name: p for p in view.clip_planes.planes()}
    from numpy import array_equal
    for name in p1:
        if name in p2 and name in pv:
            p1n, p2n, pvn = p1[name], p2[name], pv[name]
            if array_equal(p1n.normal, p2n.normal):
                pvn.normal = p1n.normal
                pvn.plane_point = (1 - f) * p1n.plane_point + f * p2n.plane_point
                # TODO: Update pv._last_distance


def _interpolate_model_positions(v1, v2, centers, f):
    # Only interplates models with positions in both views that have not changed number of instances.
    p1, p2 = v1.positions, v2.positions
    models = set(m for m, p in p1.items()
                 if m in p2 and p2[m] is not p and len(p2[m]) == len(p) and m in centers)
    for m in models:
        m.positions = _interpolated_positions(p1[m], p2[m], centers[m], f)


def _interpolated_positions(places1, places2, center, f):
    from chimerax.geometry import Places
    pf = Places([p1.interpolate(p2, p1.inverse() * center, f)
                 for p1, p2 in zip(places1, places2)])
    return pf


# Compute common center of rotation for models that move rigidly as a group
# and have the same parent model.
def _model_motion_centers(mpos1, mpos2):
    bounds = {}
    tf_bounds = []
    for m, p1 in mpos1.items():
        if m in mpos2:
            p2 = mpos2[m]
            b = m.bounds()
            if b:
                tf = p2[0] * p1[0].inverse()
                blist = _close_transform(tf, tf_bounds, m.parent)
                blist.append(b)
                bounds[m] = blist

    from chimerax.geometry import union_bounds
    centers = {m: union_bounds(blist).center() for m, blist in bounds.items()}
    return centers


def _close_transform(tf, tf_bounds, parent, max_rotation_angle=0.01, max_shift=1):
    tfinv = tf.inverse()
    center = (0, 0, 0)
    for bparent, tf2, blist in tf_bounds:
        if parent is bparent:
            shift, angle = (tfinv * tf2).shift_and_angle(center)
            if angle <= max_rotation_angle and shift < max_shift:
                return blist
    blist = []
    tf_bounds.append((parent, tf, blist))
    return blist


class NamedViewArg(Annotation):
    """Annotation for named views"""
    name = "a view name"

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import next_token
        token, text, rest = next_token(text)
        nv = _named_views(session).views
        if token in nv:
            return nv[token], text, rest
        raise AnnotationError("Expected a view name")

def view_initial(session, models=None):
    '''
    Set models to initial positions.

    Parameters
    ----------
    models : Models
      Set model positions to no rotation, no shift.
    '''

    if models is None:
        models = session.models.list()
    from chimerax.geometry import Place
    for m in models:
        m.position = Place()

def view_matrix(session, camera=None, models=None, coordinate_system=None):
    '''
    Set model and camera positions. With no options positions are reported
    for the camera and all models. Positions are specified as 12-numbers,
    the rows of a 3-row, 4-column matrix where the first 3 columns are a
    rotation matrix and the last column is a translation applied after the rotation.

    Parameters
    ----------
    camera : Place
      Set the camera position.
    models : list of (Model, Place)
      Set model positions.
    coordinate_system : Place
      Treat camera and model positions relative to this coordinate system.
      If none, then positions are in scene coordinates.
    '''
    v = session.main_view
    csys = coordinate_system
    if camera is not None:
        v.camera.position = camera if csys is None else csys*camera
    if models is not None:
        for m,p in models:
            m.position = p if csys is None else csys*p

    if camera is None and models is None:
        report_positions(session)

def report_positions(session):
    c = session.main_view.camera
    lines = ['view matrix camera %s' % _position_string(c.position)]

    # List models belonging to the scene, excluding overlay models
    # that don't use the position matrix such as 2D labels and color keys.
    mlist = session.models.scene_root_model.all_models()[1:]
    if mlist:
        lines.append('view matrix models %s\n' % model_positions_string(mlist))
    session.logger.info('\n'.join(lines))

def model_positions_string(models):
    mpos = ','.join('#%s,%s' % (m.id_string, _position_string(m.position)) for m in models)
    return mpos

def _position_string(p):
    return ','.join('%.5g' % x for x in tuple(p.matrix.flat))

def view_position(session, models, same_as_models):
    '''
    Change the scene position of some models to match the scene position of other models.
    If to_models is just one model then each model is positioned to match that one model.
    If to_models is more than one model, then models must consist of the same number of
    models and corresponding models in the two lists are aligned.

    Parameters
    ----------
    models : list of Model
      Models to move.
    same_as_models : list of Models
      Models are moved to align with these models.
    '''
    if len(same_as_models) == 1:
        tm = same_as_models[0]
        p = tm.positions
        for m in models:
            if m is not tm:
                m.positions = p
    elif len(models) != len(same_as_models):
        raise UserError('Must specify equal numbers of models to align, got %d and %d'
                        % (len(models), len(same_as_models)))
    else:
        tp = [tm.positions for tm in same_as_models]
        for m,p in zip(models, tp):
                m.positions = p

from chimerax.core.commands import Annotation, AnnotationError
class ModelPlacesArg(Annotation):
    """Annotation for model id and positioning matrix as 12 floats."""
    name = "model positions"

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import next_token, TopModelsArg, PlaceArg
        token, text, rest = next_token(text)
        fields = token.split(',')
        if len(fields) % 13:
            raise AnnotationError("Expected model id and 12 comma-separated numbers")
        mp = []
        while fields:
            tm, mtext, mrest = TopModelsArg.parse(fields[0], session)
            if len(tm) == 0:
                raise AnnotationError('No models specified by "%s"' % fields[0])
            p = PlaceArg.parse_place(fields[1:13])
            try:
                p.inverse()
            except:
                raise AnnotationError('matrix %s is not invertible' % token)
            for m in tm:
                mp.append((m,p))
            fields = fields[13:]
        return mp, text, rest


from chimerax.core.undo import UndoAction
class UndoView(UndoAction):

    def __init__(self, name, session, models, frames=None):
        super().__init__(name, can_redo=False)
        v = session.main_view
        if models is None:
            models = session.models.list()
        self._before = NamedView(v, v.center_of_rotation, models)
        self._after = None
        self._session = session
        self.frames = frames

    def finish(self, session, models):
        v = session.main_view
        if models is None:
            models = session.models.list()
        self._after = NamedView(v, v.center_of_rotation, models)
        self.can_redo = True

    def undo(self):
        with self._session.undo.block():
            view(self._session, objects=self._before,
                 frames=self.frames, need_undo=False)

    def redo(self):
        with self._session.undo.block():
            view(self._session, objects=self._after,
                 frames=self.frames, need_undo=False)


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ObjectsArg, FloatArg
    from chimerax.core.commands import StringArg, PositiveIntArg, Or, BoolArg, NoArg
    from chimerax.core.commands import PlaceArg, ModelsArg, TopModelsArg, Or, CoordSysArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        optional=[('objects', Or(ObjectsArg, NamedViewArg)),
                  ('frames', PositiveIntArg)],
        keyword=[('clip', BoolArg),
                 ('cofr', BoolArg),
                 ('orient', NoArg),
                 ('zalign', ObjectsArg),
                 ('in_front_of', AtomsArg),
                 ('pad', FloatArg)],
        synopsis='adjust camera so everything is visible')
    register('view', desc, view, logger=logger)
    desc = CmdDesc(
        synopsis='list named views')
    register('view list', desc, view_list, logger=logger)
    desc = CmdDesc(
        required=[('name', StringArg)],
        synopsis='delete named view')
    register('view delete', desc, view_delete, logger=logger)
    desc = CmdDesc(
        required=[('name', StringArg)],
        synopsis='save view with name')
    register('view name', desc, view_name, logger=logger)
    desc = CmdDesc(
        optional=[('models', ModelsArg)],
        synopsis='set models to initial positions')
    register('view initial', desc, view_initial, logger=logger)
    desc = CmdDesc(
        keyword=[('camera', PlaceArg),
                 ('models', ModelPlacesArg),
                 ('coordinate_system', CoordSysArg)],
        synopsis='set camera and model positions')
    register('view matrix', desc, view_matrix, logger=logger)
    desc = CmdDesc(
        required=[('models', TopModelsArg)],
        keyword=[('same_as_models', TopModelsArg)],
        required_arguments = ['same_as_models'],
        synopsis='move models to have same scene position as another model')
    register('view position', desc, view_position, logger=logger)
