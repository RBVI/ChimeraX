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


# -----------------------------------------------------------------------------
# Command to smoothly interpolate between saved model and camera positions.
#
#  Syntax: fly [frames] <posname1> [frames1] <posname2> [frames2] ... <posnameN>
#
# This is similar to the view <named-view> command but performs cubic
# interpolation instead of piecewise linear interpolation and provides
# a more convenient syntax for motion through several positions.
#
def fly(session, named_view_or_frames = []):

    default_frames = 100
    pnames = []
    frames = []
    for f, arg in enumerate(named_view_or_frames):
        try:
            c = int(arg)
        except ValueError:
            # Argument is a named view.
            pnames.append(arg)
            frames.append(default_frames)
        else:
            if f == 0:
                default_frames = c
            elif frames:
                frames[-1] = c
    frames = frames[:-1]

    if len(pnames) == 0:
        from chimerax.core.errors import UserError
        raise UserError('fly: Must list at least one named view')

    if 'start' in pnames:
        from .view import view_name
        view_name(session, 'start')

    from .view import _named_views
    views = _named_views(session).views		# Map name to NamedView
    for pname in pnames:
        if pname not in views:
            from chimerax.core.errors import UserError
            raise UserError('fly: Unknown position name "%s"' % pname)

    params, models = _view_parameters(session, pnames)
    
    from chimerax.geometry import natural_cubic_spline
    fparams = natural_cubic_spline(params, frames, tangents = False)
    
    FlyPlayback(session, fparams, models)

# -----------------------------------------------------------------------------
#
def _view_parameters(session, pnames):
    '''
    Return parameter array for interpolation, and list of moved models.
    '''
    moved_models = _moved_models(session, pnames)
        
    # Set initial position.
    from .view import _named_views
    named_views = _named_views(session).views	# Map name to NamedView
    nview0 = named_views[pnames[0]]
    for m,pos in nview0.positions.items():
        if m in moved_models:
            m.positions = pos

    # Find center of rotation.  Same center used for all models.
    b = session.main_view.drawing_bounds()
    if b is None:
        from chimerax.core.errors import UserError
        raise UserError('fly: Nothing displayed to calculate center of rotation')
    center = b.center()

    # Create camera/position/orientation parameter array for interpolation
    from numpy import empty, float32, dot
    from math import log
    osxf = {}
    npos = sum([len(m.positions) for m in moved_models], 0)
    params = empty((len(pnames),10+10*npos), float32)
    for i,pname in enumerate(pnames):
        nview = named_views[pname]
        # Record camera parameters
        prev_p = params[i-1,:] if i > 0 else None
        _get_camera_parameters(nview, center, params[i,:10], prev_p)
        # Record model position parameters
        mpositions = [nview.positions[m] for m in moved_models]
        mcenters = [_model_rotation_center(m, center, nview.positions) for m in moved_models]
        prev_p = params[i-1,10:] if i > 0 else None
        _get_model_position_parameters(mpositions, mcenters, params[i,10:], prev_p)

    return params, moved_models

# -----------------------------------------------------------------------------
#
def _moved_models(session, pnames):
    from .view import _named_views
    named_views = _named_views(session).views	# Map name to NamedView

    # Get set of models found in named views
    models = set()
    for pname in pnames:
        nview = named_views[pname]
        for m in nview.positions.keys():
            if not m.deleted:
                models.add(m)

    # Find which models move
    moved_models = set()
    for m in models:
        pos_start = None
        for pname in pnames:
            nvpos = named_views[pname].positions
            if m not in nvpos:
                continue
            pos = nvpos[m]
            if pos_start is None:
                pos_start = pos
            elif len(pos) != len(pos_start):
                from chimerax.core.errors import UserError
                raise UserError('fly: model #%s changes number of instance positions'
                                % (m.id_string,) + 
                                ' at named view %s from %d to %d, cannot interpolate'
                                % (pname, len(pos_start), len(pos)))
            elif pos != pos_start:
                moved_models.add(m)
                break

    # Check if all named views contain the same set of models
    for pname in pnames:
        nview = named_views[pname]
        for m in moved_models:
            if m not in nview.positions:
                from chimerax.core.errors import UserError
                raise UserError('fly: position %s does not contain model #%s'
                                % (pname, m.id_string))

    # Check that parents of moved models do not have multiple instances.
    # For that case it is not possible to interpolate rotations about a common
    # scene center.
    for m in moved_models:
        for d in m.drawing_lineage[:-1]:
            if len(d.positions) > 1:
                from chimerax.core.errors import UserError
                raise UserError('fly: model #%s has ancestor #%s with %d instances'
                                % (m.id_string, d.id_string, len(d.positions)) +
                                ', cannot interpolate using a common scene center')

    return moved_models

# -----------------------------------------------------------------------------
#
def _get_camera_parameters(named_view, center_scene, params, prev_params):
    cpos = named_view.camera['position']
    _position_to_parameters(cpos, center_scene, params, prev_params)
    # TODO: Handle clip planes
    # TODO: Handle field of view or field width.
    # TODO: Handle stereo parameters (eye separation)
                
# -----------------------------------------------------------------------------
#
def _model_rotation_center(m, center_scene, positions):
    '''Convert scene center to model parent coordinates.'''
    parent_positions = [positions[d][0] for d in m.drawing_lineage[:-1]
                        if d in positions]
    if len(parent_positions) == 0:
        center = center_scene
    else:
        from chimerax.geometry import product
        scene_pos =  product(parent_positions)
        center = scene_pos.inverse() * center_scene
    return center

# -----------------------------------------------------------------------------
#
def _get_model_position_parameters(places_list, rotation_centers, params, prev_params):
    pos_num = 0
    for positions,center in zip(places_list, rotation_centers):
        for pos in positions:
            prevp = None if prev_params is None else prev_params[pos_num*10:]
            _position_to_parameters(pos, center, params[pos_num*10:], prevp)
            pos_num += 1

# -----------------------------------------------------------------------------
#
def _position_to_parameters(pos, center, params, prev_params):
    c = pos.inverse() * center
    rotq, trans = _transform_to_parameters(pos, c)
    # Choose quaternion sign for shortest interpolation path.
    if prev_params is not None:
        rotq_prev = prev_params[0:4]
        from numpy import dot
        if dot(rotq, rotq_prev) < 0:
            rotq = -rotq
    params[0:4] = rotq
    params[4:7] = c
    params[7:10] = trans

# -----------------------------------------------------------------------------
#
def _parameters_to_position(params):
    rotq = params[0:4]
    from chimerax.geometry import length
    rotq = rotq / length(rotq)	# Normalize quaternion since cubic spline produces non-unit length.
    rotc = params[4:7]
    trans = params[7:10]
    return _transform_from_parameters(rotq, rotc, trans)
    
# -----------------------------------------------------------------------------
#
def _transform_to_parameters(transform, rotation_center):
    '''
    Convert Place to quaternion rotation about specified center and translation.
    '''
    trans = transform.origin() - rotation_center + transform.transform_vector(rotation_center)
    rotq = transform.rotation_quaternion()
    return rotq, trans

# -----------------------------------------------------------------------------
#
def _transform_from_parameters(rotq, rotc, trans):
    '''Convert quaternion, rotation center and translation to a Place.'''
    from chimerax.geometry import translation, quaternion_rotation
    return translation(trans + rotc) * quaternion_rotation(rotq) * translation(-rotc)

# -----------------------------------------------------------------------------
#
class FlyPlayback:

    def __init__(self, session, frame_params, models):

        self._session = session
        self._frame_params = frame_params
        self._models = models
        from chimerax.core.commands import motion
        motion.CallForNFrames(self._new_frame, len(frame_params), session)

    def _new_frame(self, session, frame):
        fparams = self._frame_params
        self._position_camera_and_models(fparams[frame])
        
    def _position_camera_and_models(self, params):
        # Set camera parameters
        cpos = _parameters_to_position(params)
        c = self._session.main_view.camera
        c.position = cpos

        # Set model positions
        pos_num = 0
        from chimerax.geometry import Places
        for m in self._models:
            if m.deleted:
                continue
            n = len(m.positions)
            plist = [_parameters_to_position(params[10+10*(pos_num+i):]) for i in range(n)]
            m.positions = Places(plist)
            pos_num += n

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, RepeatOf, StringArg
    desc = CmdDesc(
        optional=[('named_view_or_frames', RepeatOf(StringArg))],
        synopsis='interpolate camera and model motions for animations')
    register('fly', desc, fly, logger=logger)
