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

"""
models: Displayed data
======================

"""

ADD_MODELS = 'add models'
REMOVE_MODELS = 'remove models'
MODEL_COLOR_CHANGED = 'model color changed'
MODEL_DISPLAY_CHANGED = 'model display changed'
MODEL_ID_CHANGED = 'model id changed'
MODEL_NAME_CHANGED = 'model name changed'
MODEL_POSITION_CHANGED = 'model position changed'
MODEL_SELECTION_CHANGED = 'model selection changed'
RESTORED_MODELS = 'restored models'
RESTORED_MODEL_TABLE = 'restored model table'
# One would normally use REMOVE_MODELS trigger above, BEGIN/END_DELETE_MODELS
# is for situations where specific code needs to be in effect as models are deleted
# (e.g. batching atomic Collection pointer updating for efficiency)
BEGIN_DELETE_MODELS = "begin delete models"
END_DELETE_MODELS = "end delete models"
# TODO: register Model as data event type

# If any of the *STATE_VERSIONs change, then increase the (maximum) core session
# number in setup.py.in
MODEL_STATE_VERSION = 1
MODELS_STATE_VERSION = 1

from .state import State
from chimerax.graphics import Drawing, Pick, PickedTriangle
class Model(State, Drawing):
    """A Model is a :class:`.Drawing` together with an id number
    that allows it to be referenced in a typed command.

    Model subclasses can be saved in session files.

    Parameters
    ----------
    name : str
        The name of the model.

    Attributes
    ----------
    id : None or tuple of int
        Model/submodel identification: *e.g.*, 1.3.2 is (1, 3, 2).
        Set and unset by :py:class:`Models` instance.
    SESSION_ENDURING : bool, class-level optional
        If True, then model survives across sessions.
    SESSION_SAVE : bool, class-level optional
        If True, then model is saved in sessions.
    SESSION_WARN : bool, class-level optional
        If True and SESSION_SAVE is False then a warning is issued when
        a session is saved explaining that session save is not supported
        for this type of model.
    """

    SESSION_ENDURING = False
    SESSION_SAVE = True
    SESSION_SAVE_DRAWING = False
    SESSION_WARN = False

    def __init__(self, name, session):
        self._name = name
        Drawing.__init__(self, name)
        self.inherit_graphics_exemptions = False  # Don't copy lighting/clipping exemptions from parent
        self.session = session
        self._id = None
        self._added_to_session = False
        self._deleted = False
        self._selection_coupled = None
        from .triggerset import TriggerSet
        self.triggers  = TriggerSet()
        self.triggers.add_trigger("deleted")
        # TODO: track.created(Model, [self])
        self.opened_data_format = None # for use by 'open' command

    def cpp_del_model(self):
        '''Called by the C++ layer to request that the model be deleted'''
        self.delete()

    def delete(self):
        '''Supported API.  Delete this model.'''
        if self._deleted:
            raise RuntimeError('Model %s was deleted twice' % self._name)
        models = self.session.models
        if models.have_model(self):
            models.close([self])	# Remove from open models list.
            return
        self._deleted = True
        Drawing.delete(self)
        self.triggers.activate_trigger("deleted", self)
        self.session = None

    @property
    def selection_coupled(self):
        if self._selection_coupled is None:
            from chimerax.atomic import AtomicStructures
            self._selection_coupled = AtomicStructures(None)
        return self._selection_coupled

    @selection_coupled.setter
    def selection_coupled(self, value):
        self._selection_coupled = value

    @property
    def deleted(self):
        '''Return whether this model has already been deleted.

        Returns:
           Returns boolean value.  True if model has been deleted;
           False otherwise.
        '''
        # may be overriden in subclass, e.g. Structure
        return self._deleted

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        '''So that generic models can be properly picked.  Most model classes override this.'''
        pick = super().first_intercept(mxyz1, mxyz2, exclude=exclude)
        if isinstance(pick, PickedTriangle):
            picked_model = PickedModel(self, pick.distance)
            picked_model.picked_triangle = pick
            pick = picked_model
        return pick

    def _get_id(self):
        return self._id

    def _set_id(self, val):
        if val == self._id:
            return
        fire_trigger = self._id is not None and val is not None
        self._id = val
        if fire_trigger:
            self.session.triggers.activate_trigger(MODEL_ID_CHANGED, self)
    id = property(_get_id, _set_id)

    @property
    def id_string(self):
        '''Return the dot-separated identifier for this model.
        A top-level model (one that is not a child of another model)
        will have no dots in its identifier.  A child model identifier
        consists of its parent model identifier, followed by a dot
        (period), followed by its (undotted) identifier within
        the parent model.

        Returns:
           A string.  If the model has not been assigned an identifier,
           an empty string is returned.
        '''
        if self.id is None:
            return ''
        return '.'.join(str(i) for i in self.id)

    @property
    def atomspec(self):
        '''Return the atom specifier string for this structure.'''
        return '#' + self.id_string

    def __str__(self):
        if self.id is None:
            return self.name
        return '%s #%s' % (self.name, self.id_string)

    def _get_name(self):
        return self._name

    def _set_name(self, val):
        if val == self._name:
            return
        self._name = val
        if self._id is not None:  # model actually open
            self.session.triggers.activate_trigger(MODEL_NAME_CHANGED, self)
    name = property(_get_name, _set_name)

    def get_selected(self, include_children=False, fully=False):
        '''Is this model selected?  If fully is true then are all parts of this model selected?'''
        if fully:
            if not self.highlighted and not self.empty_drawing():
                return False
            if include_children:
                for d in self.child_drawings():
                    if isinstance(d, Model):
                        if not d.get_selected(include_children=True, fully=True):
                            return False
                    else:
                        if not d.highlighted and not d.empty_drawing():
                            return False

            return self.highlighted

        if self.highlighted:
            return True

        if include_children:
            for d in self.child_drawings():
                if isinstance(d, Model):
                    if d.get_selected(include_children=True):
                        return True
                elif d.highlighted:
                    return True

        return False

    def set_selected(self, sel, *, fire_trigger=True):
        Drawing.set_highlighted(self, sel)
        if fire_trigger:
            self._selection_changed()

    def _selection_changed(self):
        self.session.selection.trigger_fire_needed = True
        self.session.triggers.activate_trigger(MODEL_SELECTION_CHANGED, self)

    # Provide a direct way to set only the model selection status
    # without subclass interference
    set_model_selected = set_selected

    selected = property(get_selected, set_selected)
    '''selected indicates if this model has any part selected but does not include children.'''

    def _get_selected_positions(self):
        return self.highlighted_positions
    def _set_selected_positions(self, positions):
        self.highlighted_positions = positions
    selected_positions = property(_get_selected_positions, _set_selected_positions)

    def _model_set_position(self, pos):
        if pos != self.position:
            Drawing.position.fset(self, pos)
            self.session.triggers.activate_trigger(MODEL_POSITION_CHANGED, self)
    position = property(Drawing.position.fget, _model_set_position)

    def _model_set_positions(self, positions):
        if positions != self.positions:
            Drawing.positions.fset(self, positions)
            self.session.triggers.activate_trigger(MODEL_POSITION_CHANGED, self)
    positions = property(Drawing.positions.fget, _model_set_positions)

    # Drawing._set_scene_position calls _set_positions, so don't need to override

    def _get_model_color(self):
        return self.color if self.vertex_colors is None else None

    def _set_model_color(self, color):
        self.color = color
        self.vertex_colors = None
        self.session.triggers.activate_trigger(MODEL_COLOR_CHANGED, self)
    model_color = property(_get_model_color, _set_model_color)
    '''
    Getting the model color may give the dominant color.
    Setting the model color will set the model to that color.
    Color values are rgba uint8 arrays.
    '''

    # Handle undo of color changes
    def _color_undo_state(self):
        vc = self.vertex_colors
        color_state = {'colors': self.colors,
                       'vertex_colors': (vc if vc is None else vc.copy()),
                       'auto_recolor_vertices': self.auto_recolor_vertices}
        return color_state
    def _restore_colors_from_undo_state(self, color_state):
        self.colors = color_state['colors']
        vc = color_state['vertex_colors']
        same_vertex_count = (vc is not None and
                             self.vertices is not None and
                             len(vc) == len(self.vertices))
        if not same_vertex_count:
            vc = None
        self.vertex_colors = vc
        auto_recolor = color_state['auto_recolor_vertices']
        self.auto_recolor_vertices = auto_recolor
        if not same_vertex_count and auto_recolor:
            # Number of vertices changed.  Recompute colors.
            auto_recolor()

    color_undo_state = property(_color_undo_state, _restore_colors_from_undo_state)

    def add(self, models) -> None:
        '''Add child models to this model.'''
        om = self.session.models
        if type(models) is not list:
            models = [models]
        if om.have_model(self):
            # Parent already open.
            om.add(models, parent = self)
        else:
            for m in models:
                self.add_drawing(m)

    def child_models(self):
        '''Return child models.'''
        return [d for d in self.child_drawings() if isinstance(d, Model)]

    def all_models(self):
        '''Return all models including self and children at all levels.'''
        dlist = [self]
        for d in self.child_drawings():
            if isinstance(d, Model):
                dlist.extend(d.all_models())
        return dlist

    @property
    def visible(self):
        if self.display:
            p = self.parent
            return p is None or p.visible
        return False

    def __lt__(self, other):
        # for sorting (objects of the same type)
        if self.id is None:
            return self.name < other.name
        return self.id < other.id

    def _set_display(self, display):
        Drawing.set_display(self, display)
        self.session.triggers.activate_trigger(MODEL_DISPLAY_CHANGED, self)
    display = Drawing.display.setter(_set_display)

    @property
    def _save_in_session(self):
        '''Test if all parents are saved in session.'''
        m = self
        while m is not None and m.SESSION_SAVE:
            m = m.parent
        return m is None

    def take_snapshot(self, session, flags):
        p = self.parent
        if p is session.models.scene_root_model:
            p = None    # Don't include root as a parent since root is not saved.
        data = {
            'name': self.name,
            'id': self.id,
            'parent': p,
            'positions': self.positions.array(),
            'display_positions': self.display_positions,
            'allow_depth_cue': self.allow_depth_cue,
            'allow_clipping': self.allow_clipping,
            'accept_shadow': self.accept_shadow,
            'accept_multishadow': self.accept_multishadow,
            'opened_data_format': self.opened_data_format,
            'version': MODEL_STATE_VERSION,
        }
        if hasattr(self, 'clip_cap'):
            data['clip_cap'] = self.clip_cap
        if self.SESSION_SAVE_DRAWING:
            from chimerax.graphics.gsession import DrawingState
            data['drawing state'] = DrawingState.take_snapshot(self, session, flags,
                                                               include_children = False)
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        if cls is Model and data['id'] == ():
            return session.models.scene_root_model
        # TODO: Could call the cls constructor here to handle a derived class,
        #       but that would require the derived constructor have the same args.
        m = Model(data['name'], session)
        m.set_state_from_snapshot(session, data)
        return m

    def set_state_from_snapshot(self, session, data):
        self.name = data['name']
        self.id = data['id']
        p = data['parent']
        if p:
            p.add([self])

        pa = data['positions']
        from numpy import float32, float64
        if pa.dtype == float32:
            # Fix old sessions that saved array as float32
            pa = pa.astype(float64)
        from chimerax.geometry import Places
        self.positions = Places(place_array=pa)
        self.display_positions = data['display_positions']
        for d in self.all_drawings():
            for attr in ['allow_depth_cue', 'allow_clipping', 'accept_shadow', 'accept_multishadow']:
                if attr in data:
                    setattr(d, attr, data[attr])

        if 'clip_cap' in data:
            self.clip_cap = data['clip_cap']

        if 'drawing state' in data:
            from chimerax.graphics.gsession import DrawingState
            DrawingState.set_state_from_snapshot(self, session, data['drawing state'])
            self.SESSION_SAVE_DRAWING = True

        if 'opened_data_format' in data:
            self.opened_data_format = data['opened_data_format']

    def save_geometry(self, session, flags):
        '''
        Return state for saving Model and Drawing geometry that can be restored
        with restore_geometry().
        '''
        from chimerax.graphics.gsession import DrawingState
        data = {'model state': Model.take_snapshot(self, session, flags),
                'drawing state': DrawingState.take_snapshot(self, session, flags),
                'version': 1
                }
        return data

    def restore_geometry(self, session, data):
        '''
        Restore model and drawing state saved with save_geometry().
        '''
        from chimerax.graphics.gsession import DrawingState
        Model.set_state_from_snapshot(self, session, data['model state'])
        DrawingState.set_state_from_snapshot(self, session, data['drawing state'])
        return self

    def selected_items(self, itype):
        return []

    def clear_selection(self):
        self.clear_highlight()
        self._selection_changed()

    def added_to_session(self, session):
        html_title = self.get_html_title(session)
        if not html_title:
            return
        if getattr(self, 'prefix_html_title', True):
            fmt = '<i>%s</i> title:<br><b>%%s</b>' % self.name.replace('%', '%%')
        else:
            fmt = '<b>%s</b>'
        if self.has_formatted_metadata(session):
            fmt += ' <a href="cxcmd:log metadata #%s">[more&nbsp;info...]</a>' % self.id_string
        fmt += '<br>'
        session.logger.info(fmt % html_title, is_html=True)

    def removed_from_session(self, session):
        pass

    def get_html_title(self, session):
        return getattr(self, 'html_title', None)

    def show_metadata(self, session, *, verbose=False, log=None, **kw):
        '''called by 'log metadata' command.'''
        formatted_md = self.get_formatted_metadata(session, verbose=verbose, **kw)
        if log:
            if formatted_md:
                log.log(log.LEVEL_INFO, formatted_md, (None, False), True)
            else:
                log.log(log.LEVEL_INFO, "No additional info for %s" % self, (None, False), False)
        else:
            if formatted_md:
                session.logger.info(formatted_md, is_html=True)
            else:
                session.logger.info("No additional info for %s" % self)

    def has_formatted_metadata(self, session):
        '''Can override both this and 'get_formatted_metadata' if lazy evaluation desired'''
        return hasattr(self, 'formatted_metadata')

    def get_formatted_metadata(self, session, *, verbose=False, **kw):
        formatted = getattr(self, 'formatted_metadata', None)
        return getattr(self, 'verbose_formatted_metadata', formatted) if verbose else formatted

    # Atom specifier API
    def atomspec_has_atoms(self):
        # Return True if there are atoms in this model
        return False

    def atomspec_has_pseudobonds(self):
        # Return True if there are pseudobonds in this model
        return False

    def atomspec_zone(self, session, coords, distance, target_type, operator, results):
        # Ignore zone request by default
        pass

    def atomspec_model_attr(self, attrs):
        # Return true is attributes specifier matches model
        for attr in attrs:
            try:
                v = getattr(self, attr.name)
            except AttributeError:
                if not attr.no:
                    return False
            else:
                if attr.value is None:
                    tv = attr.op(v)
                else:
                    tv = attr.op(v, attr.value)
                if not tv:
                    return False
        return True

    def show_info(self):
        pass

class PickedModel(Pick):
    def __init__(self, model, distance):
        super().__init__(distance)
        self.model = model

    def description(self):
        return str(self.model)

    def drawing(self):
        return self.model

    def select(self, mode='add'):
        from chimerax.core.commands import run
        m = self.model
        if mode == 'add' or (mode == 'toggle' and not m.selected):
            run(m.session, f"select add {m.atomspec}")
        else:
            run(m.session, f"select subtract {m.atomspec}")

    def specifier(self):
        return self.model.atomspec

class Surface(Model):
    '''
    A surface is a type of model where vertex coloring, style (filled, mesh, dot) and masking
    can be controlled by user commands.
    '''
    @classmethod
    def restore_snapshot(cls, session, data):
        m = Surface(data['name'], session)
        m.set_state_from_snapshot(session, data)
        return m

from .state import StateManager
class Models(StateManager):
    '''
    Models manages the Model instances shown in the scene belonging to a session.
    It makes a root model (attribute scene_root_model) that the graphics View uses
    to render the scene.

    Another major function of Models is to assign the id numbers to each Model.
    An id number is a non-empty tuple of positive integers.  A Model can have
    child models (m.child_models()) and usually has a parent model (m.parent).
    The id number of a child is a tuple one longer than the parent and equals
    the parent id except for the last integer.  For instance a model with id
    (2,5) could have a child model with id (2,5,1).  Every model has a unique
    id number.  The id of the scene_root_model is the empty tuple, and is not
    used in commands as it has no number representation.

    While most models are children at some depth below the scene_root_model it is
    allowed to have other models with single integer id that have no parent.
    This is currently used for 2D labels which are overlay drawings and are not
    part of the 3D scene. These models are added using Models.add(models, root_model = True).
    Their id numbers can be used by commands.

    Most Model instances are added to the scene with Models.add().  In some
    instances a Model might be created for doing a calculation and is never drawn,
    is never added to the scene, and is never assigned an id number, so cannot
    be referenced in user typed commands.
    '''
    def __init__(self, session):
        import weakref
        self._session = weakref.ref(session)
        t = session.triggers
        t.add_trigger(ADD_MODELS)
        t.add_trigger(REMOVE_MODELS)
        t.add_trigger(MODEL_COLOR_CHANGED)
        t.add_trigger(MODEL_DISPLAY_CHANGED)
        t.add_trigger(MODEL_ID_CHANGED)
        t.add_trigger(MODEL_NAME_CHANGED)
        t.add_trigger(MODEL_POSITION_CHANGED)
        t.add_trigger(MODEL_SELECTION_CHANGED)
        t.add_trigger(RESTORED_MODELS)
        t.add_trigger(RESTORED_MODEL_TABLE)
        t.add_trigger(BEGIN_DELETE_MODELS)
        t.add_trigger(END_DELETE_MODELS)
        self._models = {}				# Map id to Model
        self._scene_root_model = r = Model("root", session)
        r.id = ()
        self._initialize_camera = True
        from .commands.atomspec import check_selectors
        t.add_handler(REMOVE_MODELS, check_selectors)

    def take_snapshot(self, session, flags):
        models = {}
        not_saved = []
        for id, model in self._models.items():
            assert(isinstance(model, Model))
            if not model._save_in_session:
                not_saved.append(model)
                continue
            models[id] = model
        data = {'models': models,
                'version': MODELS_STATE_VERSION}
        if not_saved:
            mwarn = [m for m in not_saved
                     if m.SESSION_WARN and (m.parent is None or m.parent.SESSION_SAVE)]
            if mwarn:
                log = self._session().logger
                log.bug('The session file will not include the following models'
                        ' because these model types have not implemented saving: %s'
                        % ', '.join('%s #%s' % (m.name, m.id_string) for m in mwarn))
        return data

    @staticmethod
    def restore_snapshot(session, data):
        mdict = data['models']
        session.triggers.activate_trigger(RESTORED_MODEL_TABLE, mdict)
        sm = session.models
        if session.restore_options['combine']:
            existing_ids = set([m.id for m in sm])
            requested_ids = set(mdict.keys())
            if existing_ids.isdisjoint(requested_ids):
                id_offset = 0
            else:
                id_offset = max([id[0] for id in existing_ids])
        else:
            id_offset = 0
        for id, model in mdict.items():
            if model:        # model can be None if it could not be restored, eg Volume w/o map file
                if id_offset and model.parent is None:
                    for am in model.all_models():
                        am.id = (am.id[0] + id_offset,) + am.id[1:]
                    test_id = model.id
                else:
                    test_id = id
                if model.parent is None and not sm.have_id(test_id):
                    sm.add([model], _from_session=True)
        session.triggers.activate_trigger(RESTORED_MODELS, session)
        return sm

    def reset_state(self, session):
        self.close([m for m in self.list() if not m.SESSION_ENDURING])

    @property
    def scene_root_model(self):
        return self._scene_root_model

    def list(self, model_id=None, type=None):
        if model_id is None:
            models = list(self._models.values())
        else:
            models = [self._models[model_id]] if model_id in self._models else []
        if type is not None:
            models = [m for m in models if isinstance(m, type)]
        return models

    def empty(self):
        return len(self._models) == 0

    def add(self, models, parent=None, minimum_id = 1, root_model = False,
            _notify=True, _need_fire_id_trigger=None, _from_session=False):
        '''
        Assigns id numbers to the specified models and their child models.
        An id number is a tuple of positive integers.
        Each model has exactly one parent (except the scene root model).
        The id number of a parent is id number number of a child with
        the last integer removed.  So the parent of a model with id (2,1,5)
        would have id (2,1).

        In the typical case the specified models have model.id = None and
        model.parent = None.
        If the parent option is not given, each of these models are given an
        unused top level id number (tuple containing 1 integer), and child
        models are also given id numbers to all depths.  If the parent argument
        is specified then each model is made a child of that parent.

        There are other unusual cases.  A specified model that has not been
        added may have model.id not None.  This is a request to use this id for
        the model. The model with the parent id must already have been added.
        If the parent option is given and does not have the parent id, it is
        an error.

        Another unusual case is that a specified model has already been added.
        This is a request to add the model to the specified parent, changing its
        id as needed.  All children to all depths will also be assigned new ids.
        If the parent option is not specified it is an error.  If the specified
        parent is the model itself or one of its descendants it is an error.

        The specified models cannot contain a model which is a descendant of
        another specified model.  That is an error.

        In all cases if the parent option is given it must be a model that has
        already been added.

        After a model has been added, model.id should not be changed except by
        this routine, or Models.assign_id(), or Models.remove() which sets model.id = None.

        The root_model option allows adding a Model that has no parent.  It will
        not be added as a child of the scene_root_model.  This is currently used
        for 2D labels which are not part of the 3D scene, but instead are overlays
        drawn on top of the graphics.  A root model has an id that is unique among
        all model ids so it can be specified in user typed commands.
        '''
        if _need_fire_id_trigger is None:
            _need_fire_id_trigger = []

        if len(self._models) == 0:
            self._initialize_camera = True

        # Add models to parents and assign model ids.
        for model in models:
            if self.have_model(model):
                # Model is already added and is being reparented.
                if parent is None:
                    raise ValueError('Attempted to add model %s to scene twice' % model)
                if model.parent is not parent:
                    # Model is being moved to a new parent
                    _need_fire_id_trigger.extend(model.all_models())
                    self._reparent_model(model, parent, minimum_id = minimum_id)
            else:
                # Model has not yet been added.
                p = self._parent_for_added_model(model, parent, root_model = root_model)
                if p:
                    p.add_drawing(model)

                # Set model id
                if model.id is None:
                    # Assign a new model id.
                    model.id = self.next_id(parent = p, minimum_id = minimum_id)
                else:
                    self._reset_next_id(parent = p)
                self._models[model.id] = model

                # Add child models
                children = model.child_models()
                if children:
                    self.add(children, parent=model, _notify=False,
                             _need_fire_id_trigger=_need_fire_id_trigger)

        # Notify that models were added
        if _notify:
            session = self._session()
            m_add = [m for model in models for m in model.all_models() if not m._added_to_session]
            for m in m_add:
                m._added_to_session = True
                m.added_to_session(session)
            session.triggers.activate_trigger(ADD_MODELS, m_add)

            # IDs that change from None to non-None don't fire the MODEL_ID_CHANGED
            # trigger, so do it by hand
            for id_changed_model in _need_fire_id_trigger:
                session = self._session()
                session.triggers.activate_trigger(MODEL_ID_CHANGED, id_changed_model)

        # Initialize view if first model added
        if self._initialize_camera and _notify and not _from_session:
            v = session.main_view
            if v.drawing_bounds():
                self._initialize_camera = False
                v.initial_camera_view()
                v.clip_planes.clear()   # Turn off clipping

        if _from_session:
            self._initialize_camera = False

    def have_model(self, model):
        return model.id is not None and (self._models.get(model.id) is model
                                         or model is self.scene_root_model)

    def _parent_for_added_model(self, model, parent, root_model = False):
        if root_model:
            if parent is not None:
                raise ValueError('Tried to add model %s as a root model but specified parent %s'
                                 % (model, parent))
            if model.id is not None and len(model.id) != 1:
                raise ValueError('Tried to add model %s as a root model but id is not a single integer'
                                 % model)
            p = None
        elif model.id is None:
            p = self.scene_root_model if parent is None else parent
        else:
            par_id = model.id[:-1]
            p = self._models.get(par_id) if par_id else self.scene_root_model
            if p is None:
                raise ValueError('Tried to add model %s but parent #%s does not exist'
                                 % (model, '.'.join('%d'% i for i in par_id)))
            if parent is not None and parent is not p:
                raise ValueError('Tried to add model %s to parent %s with incompatible id'
                                 % (model, parent))
            if model.id in self._models:
                raise ValueError('Tried to add model %s with the same id as another model %s'
                                 % (model, self._models[model.id]))
        return p

    def _reparent_model(self, model, parent, minimum_id = 1):
        # Remove old model id from table
        mt = self._models
        del mt[model.id]
        p = model.parent
        if p is not None:
            p._next_unused_id = None

        # Set new model id
        id = self.next_id(parent = parent, minimum_id = minimum_id)
        mt[id] = model
        model.id = id

        # Update model parent.
        # This also removes drawing from former parent.
        parent.add_drawing(model)

        # Change child model ids.
        self._update_child_ids(model)

    def _update_child_ids(self, model):
        id = model.id
        mt = self._models
        for child in model.child_models():
            del mt[child.id]
            child.id = id + child.id[-1:]
            mt[child.id] = child
            self._update_child_ids(child)

    def assign_id(self, model, id):
        '''Parent model for new id must already exist.'''
        cm = self._models.get(id)
        if cm:
            if cm is model:
                return
            else:
                raise ValueError('Tried to change model %s id to one in use by %s'
                                 % (model, cm))

        # Remove old id
        mt = self._models
        del mt[model.id]
        if model.parent:
            model.parent._next_unused_id = None

        # Set new id.
        model.id = id
        mt[id] = model

        # Set parent model.
        if len(id) > 1:
            p = mt[id[:-1]]
        elif model.parent is None:
            p = None 		# Root model
        else:
            p = self.scene_root_model

        if p:
            p.add_drawing(model)

        # Update child model ids.
        self._update_child_ids(model)

    def have_id(self, id):
        return id in self._models

    def __getitem__(self, i):
        '''index into models using square brackets (e.g. session.models[i])'''
        return list(self._models.values())[i]

    def __iter__(self):
        '''iterator over models'''
        return iter(self._models.values())

    def __len__(self):
        '''number of models'''
        return len(self._models)

    def __bool__(self):
        return len(self._models) != 0

    def next_id(self, parent = None, minimum_id = 1):
        # Find lowest unused id.  Typically all ids 1,...,N are used with no gaps
        # and then it is fast to assign N+1 to the next model.  But if there are
        # gaps it can take O(N**2) time to figure out ids to assign for N models.
        # This code handles the common case of no gaps quickly.
        if parent is None:
            parent = self.scene_root_model
        nid = getattr(parent, '_next_unused_id', None)
        if nid is None:
            # Find next unused id.
            cids = set(m.id[-1] for m in parent.child_models() if m.id is not None)
            if parent is self.scene_root_model:
                # Include ids of overlay models that are not part of scene root.
                for id in self._models.keys():
                    cids.add(id[0])
            for nid in range(minimum_id, minimum_id + len(cids) + 1):
                if nid not in cids:
                    break
            if nid == minimum_id + len(cids):
                parent._next_unused_id = nid + 1        # No gaps in ids
        elif nid+1 >= minimum_id:
            parent._next_unused_id = nid + 1            # No gaps in ids
        else:
            nid = minimum_id
            parent._next_unused_id = None               # Have gaps in ids
        id = parent.id + (nid,)
        return id

    def _reset_next_id(self, parent=None):
        if parent is None:
            parent = self.scene_root_model
        parent._next_unused_id = None
        
    def add_group(self, models, name=None, id=None, parent=None):
        if name is None:
            names = set([m.name for m in models])
            if len(names) == 1:
                name = names.pop() + " group"
            else:
                name = "group"
        group = Model(name, self._session())
        if id is not None:
            group.id = id
        group.add(models)
        self.add([group], parent=parent)
        return group

    def remove(self, models):
        '''
        Remove the specified models from the scene as well as all their
        children to all depths.  The models are not deleted.  Their model
        id numbers are set to None and they are removed as children from
        parent models that are still in the scene.
        '''
        # Also remove all child models, and remove deepest children first.
        dset = descendant_models(models)
        dset.update(models)
        mlist = list(dset)
        mlist.sort(key=lambda m: len(m.id), reverse=True)

        # Call remove_from_session() methods.
        session = self._session()  # resolve back reference
        for m in mlist:
            m._added_to_session = False
            m.removed_from_session(session)

        # Remove model ids
        for model in mlist:
            model_id = model.id
            if model_id is not None:
                del self._models[model_id]
                model.id = None

        # Remove models from parent if parent was not removed.
        for model in models:
            parent = model.parent
            if parent is not None and self.have_model(parent):
                parent.remove_drawing(model, delete=False)
                parent._next_unused_id = None

        # it's nice to have an accurate list of current models
        # when firing this trigger, so do it last
        session.triggers.activate_trigger(REMOVE_MODELS, mlist)

        return mlist

    def close(self, models):
        '''
        Remove the models from the scene as well as all child models
        to all depths, and delete the models.  Models that are not
        part of the scene are deleted, and models that have already
        been deleted are ignored.
        '''
        mopen = [m for m in models if self.have_model(m)]
        self.remove(mopen)
        session = self._session()  # resolve back reference
        session.triggers.activate_trigger(BEGIN_DELETE_MODELS, models)
        try:
            for m in models:
                if not Model.deleted.fget(m):
                    m.delete()
        finally:
            session.triggers.activate_trigger(END_DELETE_MODELS, models)


def descendant_models(models):
    mset = set()
    for m in models:
        for c in m.child_models():
            mset.update(c.all_models())
    return mset


def ancestor_models(models):
    '''Return set of ancestors of models that are not in specified models.'''
    ma = set()
    mset = models if isinstance(models, set) else set(models)
    for m in mset:
        p = m.parent
        if p is not None and p not in mset:
            ma.add(p)
    if ma:
        ma.update(ancestor_models(ma))
    return ma
