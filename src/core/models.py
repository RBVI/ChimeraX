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

"""
models: Displayed data
======================

"""

import weakref
from .graphics.drawing import Drawing
from .state import State, StateManager, CORE_STATE_VERSION
ADD_MODELS = 'add models'
REMOVE_MODELS = 'remove models'
MODEL_DISPLAY_CHANGED = 'model display changed'
MODEL_ID_CHANGED = 'model id changed'
MODEL_NAME_CHANGED = 'model name changed'
MODEL_POSITION_CHANGED = 'model position changed'
RESTORED_MODELS = 'restored models'
# TODO: register Model as data event type


class Model(State, Drawing):
    """A Model is a :class:`.Drawing` together with an id number
    that allows it to be referenced in a typed command.

    Model subclasses can be saved session files.

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
    """

    SESSION_ENDURING = False
    SESSION_SAVE = True

    def __init__(self, name, session):
        self._name = name
        Drawing.__init__(self, name)
        self.session = session
        self._id = None
        self._added_to_session = False
        self._deleted = False
        self._selection_coupled = None
        from .triggerset import TriggerSet
        self.triggers  = TriggerSet()
        self.triggers.add_trigger("deleted")
        # TODO: track.created(Model, [self])

    def delete(self):
        '''Delete this model.'''
        if self._deleted:
            raise RuntimeError('Model %s was deleted twice' % self._name)
        models = self.session.models
        if models.have_id(self.id) and self in models.list(model_id = self.id):
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

    def _get_id(self):
        return self._id

    def _set_id(self, val):
        if val == self._id:
            return
        fire_trigger = self._id != None and val != None
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
        if self._id != None:  # model actually open
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
            from chimerax.core.selection import SELECTION_CHANGED
            self.session.ui.thread_safe(self.session.triggers.activate_trigger,
                SELECTION_CHANGED, None)

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

    def _get_single_color(self):
        return self.color if self.vertex_colors is None else None

    def _set_single_color(self, color):
        self.color = color
        self.vertex_colors = None
    single_color = property(_get_single_color, _set_single_color)
    '''
    Getting the single color may give the dominant color.
    Setting the single color will set the model to that color.
    Color values are rgba uint8 arrays.
    '''

    def add(self, models):
        '''Add child models to this model.'''
        om = self.session.models
        if om.have_id(self.id):
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
            'accept_shadow': self.accept_shadow,
            'accept_multishadow': self.accept_multishadow,
            'version': CORE_STATE_VERSION,
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        if cls is Model and data['id'] is ():
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
        from .geometry import Places
        self.positions = Places(place_array=pa)
        self.display_positions = data['display_positions']
        for d in self.all_drawings():
            for attr in ['allow_depth_cue', 'accept_shadow', 'accept_multishadow']:
                if attr in data:
                    setattr(d, attr, data[attr])

    def selected_items(self, itype):
        return []

    def clear_selection(self):
        self.clear_highlight()

    def added_to_session(self, session):
        html_title = self.get_html_title(session)
        if not html_title:
            return
        fmt = '<i>%s</i> title:<br><b>%s</b>'
        if self.has_formatted_metadata(session):
            fmt += ' <a href="cxcmd:log metadata #%s">[more&nbsp;info...]</a>' % self.id_string
        fmt += '<br>'
        session.logger.info(fmt % (self.name, html_title) , is_html=True)

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


class Surface(Model):
    '''
    A surface is a type of model where vertex coloring, style (filled, mesh, dot) and masking
    can be controlled by user commands.
    '''
    pass

class Models(StateManager):

    def __init__(self, session):
        self._session = weakref.ref(session)
        t = session.triggers
        t.add_trigger(ADD_MODELS)
        t.add_trigger(REMOVE_MODELS)
        t.add_trigger(MODEL_DISPLAY_CHANGED)
        t.add_trigger(MODEL_ID_CHANGED)
        t.add_trigger(MODEL_NAME_CHANGED)
        t.add_trigger(MODEL_POSITION_CHANGED)
        t.add_trigger(RESTORED_MODELS)
        self._models = {}				# Map id to Model
        self._scene_root_model = r = Model("root", session)
        r.id = ()

    def take_snapshot(self, session, flags):
        models = {}
        for id, model in self._models.items():
            assert(isinstance(model, Model))
            if not model.SESSION_SAVE:
                continue
            models[id] = model
        data = {'models': models,
                'version': CORE_STATE_VERSION}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        mdict = data['models']
        session.triggers.activate_trigger(RESTORED_MODELS, tuple(mdict.values()))
        m = session.models
        for id, model in mdict.items():
            if model:        # model can be None if it could not be restored, eg Volume w/o map file
                if model.parent is None:
                    m.add([model], _from_session=True)
        return m

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

    def add(self, models, parent=None, minimum_id = 1,
            _notify=True, _need_fire_id_trigger=None, _from_session=False):
        if _need_fire_id_trigger is None:
            _need_fire_id_trigger = []
        start_count = len(self._models)

        d = self.scene_root_model if parent is None else parent
        for m in models:
            if m.parent is None or m.parent is not d:
                d.add_drawing(m)

        # Handle already added models being moved to a new parent.
        for model in models:
            if model.id and model.id[:-1] != d.id and model.id in self._models:
                # Model has id that is not a subid of parent, so assign new id.
                _need_fire_id_trigger.append(model)
                del self._models[model.id]
                model.id = None
                if model.parent is not None:
                    model.parent._next_unused_id = None

        # Assign new model ids
        for model in models:
            if model.id is None:
                model.id = self.next_id(parent = d, minimum_id = minimum_id)
            self._models[model.id] = model
            children = model.child_models()
            if children:
                self.add(children, model, _notify=False, _need_fire_id_trigger=_need_fire_id_trigger)

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
        if _notify and not _from_session and start_count == 0 and len(self._models) > 0:
            v = session.main_view
            v.initial_camera_view()
            v.clip_planes.clear()   # Turn off clipping

    def assign_id(self, model, id):
        '''Parent model for new id must already exist.'''
        mt = self._models
        del mt[model.id]
        model.id = id
        mt[id] = model
        p = mt[id[:-1]] if len(id) > 1 else self.scene_root_model
        p._next_unused_id = None
        self.add([model], parent = p)

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

    def add_group(self, models, name=None, id=None):
        if name is None:
            names = set([m.name for m in models])
            if len(names) == 1:
                name = names.pop() + " group"
            else:
                name = "group"
        parent = Model(name, self._session())
        if id is not None:
            parent.id = id
        parent.add(models)
        self.add([parent])
        return parent

    def remove(self, models):
        # Also remove all child models, and remove deepest children first.
        dset = descendant_models(models)
        dset.update(models)
        mlist = list(dset)
        mlist.sort(key=lambda m: len(m.id), reverse=True)
        session = self._session()  # resolve back reference
        for m in mlist:
            m._added_to_session = False
            m.removed_from_session(session)
        for model in mlist:
            model_id = model.id
            if model_id is not None:
                del self._models[model_id]
                model.id = None
                if len(model_id) == 1:
                    parent = self.scene_root_model
                else:
                    parent = self._models[model_id[:-1]]
                parent.remove_drawing(model, delete=False)
                parent._next_unused_id = None

        # it's nice to have an accurate list of current models
        # when firing this trigger, so do it last
        session.triggers.activate_trigger(REMOVE_MODELS, mlist)

        return mlist

    def close(self, models):
        # Removed models include children of specified models.
        mremoved = self.remove(models)
        for m in mremoved:
            m.delete()

    def open(self, filenames, id=None, format=None, name=None, **kw):
        from . import io, toolshed
        session = self._session()  # resolve back reference
        collation_okay = True
        if isinstance(filenames, str):
            fns = [filenames]
        else:
            fns = filenames
        for fn in fns:
            fmt = io.deduce_format(fn, has_format=format)[0]
            if fmt and fmt.category in [toolshed.SCRIPT]:
                collation_okay = False
                break
        from .logger import Collator
        log_errors = kw.pop('log_errors', True)
        if collation_okay:
            descript = "files" if len(fns) > 1 else fns[0]
            with Collator(session.logger,
                    "Summary of feedback from opening " + descript, log_errors):
                models, status = io.open_multiple_data(
                    session, filenames, format=format, name=name, **kw)
        else:
            models, status = io.open_multiple_data(
                session, filenames, format=format, name=name, **kw)
        if status:
            log = session.logger
            log.status(status, log=True)
        if models:
            if len(models) > 1:
                from . import io
                name = io.model_name_from_path(filenames[0])
                if len(filenames) > 1:
                    name += '...'
                self.add_group(models, name=name)
            else:
                self.add(models)
        return models


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
