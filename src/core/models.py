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
from .state import State, CORE_STATE_VERSION
ADD_MODELS = 'add models'
REMOVE_MODELS = 'remove models'
MODEL_DISPLAY_CHANGED = 'model display changed'
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
    bundle_info : a :py:class:`~chimerax.core.toolshed.BundleInfo` instance
        The tool that provides the subclass.
    SESSION_ENDURING : bool, class-level optional
        If True, then model survives across sessions.
    SESSION_SKIP : bool, class-level optional
        If True, then model is not saved in sessions.
    """

    SESSION_ENDURING = False
    SESSION_SKIP = False
    bundle_info = None    # default, should be set in subclass

    def __init__(self, name, session):
        Drawing.__init__(self, name)
        self.session = session
        self.id = None
        self._deleted = False
        # TODO: track.created(Model, [self])

    def delete(self):
        self._deleted = True
        Drawing.delete(self)
        delattr(self, "session")

    @property
    def deleted(self):
        # may be overriden in subclass, e.g. Structure
        return self._deleted

    def id_string(self):
        if self.id is None:
            return ''
        return '.'.join(str(i) for i in self.id)

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
        if self.id is None:
            for m in models:
                self.add_drawing(m)
        else:
            self.session.models.add(models, parent = self)

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
            p = getattr(self, 'parent', None)
            return p is None or p.visible
        return False
    
    def _set_display(self, display):
        Drawing.set_display(self, display)
        self.session.triggers.activate_trigger(MODEL_DISPLAY_CHANGED, self)
    display = Drawing.display.setter(_set_display)

    def take_snapshot(self, session, flags):
        p = getattr(self, 'parent', None)
        if p is session.models.drawing:
            p = None    # Don't include root as a parent since root is not saved.
        data = {
            'name': self.name,
            'id': self.id,
            'parent': p,
            'positions': self.positions.array(),
            'version': CORE_STATE_VERSION,
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        if cls is Model and data['id'] is ():
            return session.models.drawing
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
        from .geometry import Places
        self.positions = Places(place_array=data['positions'])

    def reset_state(self, session):
        pass

    def selected_items(self, itype):
        return []

    def added_to_session(self, session):
        pass

    def removed_from_session(self, session):
        pass

    # Atom specifier API
    def atomspec_has_atoms(self):
        # Return True if there are atoms in this model
        return False


class Models(State):

    def __init__(self, session):
        self._session = weakref.ref(session)
        t = session.triggers
        t.add_trigger(ADD_MODELS)
        t.add_trigger(REMOVE_MODELS)
        t.add_trigger(MODEL_DISPLAY_CHANGED)
        self._models = {}
        self.drawing = r = Model("root", session)
        r.id = ()

    def take_snapshot(self, session, flags):
        models = {}
        for id, model in self._models.items():
            assert(isinstance(model, Model))
            if model.SESSION_SKIP:
                continue
            models[id] = model
        data = {'models': models,
                'version': CORE_STATE_VERSION}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        m = session.models
        for id, model in data['models'].items():
            if model:        # model can be None if it could not be restored, eg Volume w/o map file
                if not hasattr(model, 'parent'):
                    m.add([model], _from_session=True)
        return m

    def reset_state(self, session):
        self.remove([m for m in self.list() if not m.SESSION_ENDURING])

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

    def add(self, models, parent=None, _notify=True, _from_session=False):
        start_count = len(self._models)

        d = self.drawing if parent is None else parent
        for m in models:
            if not hasattr(m, 'parent') or m.parent is not d:
                d.add_drawing(m)

        # Assign id numbers
        m_all = list(models)
        for model in models:
            if model.id is None:
                model.id = self._next_child_id(d)
            self._models[model.id] = model
            children = model.child_models()
            if children:
                m_all.extend(self.add(children, model, _notify=False))

        if _notify:
            session = self._session()
            for m in m_all:
                m.added_to_session(session)
            session.triggers.activate_trigger(ADD_MODELS, m_all)
            if not _from_session and start_count == 0 and len(self._models) > 0:
                v = session.main_view
                v.initial_camera_view()
                v.clip_planes.clear()   # Turn off clipping

        return m_all

    def __getitem__(self, i):
        '''index into models using square brackets (e.g. session.models[i])'''
        return list(self._models.values())[i]

    def __iter__(self):
        '''iterator over models'''
        return iter(self._models.values())

    def __len__(self):
        '''number of models'''
        return len(self._models)

    def _next_child_id(self, parent):
        # Find lowest unused id.  Typically all ids 1,...,N are used with no gaps
        # and then it is fast to assign N+1 to the next model.  But if there are
        # gaps it can take O(N**2) time to figure out ids to assign for N models.
        # This code handles the common case of no gaps quickly.
        nid = getattr(parent, '_next_unused_id', None)
        if nid is None:
            # Find next unused id.
            cids = set(m.id[-1] for m in parent.child_models() if m.id is not None)
            for nid in range(1, len(cids) + 2):
                if nid not in cids:
                    break
            if nid == len(cids) + 1:
                parent._next_unused_id = nid + 1        # No gaps in ids
        else:
            parent._next_unused_id = nid + 1            # No gaps in ids
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
        m_all = self.add([parent])
        return [parent] + m_all

    def remove(self, models):
        # Also remove all child models, and remove deepest children first.
        dset = descendant_models(models)
        dset.update(models)
        mlist = list(dset)
        mlist.sort(key=lambda m: len(m.id), reverse=True)
        session = self._session()  # resolve back reference
        for m in mlist:
            m.removed_from_session(session)
        for model in mlist:
            model_id = model.id
            if model_id is not None:
                del self._models[model_id]
                model.id = None
                if len(model_id) == 1:
                    parent = self.drawing
                else:
                    parent = self._models[model_id[:-1]]
                parent.remove_drawing(model, delete=False)
                parent._next_unused_id = None

        # it's nice to have an accurate list of current models
        # when firing this trigger, so do it last
        session.triggers.activate_trigger(REMOVE_MODELS, mlist)

    def close(self, models):
        self.remove(models)
        for m in models:
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
                from os.path import basename
                name = basename(filenames[0])
                if len(filenames) > 1:
                    name += '...'
                descript = "files" if len(fns) > 1 else fns[0]
                with Collator(session.logger,
                        "Summary of additional actions taken when opening " + descript, log_errors):
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
        if hasattr(m, 'parent'):
            p = m.parent
            if p not in mset:
                ma.add(p)
    if ma:
        ma.update(ancestor_models(ma))
    return ma
