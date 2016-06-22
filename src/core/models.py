# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
models: Displayed data
======================

"""

import weakref
from .graphics.drawing import Drawing
from .state import State, CORE_STATE_VERSION
ADD_MODELS = 'add models'
REMOVE_MODELS = 'remove models'
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
        # TODO: track.created(Model, [self])

    def delete(self):
        Drawing.delete(self)
        delattr(self, "session")

    def id_string(self):
        return '.'.join(str(i) for i in self.id)

    def _get_single_color(self):
        return None
    def _set_single_color(self, color):
        return
    single_color = property(_get_single_color, _set_single_color)
    '''
    Getting the single color may give the dominant color.
    Setting the single color will set the model to that color.
    Color values are rgba uint8 arrays.
    '''

    def add(self, models):
        for m in models:
            self.add_drawing(m)

    def child_models(self):
        '''Return all models including self and children at all levels.'''
        return [d for d in self.child_drawings() if isinstance(d, Model)]

    def all_models(self):
        '''Return all models including self and children at all levels.'''
        dlist = [self]
        for d in self.child_drawings():
            if isinstance(d, Model):
                dlist.extend(d.all_models())
        return dlist

    def take_snapshot(self, session, flags):
        p = getattr(self, 'parent', None)
        if p is session.models.drawing:
            p = None	# Don't include root as a parent since root is not saved.
        data = {'name':self.name,
                'id':self.id,
                'parent':p,
                'positions':self.positions.array(),
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
        self.positions = Places(place_array = data['positions'])

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
        session.triggers.add_trigger(ADD_MODELS)
        session.triggers.add_trigger(REMOVE_MODELS)
        self._models = {}
        from .graphics.drawing import Drawing
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
            if model_id not in self._models:
                return []
            # find all submodels
            size = len(model_id)
            model_ids = [x for x in self._models if x[0:size] == model_id]
            # sort so submodels are removed before parent models
            model_ids.sort(key=len, reverse=True)
            models = [self._models[x] for x in model_ids]
        if not type is None:
            models = [m for m in models if isinstance(m,type)]
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
                v.clip_planes.clear()	# Turn off clipping

        return m_all

    def _next_child_id(self, parent):
        # Find lowest unused id.  Typically all ids 1,...,N are used with no gaps
        # and then it is fast to assign N+1 to the next model.  But if there are
        # gaps it can take O(N**2) time to figure out ids to assign for N models.
        # This code handles the common case of no gaps quickly.
        nid = getattr(parent, '_next_unused_id', None)
        if nid is None:
            # Find next unused id.
            cids = set(m.id[-1] for m in parent.child_models() if not m.id is None)
            for nid in range(1,len(cids)+2):
                if not nid in cids:
                    break
            if nid == len(cids)+1:
                parent._next_unused_id = nid + 1	# No gaps in ids
        else:
            parent._next_unused_id = nid + 1		# No gaps in ids
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
        mlist = descendant_models(models)
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
                parent.remove_drawing(model, delete = False)
                parent._next_unused_id = None

        # it's nice to have an accurate list of current models
        # when firing this trigger, so do it last
        session.triggers.activate_trigger(REMOVE_MODELS, mlist)

    def close(self, models):
        self.remove(models)
        for m in models:
            m.delete()

    def open(self, filenames, id=None, format=None, name=None, **kw):
        from . import io
        session = self._session()  # resolve back reference
        collation_okay = True
        if isinstance(filenames, str):
            fns = [filenames]
        else:
            fns = filenames
        for fn in fns:
            if io.category(io.deduce_format(fn, has_format=format)[0]) == io.SCRIPT:
                collation_okay = False
                break
        if collation_okay:
            from .logger import Collator
            descript = "files" if len(fns) > 1 else fns[0]
            with Collator(session.logger, "Summary of problems opening " + descript,
                                                    kw.pop('log_errors', True)):
                models, status = io.open_multiple_data(session, filenames,
                format=format, name=name, **kw)
        else:
            models, status = io.open_multiple_data(session, filenames,
                format=format, name=name, **kw)
        if status:
            log = session.logger
            log.status(status, log=True)
        if models:
            if len(models) > 1:
                self.add_group(models)
            else:
                self.add(models)
        return models


def descendant_models(models):
    mset = set()
    for m in models:
        mset.update(m.all_models())
    return list(mset)

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
