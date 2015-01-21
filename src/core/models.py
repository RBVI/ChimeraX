# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
models: model support
=====================

TODO: Stubs for now.

"""

import weakref
from .graphics.drawing import Drawing
from .session import State
ADD_MODELS = 'add models'
REMOVE_MODELS = 'remove models'
# TODO: register Model as data event type


class Model(State, Drawing):
    """All models are drawings.

    That means that regardless of whether or not there is a GUI,
    each model maintains its geometry.

    Every model subclass that can be in a session file, needs to be
    registered.
    """

    def __init__(self, name):
        Drawing.__init__(self, name)
        self.id = None
        # TODO: track.created(Model, [self])

    def delete(self):
        if self.id is not None:
            raise ValueError("model is still open")
        Drawing.delete(self)
        # TODO: track.deleted(Model, [self])


class Models(State):

    VERSION = 1     # snapshot version

    def __init__(self, session):
        self._session = weakref.ref(session)
        session.triggers.add_trigger(ADD_MODELS)
        session.triggers.add_trigger(REMOVE_MODELS)
        self._models = {}
        from .graphics.drawing import Drawing
        self.drawing = Drawing("root")

        # TODO: malloc-ish management of model ids, so they may be reused
        from itertools import count as _count
        self._id_counter = _count(1)

    def take_snapshot(self, session, flags):
        data = {}
        for id, model in self._models.items():
            assert(isinstance(model, Model))
            data[id] = [session.unique_id(model),
                        model.take_snapshot(session, flags)]
        return [self.VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.VERSION:
            raise RuntimeError("Unexpected version")

        for id, [uid, [model_version, model_data]] in data.items():
            if phase == State.PHASE1:
                try:
                    cls = session.class_of_unique_id(uid, Model)
                except KeyError:
                    session.log.warning(
                        'Unable to restore model %s (%s)'
                        % (id, session.class_name_of_unique_id(uid)))
                    continue
                model = cls("unknown name until restored")
                model.id = id
                self._models[id] = model
                parent = self.drawing   # TODO: figure out based on id
                parent.add_drawing(model)
                session.restore_unique_id(model, uid)
            else:
                model = session.unique_obj(uid)
            model.restore_snapshot(phase, session, model_version, model_data)

    def reset_state(self):
        models = self._models.values()
        self._models.clear()
        for model in models:
            model.delete()

    def list(self):
        return list(self._models.values())

    def add(self, models, id=None):
        session = self._session()  # resolve back reference
        for model in models:
            # TODO:
            # if id is not None:
            #     model.id = id
            # else:
            if 1:
                model.id = next(self._id_counter)
            self._models[model.id] = model
            parent = self.drawing   # TODO: figure out based on id
            parent.add_drawing(model)
        session.triggers.activate_trigger(ADD_MODELS, models)

    def remove(self, models):
        session = self._session()  # resolve back reference
        session.triggers.activate_trigger(REMOVE_MODELS, models)
        for model in models:
            model_id = model.id
            if model_id is None:
                continue
            model.id = None
            del self._models[model_id]
            parent = self.drawing   # TODO: figure out based on id
            parent.remove_drawing(model)

    def open(self, filename, id=None, **kw):
        from . import io
        session = self._session()  # resolve back reference
        models, status = io.open(session, filename, **kw)
        if status:
            session.logger.status(status)
        if models:
            start_count = len(self._models)
            self.add(models, id=id)
            if start_count == 0 and len(self._models) > 0:
                session.main_view.initial_camera_view()

    def close(self, model_id):
        if model_id in self._models:
            model = self._models[model_id]
            self.remove(model)
            model.delete()
