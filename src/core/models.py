# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
models: model support
=====================

TODO: Stubs for now.

"""

import weakref
from .graphics.drawing import Drawing
from . import session
ADD_MODELS = 'add models'
REMOVE_MODELS = 'remove models'
# TODO: register Model as data event type


class Model(session.State, Drawing):
    """All models are drawings.  That means that regardless of whether or not
    there is a GUI, each model maintains its geometry.
    """

    def __init__(self, name):
        Drawing.__init__(self, name)
        self.id = None
        # self.name = "unknown"
        # TODO: track.created(Model, [self])

    def destroy(self):
        if self.id is not None:
            raise ValueError("model is still open")
        self.delete()
        # TODO: track.deleted(Model, [self])


class Models(session.State):

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

    def take_snapshot(self, session, flags=session.State.ALL):
        version = 0
        data = {}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        if len(data) != 2 or data[0] != 0 or data[1]:
            raise RuntimeError("Unexpected version or data")

    def reset_state(self):
        pass

    def list(self):
        return self._models

    def add(self, models):
        session = self._session()  # resolve back reference
        for model in models:
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
            if model_id is not None:
                model.id = None
                del self._models[model_id]

    def open(self, filename, id=None):
        from . import io
        print('id:', id)
        session = self._session()  # resolve back reference
        models, status = io.open(session, filename)
        if status:
            session.logger.status(status)
        if models:
            start_count = len(self._models)
            self.add(models)
            if start_count == 0 and len(self._models) > 0:
                session.main_view.initial_camera_view()

    def close(self, model_id):
        if model_id in self._models:
            model = self._models[model_id]
            self.remove(model)
            model.destroy()
