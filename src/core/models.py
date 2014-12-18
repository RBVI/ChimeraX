# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
models: model support
=====================

TODO: Stubs for now.

"""

import weakref
ADD_MODELS = 'add models'
REMOVE_MODELS = 'remove models'


# TODO: register Model as data event type
class Model:

    def __init__(self):
        self.id = None
        self.name = "unknown"
        self.graphics = None
        # TODO: track.created(Model, [self])

    # def save(self):
    #    raise NotImplemented

    # def restore(self):
    #    raise NotImplemented

    # def export(self):
    #    raise NotImplemented

    def make_graphics(self, parent_drawing):
        raise NotImplemented

    def destroy(self):
        if self.id is not None:
            raise ValueError("model is still open")
        if self.graphics:
            self.graphics.delete()
            self.graphics = None
        # TODO: track.deleted(Model, [self])


class Models:

    def __init__(self, session):
        self._session = weakref.ref(session)
        session.triggers.add_trigger(ADD_MODELS)
        session.triggers.add_trigger(REMOVE_MODELS)
        self._models = {}

        # TODO: malloc-ish management of model ids, so they may be reused
        from itertools import count as _count
        self._id_counter = _count(1)

    def list(self):
        return self._models

    def add(self, models):
        session = self._session()  # resolve back reference
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
        session = self._session()  # resolve back reference
        models, status = io.open(session, filename)
        if status:
            session.logger.status(status)
        if models:
            start_count = len(self._models)
            for model in models:
                model.id = next(self._id_counter)
                if session.main_drawing:
                    model.make_graphics(session.main_drawing)
            self.add(models)
            if start_count == 0 and len(self._models) > 0:
                session.view.inital_camera_view()

    def close(self, model_id):
        if model_id in self._models:
            model = self._models[model_id]
            self.remove(model)
            model.destroy()
