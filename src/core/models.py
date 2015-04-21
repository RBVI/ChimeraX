# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
models: model support
=====================

"""

import weakref
from .graphics.drawing import Drawing
from .session import State
ADD_MODELS = 'add models'
ADD_MODEL_GROUP = 'add model group'
REMOVE_MODELS = 'remove models'
# TODO: register Model as data event type


class Model(State, Drawing):
    """All models are drawings.

    That means that regardless of whether or not there is a GUI,
    each model maintains its geometry.

    Every model subclass that can be in a session file, needs to be
    registered.
    """

    MODEL_STATE_VERSION = 1

    def __init__(self, name):
        Drawing.__init__(self, name)
        self.id = None  # tuple: e.g., 1.2.1 is (1, 2, 1)
        # TODO: track.created(Model, [self])

    def delete(self):
        if self.id is not None:
            raise ValueError("model is still open")
        Drawing.delete(self)
        # TODO: track.deleted(Model, [self])

    def take_snapshot(self, session, flags):
        return [self.MODEL_STATE_VERSION, self.name]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.MODEL_STATE_VERSION:
            raise RuntimeError("Unexpected version or data")
        self.name = data

    def reset_state(self):
        pass

    def selected_items(self, itype):
        return ()

    def anything_selected(self):
        return False

class Models(State):

    VERSION = 1     # snapshot version

    def __init__(self, session):
        self._session = weakref.ref(session)
        session.triggers.add_trigger(ADD_MODELS)
        session.triggers.add_trigger(ADD_MODEL_GROUP)
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
                session.restore_unique_id(model, uid)
            else:
                model = session.unique_obj(uid)
                if len(model.id) == 1:
                    parent = self.drawing
                else:
                    parent = self._models[model.id[:-1]]
                parent.add_drawing(model)
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
        if id is not None:
            base_model_id = id
        else:
            base_model_id = (next(self._id_counter), )  # model id's are tuples
        multi_model = len(models) > 1
        if not multi_model:
            parent = self.drawing
        else:
            parent = Model('container')  # TODO: replace with appropriate name
            parent.id = base_model_id
            self._models[parent.id] = parent
            self.drawing.add_drawing(parent)
            from itertools import count as count
            counter = count(1)
        for model in models:
            if not multi_model:
                model.id = base_model_id
            else:
                model.id = base_model_id + (next(counter),)
            self._models[model.id] = model
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
            if len(model_id) == 1:
                parent = self.drawing
            else:
                parent = self._models[model_id[:-1]]
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
        return models

    def close(self, model_id):
        if model_id not in self._models:
            return
        # find all submodels
        size = len(model_id)
        model_ids = [x for x in self._models if x[0:size] == model_id]
        # sort so submodels are removed before parent models
        model_ids.sort(key=len, reverse=True)
        models = [self._models[x] for x in model_ids]
        self.remove(models)
        for m in models:
            m.delete()
