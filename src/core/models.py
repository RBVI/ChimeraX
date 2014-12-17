# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
models: model support
=====================

TODO: Stubs for now.

"""

import weakref


class Model:
    pass


class Models:

    def __init__(self, session):
        self._session = weakref.ref(session)
        self._models = {}

    def list(self):
        return self._models

    def close(self, model_id):
        if model_id in self._models:
            del self._models[model_id]
