from . import io
from . import models
from .session import State

CATEGORY = io.GENERIC3D


class Generic3DModel(models.Model):
    """Commom base class for generic 3D data"""

    # TODO: just save/restore drawing state

    def take_snapshot(self, session, flags):
        State.take_snapshot(self, session, flags)

    def restore_snapshot(self, phase, session, version, data):
        State.restore_snapshot(self, phase, session, version, data)

    def reset_state(self):
        State.reset_state(self)
