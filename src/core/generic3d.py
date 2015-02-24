# vi: set expandtab shiftwidth=4 softtabstop=4:
from . import io
from . import models
from .graphics.drawing import Drawing

CATEGORY = io.GENERIC3D


class Generic3DModel(models.Model):
    """Commom base class for generic 3D data"""

    def take_snapshot(self, session, flags):
        return Drawing.take_snapshot(self, session, flags)

    def restore_snapshot(self, phase, session, version, data):
        return Drawing.restore_snapshot(self, phase, session, version, data)

    def reset_state(self):
        pass
