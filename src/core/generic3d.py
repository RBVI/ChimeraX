# vim: set expandtab shiftwidth=4 softtabstop=4:
from .models import Model

from . import io
CATEGORY = io.GENERIC3D


class Generic3DModel(Model):
    """Commom base class for generic 3D data"""

    def take_snapshot(self, session, flags):
        from .state import CORE_STATE_VERSION
        from .graphics.gsession import DrawingState
        data = {
            'model state': Model.take_snapshot(self, session, flags),
            'drawing state': DrawingState(self).take_snapshot(session, flags)
        }
        return CORE_STATE_VERSION, data

    @classmethod
    def restore_snapshot(cls, session, bundle_info, version, data):
        m = cls('name', session)
        model_version, model_data = data['model state']
        m.set_state_from_snapshot(model_version, model_data)
        drawing_version, drawing_data = data['drawing state']
        from .graphics.gsession import DrawingState
        DrawingState(m).set_state_from_snapshot(session, bundle_info, drawing_version, drawing_data)
        return m

    def reset_state(self, session):
        pass
