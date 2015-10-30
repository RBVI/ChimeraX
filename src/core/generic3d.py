# vim: set expandtab shiftwidth=4 softtabstop=4:
from . import io
from .models import Model
from .graphics.gsession import DrawingState
from .state import CORE_STATE_VERSION

CATEGORY = io.GENERIC3D


class Generic3DModel(Model):
    """Commom base class for generic 3D data"""

    def take_snapshot(self, session, flags):
        draw_state = DrawingState(self)
        return CORE_STATE_VERSION, [
            Model.take_snapshot(self, session, flags),
            draw_state.take_snapshot(session, flags)
        ]

    def restore_snapshot_init(self, session, tool_info, version, data):
        (model_version, model_data), (drawing_version, drawing_data) = data
        Model.restore_snapshot_init(
            self, session, tool_info, model_version, model_data)
        draw_state = DrawingState(self)
        draw_state.restore_snapshot_init(
            session, tool_info, drawing_version, drawing_data)

    def reset_state(self, session):
        pass
