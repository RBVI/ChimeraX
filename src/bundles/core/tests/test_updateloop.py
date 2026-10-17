from types import SimpleNamespace

from chimerax.core.updateloop import UpdateLoop
from chimerax.graphics.vulkan import VulkanDrawableUnavailable


class _Triggers:
    def __init__(self):
        self.names = []

    def activate_trigger(self, name, data):
        self.names.append(name)


class _View:
    def __init__(self, draw_error=None):
        self.camera = SimpleNamespace(always_draw=False)
        self.frame_number = 0
        self.redraw_needed = True
        self._draw_error = draw_error

    def check_for_drawing_change(self):
        return self.redraw_needed

    def draw(self, check_for_changes=False):
        self.redraw_needed = False
        if self._draw_error is not None:
            raise self._draw_error


def _session(view):
    graphics_window = SimpleNamespace(is_drawable=True)
    return SimpleNamespace(
        main_view=view,
        triggers=_Triggers(),
        ui=SimpleNamespace(
            is_gui=True,
            main_window=SimpleNamespace(graphics_window=graphics_window),
        ),
        logger=SimpleNamespace(error=lambda message: None, bug=lambda message: None),
    )


def test_temporarily_unavailable_vulkan_drawable_is_retried(monkeypatch):
    import chimerax.atomic
    import chimerax.surface

    monkeypatch.setattr(chimerax.atomic, "check_for_changes", lambda session: None)
    monkeypatch.setattr(chimerax.surface, "update_clip_caps", lambda view: None)
    view = _View(VulkanDrawableUnavailable("temporarily unavailable"))
    session = _session(view)

    assert not UpdateLoop(session).draw_new_frame()
    assert view.redraw_needed
    assert "frame drawn" not in session.triggers.names


def test_successful_draw_emits_frame_drawn(monkeypatch):
    import chimerax.atomic
    import chimerax.surface

    monkeypatch.setattr(chimerax.atomic, "check_for_changes", lambda session: None)
    monkeypatch.setattr(chimerax.surface, "update_clip_caps", lambda view: None)
    view = _View()
    session = _session(view)

    assert UpdateLoop(session).draw_new_frame()
    assert "frame drawn" in session.triggers.names
