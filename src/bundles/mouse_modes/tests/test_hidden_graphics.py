from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("window", [None, SimpleNamespace(isVisible=lambda: False)])
def test_mouse_tracking_before_graphics_is_shown(window):
    from chimerax.mouse_modes.mousemodes import MouseModes

    # No cursor mapping or session access is valid until the window is shown.
    modes = SimpleNamespace(graphics_window=window)
    assert MouseModes._graphics_cursor_position(modes) is None
