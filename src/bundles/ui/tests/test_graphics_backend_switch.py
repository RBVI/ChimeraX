from types import SimpleNamespace


class _UpdateLoop:
    def __init__(self):
        self.calls = []

    def block_redraw(self):
        self.calls.append("block")

    def unblock_redraw(self):
        self.calls.append("unblock")

    def update_graphics_now(self):
        self.calls.append("update")


class _Triggers:
    def __init__(self):
        self.events = []

    def activate_trigger(self, name, data):
        self.events.append((name, data))


def _window(install):
    update_loop = _UpdateLoop()
    triggers = _Triggers()
    calls = []
    window = SimpleNamespace(
        render_endpoint=SimpleNamespace(backend_name="opengl"),
        opengl_context=SimpleNamespace(backend_name="opengl"),
        view=SimpleNamespace(redraw_needed=False),
        session=SimpleNamespace(update_loop=update_loop, triggers=triggers),
        isVisible=lambda: True,
        show=lambda: calls.append("show"),
        _release_graphics_backend=lambda: calls.append("release"),
        _recreate_native_window=lambda backend, make_visible=False: calls.append(
            ("recreate", backend, make_visible)
        ),
        _install_graphics_backend=install,
    )
    return window, update_loop, triggers, calls


def test_backend_switch_recreates_native_window_before_install():
    from chimerax.ui.graphics import GraphicsWindow

    installed = []
    window, update_loop, triggers, calls = _window(installed.append)

    GraphicsWindow.switch_backend(window, "vulkan")

    assert calls == ["release", ("recreate", "vulkan", True), "show"]
    assert installed == ["vulkan"]
    assert update_loop.calls == ["block", "unblock"]
    assert window.view.redraw_needed
    assert triggers.events[-1][1] == {"backend": "vulkan"}


def test_failed_backend_switch_restores_previous_backend():
    from chimerax.ui.graphics import GraphicsWindow

    installed = []

    def install(backend):
        installed.append(backend)
        if backend == "vulkan":
            raise RuntimeError("no swapchain")

    window, update_loop, triggers, calls = _window(install)

    try:
        GraphicsWindow.switch_backend(window, "vulkan")
    except RuntimeError as error:
        assert str(error) == "no swapchain"
    else:
        raise AssertionError("failed backend switch did not propagate its error")

    assert installed == ["vulkan", "opengl"]
    assert calls == [
        "release",
        ("recreate", "vulkan", True),
        "release",
        ("recreate", "opengl", True),
        "show",
    ]
    assert update_loop.calls == ["block", "unblock"]
    assert triggers.events[-1][1] == {"backend": "opengl"}


def _native_window(first_show):
    trackpad_windows = []
    calls = []
    window = SimpleNamespace(
        destroy=lambda: calls.append("destroy"),
        show=lambda: calls.append("show"),
        session=SimpleNamespace(ui=SimpleNamespace(mouse_modes=SimpleNamespace(
            trackpad=SimpleNamespace(set_graphics_window=trackpad_windows.append)))),
    )
    if first_show:
        window._first_show = True
    return window, calls, trackpad_windows


def _stub_backend_registry(monkeypatch, calls):
    from chimerax.graphics import ral
    backend = SimpleNamespace(
        configure_surface=lambda window, ui: calls.append("configure"))
    monkeypatch.setattr(ral.backend_registry, "get", lambda name: backend)


def test_recreated_native_window_reenables_trackpad_touch(monkeypatch):
    # destroy() replaces the macOS NSView on which touch delivery was enabled,
    # so trackpad gestures stopped working after any backend switch.
    from chimerax.ui.graphics import GraphicsWindow

    window, calls, trackpad_windows = _native_window(first_show=True)
    _stub_backend_registry(monkeypatch, calls)

    GraphicsWindow._recreate_native_window(window, "vulkan")

    assert calls == ["destroy", "configure"]
    assert trackpad_windows == [window]


def test_native_window_before_first_show_leaves_trackpad_to_show_event(monkeypatch):
    from chimerax.ui.graphics import GraphicsWindow

    window, calls, trackpad_windows = _native_window(first_show=False)
    _stub_backend_registry(monkeypatch, calls)

    GraphicsWindow._recreate_native_window(window, "vulkan")

    assert trackpad_windows == []


def _exposable_window(backend, exposed=True):
    return SimpleNamespace(
        isExposed=lambda: exposed,
        render_endpoint=SimpleNamespace(backend_name=backend),
        view=SimpleNamespace(redraw_needed=False),
    )


def _expose(window):
    from chimerax.ui.graphics import GraphicsWindow
    window._layer_keeps_contents = lambda: GraphicsWindow._layer_keeps_contents(window)
    GraphicsWindow.exposeEvent(window, None)
    redraw = window.view.redraw_needed
    window.view.redraw_needed = False
    return redraw


def test_vulkan_expose_after_present_does_not_redraw_on_mac(monkeypatch):
    # MoltenVK marks the layer as needing display on every present, so Qt
    # sends an Expose after every frame; redrawing for each one made Vulkan
    # render continuously while nothing changed.
    import sys
    monkeypatch.setattr(sys, "platform", "darwin")
    window = _exposable_window("vulkan")

    assert _expose(window)          # newly exposed window draws
    assert not _expose(window)      # Expose caused by its own present
    assert not _expose(window)


def test_vulkan_window_redraws_when_exposed_again(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "platform", "darwin")
    window = _exposable_window("vulkan")
    _expose(window)
    window.isExposed = lambda: False
    _expose(window)
    window.isExposed = lambda: True

    assert _expose(window)


def test_opengl_and_other_platforms_redraw_on_every_expose(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "platform", "darwin")
    window = _exposable_window("opengl")
    assert _expose(window) and _expose(window)

    monkeypatch.setattr(sys, "platform", "linux")
    window = _exposable_window("vulkan")
    assert _expose(window) and _expose(window)
