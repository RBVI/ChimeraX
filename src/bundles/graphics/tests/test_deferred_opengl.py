"""Construction and cleanup must work before the first 3D window is shown."""

from types import SimpleNamespace


def test_renderer_can_be_constructed_and_deleted_without_pyopengl(monkeypatch):
    from chimerax.graphics import opengl

    def unexpected_import(*args, **kw):
        raise AssertionError("PyOpenGL loaded before a context was used")

    monkeypatch.setattr(opengl, "GL", None)
    monkeypatch.setattr(opengl, "_initialize_pyopengl", unexpected_import)
    context = opengl.OpenGLContext(None, None)
    render = opengl.Render(context)
    try:
        assert not context.created
        assert render.default_framebuffer()._draw_buffer == 0x0405
    finally:
        render.delete()
        context.delete()


def test_first_make_current_initializes_bindings_before_using_context(monkeypatch):
    from chimerax.graphics import opengl

    events = []
    monkeypatch.setattr(opengl, "_initialize_pyopengl",
                        lambda: events.append("bindings"))
    window = object()
    context = opengl.OpenGLContext(window, None)

    def make_current(surface):
        assert surface is window
        assert events == ["bindings"]
        events.append("current")
        return True

    context._contexts["mono"] = SimpleNamespace(makeCurrent=make_current)
    monkeypatch.setattr(context, "_check_thread", lambda: None)
    try:
        assert context.make_current()
        assert events == ["bindings", "current"]
    finally:
        context._contexts.clear()
        context.delete()
