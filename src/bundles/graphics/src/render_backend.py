"""
Render backend registry and base class.

Backends are small adapters that let View delegate actual GPU interaction to
either the existing OpenGL path or a new Metal path.  New backends register
themselves by calling register_backend(); View queries the registry via
switch_backend() and active_backend().

Protocol
--------
A backend class must implement:

    name : str  (class attribute)
    initialize(view, window_id, width, height) -> bool
    delete()
    make_current() -> bool
    done_current()
    resize(width, height)
    render(drawing, camera)
    swap_buffers()
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .view import View
    from .drawing import Drawing
    from .camera import Camera

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_registry: dict[str, tuple] = {}   # name -> (class, session)
_active_backends: dict[int, object] = {}  # id(session) -> backend instance


def register_backend(name: str, backend_class, session) -> None:
    """Register a backend class under *name* for the given session."""
    _registry[name] = (backend_class, session)


def switch_backend(name: str, session) -> None:
    """
    Switch session.main_view to the named backend.

    If the named backend is already active this is a no-op.  Raises
    ValueError if the name is not registered (for 'opengl' a built-in
    OpenGLBackend is always available).
    """
    sid = id(session)
    current = _active_backends.get(sid)
    if current is not None and current.name == name:
        return

    if name not in _registry and name != "opengl":
        raise ValueError(
            f"No render backend named {name!r} is registered.  "
            f"Available: opengl, {list(_registry)}"
        )

    if current is not None:
        try:
            current.delete()
        except Exception:
            pass

    if name == "opengl":
        backend = OpenGLBackend(session)
    else:
        cls, _ = _registry[name]
        backend = cls(session)

    _active_backends[sid] = backend

    # Tell the View to re-initialize rendering through the new backend.
    view = getattr(session, "main_view", None)
    if view is not None:
        view._render_backend = backend
        # Re-render to apply the switch immediately.
        view.redraw_needed = True


def active_backend(session) -> object | None:
    """Return the currently active backend for *session*, or None."""
    return _active_backends.get(id(session))


# --------------------------------------------------------------------------
# OpenGL backend (thin wrapper around the existing Render path)
# --------------------------------------------------------------------------

class OpenGLBackend:
    """
    Delegates to the existing chimerax.graphics.opengl.Render.

    This wrapper is never constructed explicitly by user code; View creates
    it automatically when no other backend is active.
    """

    name = "opengl"

    def __init__(self, session):
        self._session = session
        self._render = None   # chimerax.graphics.opengl.Render

    def initialize(self, view, window_id: int, width: int, height: int) -> bool:
        # The existing OpenGL path is already handled by View.initialize_rendering.
        # We just record the Render object View created.
        self._render = view.render
        return self._render is not None

    def delete(self):
        if self._render is not None:
            self._render.delete()
            self._render = None

    def make_current(self) -> bool:
        if self._render is None:
            return False
        return self._render.make_current()

    def done_current(self):
        if self._render is not None:
            self._render.done_current()

    def resize(self, width: int, height: int):
        pass  # View handles this internally for OpenGL

    def render(self, drawing, camera):
        pass  # View calls its existing draw() path, not this method

    def swap_buffers(self):
        if self._render is not None:
            self._render.swap_buffers()
