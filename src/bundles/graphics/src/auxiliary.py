# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California.
# All rights reserved. This software provided pursuant to a license agreement
# containing restrictions on its disclosure, duplication, and use.
# === UCSF ChimeraX Copyright ===

"""Small helpers for QWindows that render with the main graphics context.

This module intentionally does not own tool UI or backend-change policy.  A
tool still decides when to replace its native window and how to construct its
View; these helpers only centralize the platform-sensitive operations shared
by embedded graphics tools.
"""


def configure_window_surface(window, render):
    """Configure a QWindow's native surface for the active renderer."""
    device = getattr(render.opengl_context, "graphics_device", None)
    if device is not None:
        device.configure_surface(window)
        set_target_window = getattr(render, "set_target_window", None)
        if set_target_window is not None:
            set_target_window(window)
        return

    # Compatibility for contexts created outside the RAL.
    import sys
    from Qt.QtGui import QSurface

    context = getattr(render, "opengl_context", None)
    if hasattr(context, "_vulkan"):
        surface_type = (QSurface.SurfaceType.MetalSurface
                        if sys.platform == "darwin"
                        else QSurface.SurfaceType.VulkanSurface)
        window.setSurfaceType(surface_type)
        context.prepare_window(window)
    elif hasattr(context, "_metal"):
        window.setSurfaceType(QSurface.SurfaceType.MetalSurface)
    else:
        window.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)

    set_target_window = getattr(render, "set_target_window", None)
    if set_target_window is not None:
        set_target_window(window)


def create_window_container(window, parent, render):
    """Configure *window* for *render* and embed it under its final parent.

    Surface type and Vulkan platform state must be established before
    createWindowContainer() forces creation of the native child window.
    """
    from Qt.QtWidgets import QWidget

    configure_window_surface(window, render)
    return QWidget.createWindowContainer(window, parent)


def create_render_endpoint(window, parent, view, device):
    """Attach *view* to *window* on *device* and create its Qt container."""

    endpoint = device.create_endpoint(view, window)
    from Qt.QtWidgets import QWidget

    widget = QWidget.createWindowContainer(window, parent)
    return widget, endpoint


def draw_on_window(window, view, main_view, draw):
    """Run *draw* with the shared context targeting *window*.

    Returns False when the view belongs to a context that has already been
    replaced.  Context restoration is guaranteed even when drawing raises.
    """
    endpoint = getattr(view, "render_endpoint", None)
    if endpoint is not None:
        endpoint.draw(draw)
        return True

    render = view.render
    main_render = main_view.render
    if (render is None or main_render is None or
            render.opengl_context is not main_render.opengl_context):
        return False

    previous_window = render.use_shared_context(window)
    try:
        draw()
    finally:
        if hasattr(render.opengl_context, "_vulkan"):
            # A failed draw may still own an acquired Vulkan image/frame.
            # End it before changing surfaces; the native context rejects a
            # surface switch while a frame is active.
            try:
                render.done_current()
            finally:
                main_render.use_shared_context(previous_window)
        else:
            try:
                main_render.use_shared_context(previous_window)
            finally:
                render.done_current()
    return True


def release_window(window, view, main_view):
    """Release an auxiliary presentation surface before destroying its QWindow."""
    render = view.render
    if render is None:
        return

    endpoint = getattr(view, "render_endpoint", None)
    if endpoint is not None:
        endpoint.device.backend.release_surface(render, window)
        return

    context = render.opengl_context
    if getattr(context, "window", None) is window:
        main_render = main_view.render
        main_window = getattr(main_render, "_target_window", None)
        if main_window is not None and main_window is not window:
            main_render.use_shared_context(main_window)
    release = getattr(render, "release_window", None)
    if release is not None:
        release(window)
