# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE VERSION 2.1.
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""Overlap tool definitions with the bundled Cocoa menu bar's creation."""
import ctypes
import sys
from importlib import import_module
from threading import Thread, current_thread, main_thread


_QT_WIDGETS_UUID = '3208c5aa6f153f55a3ef0eec3b210525'
_MENU_BAR_SYMBOL = '_ZNK11QMainWindow7menuBarEv'
_TOOL_MODULES = (
    ('Log', ('chimerax.log.parse', 'chimerax.log.view')),
    ('Command Line Interface', ('chimerax.cmd_line.tool',)),
    ('Model Panel', ('chimerax.model_panel.tool',)),
    ('Toolbar', ('chimerax.toolbar.tool',)),
)


def _definition_modules(ui):
    if not ui.autostart_tools:
        return ()
    names = ui.settings.autostart
    return tuple(module for name, modules in _TOOL_MODULES if name in names
        for module in modules if module not in sys.modules)


def _import_definitions(modules):
    # These built-in modules define classes and functions; they create no
    # widgets or tool instances. Actual tools are started normally on main.
    for name in modules:
        try:
            import_module(name)
        except Exception:
            # Let normal tool startup retry and report an import failure.
            pass


def _prepare_menu_bar(window):
    from .native_application import _image, _python_api
    import shiboken6

    # The application constructor already validated these binding images.
    widgets, pyside, shiboken, cpp = window.session.ui._native_application_libraries
    if _image(widgets, _MENU_BAR_SYMBOL)[1] != _QT_WIDGETS_UUID:
        raise ValueError('Unrecognized bundled QtWidgets framework')
    if not shiboken6.isValid(window):
        raise ValueError('Invalid main window')
    pointer = shiboken6.getCppPointer(window)[0]
    ptr = ctypes.c_void_p
    create = ctypes.CFUNCTYPE(ptr, ptr)((_MENU_BAR_SYMBOL, widgets))
    enter = _python_api(shiboken, '_ZN8Shiboken19PythonContextMarkerC1Ev', None, ptr)
    leave = _python_api(shiboken, '_ZN8Shiboken19PythonContextMarkerD1Ev', None, ptr)
    errors = _python_api(shiboken, '_ZN8Shiboken6Errors8occurredEv', ptr)

    def construct():
        # Qt stays on the main thread. Only this native call releases the GIL.
        # QMainWindow owns the new C++ menu bar. The ordinary binding call
        # afterward wraps that existing object and records its Python parent.
        create(pointer)
        errors()

    return construct, enter, leave


def menu_bar(window):
    """Return the normal menu wrapper after independent imports have joined."""
    cls = type(window)
    if (sys.platform != 'darwin' or sys.implementation.name != 'cpython'
            or current_thread() is not main_thread()
            or cls.__module__ != 'chimerax.ui.gui' or cls.__name__ != 'MainWindow'):
        return window.menuBar()
    ui = window.session.ui
    if not getattr(ui, '_native_qapplication', False):
        return window.menuBar()
    modules = _definition_modules(ui)
    if not modules:
        return window.menuBar()
    try:
        construct, enter, leave = _prepare_menu_bar(window)
    except (AttributeError, OSError, ValueError):
        return window.menuBar()

    context = ctypes.c_byte()
    worker = Thread(target=_import_definitions, args=(modules,),
        name='ChimeraX tool definitions')
    started = False
    enter(ctypes.byref(context))
    try:
        try:
            worker.start()
            started = True
        except RuntimeError:
            return window.menuBar()
        construct()
        bar = window.menuBar()
        window._native_menu_bar_overlap = True
        return bar
    finally:
        try:
            if started:
                worker.join()
        finally:
            leave(ctypes.byref(context))
