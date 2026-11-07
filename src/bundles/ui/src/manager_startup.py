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

"""Build bundled manager state during the main thread's native menu creation."""
import ctypes
import os.path
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import find_spec
from threading import Thread, current_thread, main_thread


# These managers only build Python state and retain lazy provider callbacks.
# Arbitrary bundle managers must continue to initialize on the main thread.
_MANAGERS = {
    ('ChimeraX-PresetMgr', 'chimerax.preset_mgr'): {'presets'},
    ('ChimeraX-RotamerLibMgr', 'chimerax.rotamers'): {'rotamers'},
    ('ChimeraX-SchemeMgr', 'chimerax.scheme_mgr'): {'url_schemes'},
    ('ChimeraX-DataFormats', 'chimerax.data_formats'): {'data formats'},
    ('ChimeraX-Toolbar', 'chimerax.toolbar'): {'toolbar'},
    ('ChimeraX-SaveCommand', 'chimerax.save_command'): {'save command'},
    ('ChimeraX-OpenCommand', 'chimerax.open_command'): {'open command'},
    ('ChimeraX-Segmentations', 'chimerax.segmentations'): {'segmentation formats'},
    ('ChimeraX-Alignments', 'chimerax.seqalign'): {'alignments'},
    ('ChimeraX-ItemsInspection', 'chimerax.items_inspection'): {'items inspection'},
}


def _bundled_managers(toolshed):
    from chimerax import core
    from chimerax.core.toolshed.info import BundleInfo

    if toolshed._manager_instances:
        return False
    # Scheme registration becomes native work once WebEngine has loaded.
    if any(name in sys.modules for name in (
            'Qt.QtWebEngineCore', 'PySide6.QtWebEngineCore')):
        return False
    package_root = os.path.dirname(os.path.dirname(core.__file__))
    active = False
    for bi in toolshed._installed_bundle_info:
        # This manager eagerly invokes provider code, which may create UI.
        if any(key.startswith('segmentation formats/') for key in bi.providers):
            return False
        managers = {name for name, kw in bi.managers.items()
                    if kw.get('autostart', 'true') != 'false'}
        if not managers:
            continue
        if (type(bi) is not BundleInfo
                or not managers <= _MANAGERS.get((bi.name, bi.package_name), set())):
            return False
        try:
            spec = find_spec(bi.package_name)
        except (ImportError, AttributeError, ValueError):
            return False
        expected = os.path.join(package_root, bi.package_name.split('.')[-1], '__init__.py')
        if (spec is None or not isinstance(spec.loader, SourceFileLoader)
                or not spec.origin
                or os.path.realpath(spec.origin) != os.path.realpath(expected)):
            return False
        active = True
    return active


def _overlap(window, toolshed, session):
    from .native_menu import _prepare_menu_bar, _definition_modules, _import_definitions

    try:
        construct, enter, leave = _prepare_menu_bar(window)
    except (AttributeError, OSError, ValueError):
        return None
    modules = _definition_modules(session.ui)
    results, errors = [], []

    def initialize():
        try:
            results.append(toolshed._init_managers(session))
            _import_definitions(modules)
        except BaseException as error:
            errors.append((error, error.__traceback__))

    worker = Thread(target=initialize, name='ChimeraX manager state')
    started = False
    context = ctypes.c_byte()
    enter(ctypes.byref(context))
    try:
        try:
            worker.start()
            started = True
        except RuntimeError:
            return None
        construct()
        # Wrap the existing, QMainWindow-owned menu through the normal binding.
        window.menuBar()
        window._native_menu_bar_overlap = True
    finally:
        try:
            if started:
                worker.join()
        finally:
            leave(ctypes.byref(context))
    if errors:
        error, traceback = errors[0]
        raise error.with_traceback(traceback)
    return results[0]


def initialize_managers(toolshed, session):
    """Join manager state before normal custom initialization or window content."""
    ui = session.ui
    if (sys.platform != 'darwin' or sys.implementation.name != 'cpython'
            or current_thread() is not main_thread()
            or not getattr(ui, '_native_qapplication', False)
            or ui.main_window is not None
            or hasattr(ui, '_prepared_main_window')):
        return toolshed._init_managers(session)
    from .gui import UI, MainWindow
    from chimerax.core.toolshed import Toolshed
    if type(ui) is not UI or type(toolshed) is not Toolshed or not _bundled_managers(toolshed):
        return toolshed._init_managers(session)

    import shiboken6
    window = MainWindow(ui, session, _defer_build=True)
    try:
        failed = _overlap(window, toolshed, session)
    except BaseException:
        shiboken6.delete(window)
        raise
    if failed is None:
        shiboken6.delete(window)
        return toolshed._init_managers(session)
    ui._prepared_main_window = window
    ui._manager_state_overlap = True
    return failed
