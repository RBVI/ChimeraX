# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
ui.widgets: ChimeraX graphical user interface widgets
=====================================================
"""

# Load widget implementations only when their public exports are requested.
_widget_modules = {
    'Citation': 'citation',
    'ColorButton': 'color_button',
    'MultiColorButton': 'color_button',
    'hex_color_name': 'color_button',
    'radio_buttons': 'composite',
    'button_row': 'composite',
    'vertical_layout': 'composite',
    'horizontal_layout': 'composite',
    'row_frame': 'composite',
    'column_frame': 'composite',
    'EntriesRow': 'composite',
    'CollapsiblePanel': 'composite',
    'ModelMenu': 'composite',
    'DisclosureArea': 'disclosure',
    'MarkedHistogram': 'histogram',
    'ModelListWidget': 'item_chooser',
    'ModelMenuButton': 'item_chooser',
    'ItemListWidget': 'item_chooser',
    'ItemMenuButton': 'item_chooser',
    'ItemTable': 'item_table',
    'get_csv_info': 'item_table',
    'LongJobDialog': 'job_dialog',
    'PaletteChooser': 'palette_chooser',
    'Slider': 'slider',
    'LogSlider': 'slider',
    'FloatSlider': 'slider',
    'TwoThreeStateCheckBox': 'tricheckbox',
    'HtmlView': 'htmlview',
    'ChimeraXHtmlView': 'htmlview',
}

# Keep the existing package-module exports as well as the widget names.
_widget_submodules = ('citation', 'color_button', 'composite', 'disclosure', 'histogram', 'item_chooser', 'item_table', 'job_dialog', 'palette_chooser', 'slider', 'tricheckbox')


def __getattr__(name):
    from importlib import import_module
    if name in _widget_submodules:
        value = import_module('.' + name, __name__)
    elif name in _widget_modules:
        value = getattr(import_module('.' + _widget_modules[name], __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_widget_modules) | set(_widget_submodules))


__all__ = [*_widget_submodules, *(_name for _name in _widget_modules
                                if _name not in ('HtmlView', 'ChimeraXHtmlView'))]
import Qt as _qt
from importlib.util import find_spec as _find_spec
_binding = ('PySide6' if _qt.using_pyside6 else 'PyQt6' if _qt.using_pyqt6
            else 'PySide2' if _qt.using_pyside2 else 'PyQt5')
if _find_spec(_binding + '.QtWebEngineWidgets') is not None:
    __all__ += ['HtmlView', 'ChimeraXHtmlView']
