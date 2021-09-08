# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
ui.widgets: ChimeraX graphical user interface widgets
=====================================================
"""

from Qt import qt_have_web_engine
if qt_have_web_engine():
    from .htmlview import HtmlView, ChimeraXHtmlView
from .color_button import ColorButton, MultiColorButton, hex_color_name
from .citation import Citation
from .histogram import MarkedHistogram
from .item_chooser import ModelListWidget, ModelMenuButton, ItemListWidget, ItemMenuButton
from .item_table import ItemTable
from .composite import radio_buttons, button_row, vertical_layout, horizontal_layout, row_frame
from .composite import EntriesRow, CollapsiblePanel, ModelMenu
from .slider import Slider, LogSlider
