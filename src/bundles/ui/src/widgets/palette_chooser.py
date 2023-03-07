# vim: set expandtab ts=4 sw=4:

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

from Qt.QtWidgets import QPushButton, QMenu, QHBoxLayout, QWidget, QLabel
from Qt.QtCore import Qt

class PaletteChooser(QWidget):
    """Widget for choosing/showing palettes.

    It is given a callback function to invoke if the user applies a palette.  The function will be called
    with the name of the palette.  If RGB/RGBA values are desired, access the widget's 'rgbs'/'rgbas'
    attribute.  Components of the rgb(a) values will be in the range 0-1.

    To get the widget to show palettes of the proper length and that correspond to your tool's current
    settings, you need to set the widget's rgbs/rgbas attribute with a list of rgb(a) values.  Again, the
    components of these values must be in the range 0-1.

    If 'auto_apply' is True, palettes will be applied as soon as the user selects them from the palette
    menu.  If it is False, the widget will have a separate Apply button and palettes will only be applied
    when that button is clicked.
    """

    NO_PALETTE = "custom"
    NO_NUM_PALETTES_PREFIX = "No "
    NO_NUM_PALETTES_SUFFIX = "-color palettes known"

    def __init__(self, apply_cb, *, auto_apply=True):
        super().__init__()
        self._apply_cb = apply_cb
        self._auto_apply = auto_apply

        layout = QHBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(1,1,1,1)
        layout.addStretch(1)
        if not auto_apply:
            self.apply_button = QPushButton("Apply")
            self.apply_button.clicked.connect(self._apply_palette)
            layout.addWidget(self.apply_button, alignment=Qt.AlignRight)
        layout.addWidget(QLabel("palette"))
        self.palette_menu_button = QPushButton()
        self.palette_menu = QMenu()
        self.palette_menu.triggered.connect(self._palette_menu_cb)
        self.palette_menu_button.setMenu(self.palette_menu)
        layout.addWidget(self.palette_menu_button, alignment=Qt.AlignLeft)
        layout.addStretch(1)
        self._last_palette_size = None
        self.rgbas = []
        self.setLayout(layout)

    @property
    def rgbs(self):
        rgbas = self.rgbas
        if rgbas is None:
            return None
        return [(r,g,b) for r,g,b,a in rgbas]

    @rgbs.setter
    def rgbs(self, rgbs):
        self.rgbas = [(r,g,b,1.0) for r,g,b in rgbs]

    @property
    def rgbas(self):
        palette_name = self.palette_menu_button.text()
        if not palette_name or palette_name == self.NO_PALETTE:
            return None
        from chimerax.core.colors import BuiltinColormaps
        return BuiltinColormaps[palette_name].colors

    @rgbas.setter
    def rgbas(self, rgbas):
        import numpy
        if numpy.array_equal(rgbas, self.rgbas):
            return
        from chimerax.core.colors import palette_name
        palette = palette_name(rgbas)
        if palette is None:
            palette = self.NO_PALETTE
            enabled = False
        else:
            enabled = True
        if not self._auto_apply:
            self.apply_button.setEnabled(enabled)
        self.palette_menu_button.setText(palette)
        if self._last_palette_size is None or len(rgbas) != self._last_palette_size:
            self._update_palette_menu(len(rgbas))
            self._last_palette_size = len(rgbas)

    def _apply_palette(self):
        self._apply_cb(self.palette_menu_button.text())

    @property
    def _palette_menu_button_valid(self):
        return self.palette_menu_button.text() != self.NO_PALETTE

    def _palette_menu_cb(self, action):
        menu_entry = action.text()
        valid_menu_entry = not (menu_entry.startswith(self.NO_NUM_PALETTES_PREFIX) \
            and menu_entry.endswith(self.NO_NUM_PALETTES_SUFFIX))
        self.apply_button.setEnabled(valid_menu_entry)
        if valid_menu_entry:
            self.palette_menu_button.setText(menu_entry)
            if self._auto_apply:
                self._apply_palette()

    def _update_palette_menu(self, num_colors):
        from chimerax.core.colors import BuiltinColormaps
        self.relevant_palettes = { name:cm for name, cm in BuiltinColormaps.items()
            if len(cm.colors) == num_colors }
        self.palette_menu.clear()
        if self.relevant_palettes:
            if not self._auto_apply:
                self.apply_button.setEnabled(True)
            palette_names = sorted(list(self.relevant_palettes))
            for name in palette_names:
                self.palette_menu.addAction(name)
        else:
            if not self._auto_apply:
                self.apply_button.setEnabled(False)
            self.palette_menu.addAction(self.NO_NUM_PALETTES_PREFIX + str(num_colors)
                + self.NO_NUM_PALETTES_SUFFIX)
