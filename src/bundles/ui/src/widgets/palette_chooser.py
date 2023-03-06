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

    It is given a list of ColorButtons (either initially or later by setting the 'wells' attribute)
    and will show the palette name corresponding to the values of those buttons (or 'custom' if no
    palette corresponds).  The list of color buttons can be changed as needed.  PaletteChooser does
    not manage the color buttons (i.e. they must be placed in a layout elsewhere).

    The 'apply_cb' will be called if the user clicks the PaletteChooser's Apply button.  The value
    given to the callback will be the name of the palette and no color button values will be changed
    (at least directly changed by the PaletteChooser).  If no callback is supplied, then the
    color button values will be changed to match the palette.
    """
    def __init__(self, wells=[], *, apply_cb=None):
        super().__init__()

        layout = QHBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(1,1,1,1)
        layout.addStretch(1)
        self.palette_button = QPushButton("Apply")
        self.palette_button.clicked.connect(self._apply_palette)
        layout.addWidget(self.palette_button, alignment=Qt.AlignRight)
        layout.addWidget(QLabel("palette"))
        self.palette_menu_button = QPushButton()
        self.palette_menu = QMenu()
        self.palette_menu.triggered.connect(lambda act, *, mbut=self.palette_menu_button,
            abut=self.palette_button: (mbut.setText(act.text()),abut.setEnabled(True)))
        self.palette_menu_button.setMenu(self.palette_menu)
        layout.addWidget(self.palette_menu_button, alignment=Qt.AlignLeft)
        layout.addStretch(1)
        self.setLayout(layout)

        self._wells = []
        self.wells = wells
        self._apply_cb = apply_cb

    def update(self):
        from chimerax.core.colors import palette_name
        palette = palette_name([well.color/255.0 for well in self._wells])
        if palette is None:
            palette = "custom"
            enabled = False
        else:
            enabled = True
        self.palette_menu_button.setText(palette)
        self.palette_button.setEnabled(enabled)

    @property
    def wells(self):
        return self._wells[:]

    @wells.setter
    def wells(self, new_wells):
        if self._wells == new_wells:
            return
        for old_well in self._wells:
            old_well.color_changed.disconnect(self.update)
        for new_well in new_wells:
            new_well.color_changed.connect(self.update)
        if len(self._wells) != len(new_wells):
            self._update_palette_menu(len(new_wells))
        self._wells[:] = new_wells
        self.update()

    def _apply_palette(self):
        palette_name = self.palette_menu_button.text()
        if self._apply_cb is None:
            from chimerax.core.colors import BuiltinColormaps
            for well, rgb_a in zip(self._wells, BuiltinColormaps[palette_name].colors):
                if len(rgb_a) < 4:
                    rgb_a += [1.0]
                well.color = [255.0 * c for c in rgb_a]
        else:
            self._apply_cb(palette_name)

    def _update_palette_menu(self, num_colors):
        from chimerax.core.colors import BuiltinColormaps
        self.relevant_palettes = { name:cm for name, cm in BuiltinColormaps.items()
            if len(cm.colors) == num_colors }
        self.palette_menu.clear()
        if self.relevant_palettes:
            self.palette_button.setEnabled(True)
            self.palette_menu_button.setEnabled(True)
            palette_names = sorted(list(self.relevant_palettes))
            if self.palette_menu_button.text() not in self.relevant_palettes:
                self.palette_menu_button.setText(palette_names[0])
            for name in palette_names:
                self.palette_menu.addAction(name)
        else:
            self.palette_button.setEnabled(False)
            self.palette_menu_button.setEnabled(False)
            self.palette_menu_button.setText("No %d-color palettes known" % num_colors)
