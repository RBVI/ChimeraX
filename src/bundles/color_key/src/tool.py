# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.tools import ToolInstance
from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QScrollArea, QWidget, QGridLayout
from Qt.QtWidgets import QLineEdit, QPushButton, QMenu, QDoubleSpinBox
from Qt.QtCore import Qt
from chimerax.core.commands import run, StringArg
from chimerax.core.colors import hex_color

_mouse_mode = None

class ColorKeyTool(ToolInstance):

    #help = "help:user/tools/modelpanel.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        layout = QVBoxLayout()
        layout.setSpacing(2)
        parent.setLayout(layout)

        from .model import get_model
        self.key = key = get_model(session)
        self.handlers = [
            key.triggers.add_handler('key closed', self._key_closed),
            key.triggers.add_handler('key changed', self._key_changed),
        ]
        self.colors_layout = QVBoxLayout()
        num_layout = QHBoxLayout()
        self.colors_layout.addLayout(num_layout)
        num_layout.addWidget(QLabel("Number of colors/labels:"), alignment=Qt.AlignRight)
        self.num_colors = QSpinBox()
        self.num_colors.setMinimum(2)
        self.num_colors.valueChanged.connect(self._colors_changed)
        num_layout.addWidget(self.num_colors, alignment=Qt.AlignLeft)
        layout.addLayout(self.colors_layout)
        self.wells = []
        self.labels = []
        self.wl_scroller = QScrollArea()
        self.wl_widget = QWidget()
        self.wl_layout = QGridLayout()
        self.wl_layout.setContentsMargins(1,1,1,1)
        self.wl_layout.setSpacing(2)
        self.wl_widget.setLayout(self.wl_layout)
        self.colors_layout.addWidget(self.wl_widget)

        reverse_layout = QHBoxLayout()
        reverse_layout.setSpacing(2)
        rev_but = QPushButton("Reverse")
        rev_but.clicked.connect(self._reverse_data)
        reverse_layout.addWidget(rev_but, alignment=Qt.AlignRight)
        reverse_layout.addWidget(QLabel("the above"), alignment=Qt.AlignLeft)
        layout.addLayout(reverse_layout)

        palette_layout = QHBoxLayout()
        palette_layout.setSpacing(2)
        palette_layout.setContentsMargins(1,1,1,1)
        self.palette_button = QPushButton("Apply")
        self.palette_button.clicked.connect(self._apply_palette)
        palette_layout.addWidget(self.palette_button, alignment=Qt.AlignRight, stretch=1)
        palette_layout.addWidget(QLabel("palette"))
        self.palette_menu_button = QPushButton()
        self.palette_menu = QMenu()
        self.palette_menu.triggered.connect(lambda act, *, but=self.palette_menu_button:
            but.setText(act.text()))
        self.palette_menu_button.setMenu(self.palette_menu)
        palette_layout.addWidget(self.palette_menu_button, alignment=Qt.AlignLeft, stretch=1)
        layout.addLayout(palette_layout)
        self._update_colors_layout() # which also updates palette menu

        global _mouse_mode
        if _mouse_mode is None:
            from .mouse_key import ColorKeyMouseMode
            _mouse_mode = ColorKeyMouseMode(session)
            session.ui.mouse_modes.add_mode(_mouse_mode)

        class ScreenFloatSpinBox(QDoubleSpinBox):
            def __init__(self, *args, minimum=0.0, maximum=1.0, **kw):
                super().__init__(*args, **kw)
                self.setDecimals(5)
                self.setMinimum(minimum)
                self.setMaximum(maximum)
                self.setSingleStep(0.01)

        position_layout = QHBoxLayout()
        pos_label = QLabel("Position: x")
        pos_label.setToolTip("Position of lower left corner of the colored part of the key")
        position_layout.addWidget(pos_label, alignment=Qt.AlignRight, stretch=1)
        self.pos_x_box = ScreenFloatSpinBox(minimum=-1.0, maximum=2.0)
        self.pos_x_box.setValue(self.key.position[0])
        self.pos_x_box.valueChanged.connect(self._new_key_position)
        position_layout.addWidget(self.pos_x_box)
        position_layout.addWidget(QLabel(" y"))
        self.pos_y_box = ScreenFloatSpinBox(minimum=-1.0, maximum=2.0)
        self.pos_y_box.setValue(self.key.position[1])
        self.pos_y_box.valueChanged.connect(self._new_key_position)
        position_layout.addWidget(self.pos_y_box, alignment=Qt.AlignLeft, stretch=1)
        layout.addLayout(position_layout)

        position_layout = QHBoxLayout()
        pos_label = QLabel("Size: width")
        pos_label.setToolTip("Size of the colored part of they key, from 0-1 (fraction of screen size)")
        position_layout.addWidget(pos_label, alignment=Qt.AlignRight, stretch=1)
        self.size_w_box = ScreenFloatSpinBox()
        self.size_w_box.setValue(self.key.size[0])
        self.size_w_box.valueChanged.connect(self._new_key_size)
        position_layout.addWidget(self.size_w_box)
        position_layout.addWidget(QLabel(" height"))
        self.size_h_box = ScreenFloatSpinBox()
        self.size_h_box.setValue(self.key.size[1])
        self.size_h_box.valueChanged.connect(self._new_key_size)
        position_layout.addWidget(self.size_h_box, alignment=Qt.AlignLeft, stretch=1)
        layout.addLayout(position_layout)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Close | qbbox.Help)
        bbox.rejected.connect(self.delete)
        delete_button = bbox.addButton("Delete", qbbox.DestructiveRole)
        delete_button.clicked.connect(self._delete_key)
        #bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        self.key = None
        super().delete()

    def _apply_palette(self):
        print("sending command:", "key " + StringArg.unparse(self.palette_menu_button.text())
            + " " + " ".join([StringArg.unparse(':' + label.text()) for label in reversed(self.labels)]))
        run(self.session, "key " + StringArg.unparse(self.palette_menu_button.text())
            + " " + " ".join([StringArg.unparse(':' + label.text()) for label in reversed(self.labels)]))

    def _colors_labels_arg(self, wells=None, labels=None):
        if wells is None:
            wells = self.wells
        if labels is None:
            labels = self.labels
        return " ".join([StringArg.unparse("%s:%s" % (hex_color(well.color), label.text()))
            for well, label in reversed(list(zip(wells, labels)))])

    def _delete_key(self):
        from chimerax.core.commands import run
        run(self.session, "key delete")

    def _key_changed(self, trig_name, what_changed):
        if what_changed == "rgbas_and_labels":
            self._update_colors_layout()
        elif what_changed == "position":
            self.pos_x_box.blockSignals(True)
            self.pos_y_box.blockSignals(True)
            self.pos_x_box.setValue(self.key.position[0])
            self.pos_y_box.setValue(self.key.position[1])
            self.pos_x_box.blockSignals(False)
            self.pos_y_box.blockSignals(False)
        elif what_changed == "size":
            self.size_w_box.blockSignals(True)
            self.size_h_box.blockSignals(True)
            self.size_w_box.setValue(self.key.size[0])
            self.size_h_box.setValue(self.key.size[1])
            self.size_w_box.blockSignals(False)
            self.size_h_box.blockSignals(False)

    def _key_closed(self, *args):
        self.delete()

    def _new_key_data(self, *args):
        run(self.session, "key " + self._colors_labels_arg())

    def _new_key_position(self):
        run(self.session, "key pos %s,%s" % (self.pos_x_box.cleanText(), self.pos_y_box.cleanText()))

    def _new_key_size(self):
        run(self.session, "key size %s,%s" % (self.size_w_box.cleanText(), self.size_h_box.cleanText()))

    def _colors_changed(self, num_colors):
        if num_colors > len(self.wells):
            arg = self._colors_labels_arg() + ' ' + " ".join(["white: "] * (num_colors - len(self.wells)))
        elif num_colors < len(self.wells):
            skip = len(self.wells) - num_colors
            arg = self._colors_labels_arg(wells=self.wells[skip:], labels=self.labels[skip:])
        else:
            arg = self._colors_labels_arg()
        run(self.session, "key " + arg)

    def _reverse_data(self, *args):
        run(self.session, "key " + self._colors_labels_arg(wells=reversed(self.wells),
            labels=reversed(self.labels)))

    def _update_colors_layout(self):
        rgbas_and_labels = self.key.rgbas_and_labels
        num_colors = len(rgbas_and_labels)
        self.num_colors.blockSignals(True)
        try:
            self.num_colors.setValue(num_colors)
        finally:
            self.num_colors.blockSignals(False)

        if num_colors != len(self.wells):
            self._update_palette_menu(num_colors)

        if num_colors > 10:
            if len(self.wells) > 10:
                # want scroll area to adjust to changes so...
                self.wl_scroller.takeWidget()
            else:
                self.colors_layout.removeWidget(self.wl_widget)
                self.colors_layout.addWidget(self.wl_scroller)
                self.wl_scroller.show()
        elif len(self.wells) > 10:
            self.wl_scroller.takeWidget()
            self.wl_scroller.hide()
            self.colors_layout.removeWidget(self.wl_scroller)
            self.colors_layout.addWidget(self.wl_widget)
        if num_colors < len(self.wells):
            for well in self.wells[num_colors:]:
                self.wl_layout.removeWidget(well)
                well.hide()
            self.wells = self.wells[:num_colors]
            for label in self.labels[num_colors:]:
                self.wl_layout.removeWidget(label)
                label.hide()
            self.labels = self.labels[:num_colors]
        elif num_colors > len(self.wells):
            from chimerax.ui.widgets import ColorButton
            for i in range(len(self.wells), num_colors):
                well = ColorButton()
                self.wl_layout.addWidget(well, i, 0)
                well.color_changed.connect(self._new_key_data)
                self.wells.append(well)
                label = QLineEdit()
                self.wl_layout.addWidget(label, i, 1)
                label.textEdited.connect(self._new_key_data)
                self.labels.append(label)
        if num_colors > 10:
            # sort of as per the Qt docs, this call needs to be after the layout is populated
            self.wl_scroller.setWidget(self.wl_widget)

        for i in range(num_colors):
            rgba, text = rgbas_and_labels[i]
            self.wells[num_colors - 1 - i].color = [int(255.0 * x + 0.5) for x in rgba]
            self.labels[num_colors - 1 - i].setText("" if text is None else text)

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


