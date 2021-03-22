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
from Qt.QtWidgets import QLineEdit, QPushButton, QMenu, QDoubleSpinBox, QCheckBox
from Qt.QtCore import Qt
from chimerax.core.commands import run, StringArg, camel_case
from chimerax.core.colors import color_name, rgba8_to_rgba
from .cmd import auto_color_strings, palette_name, palette_equal

_mouse_mode = None

class ColorKeyTool(ToolInstance):

    help = "help:user/tools/colorkey.html"

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
        reverse_layout.addStretch(1)
        rev_but = QPushButton("Reverse")
        rev_but.clicked.connect(self._reverse_data)
        reverse_layout.addWidget(rev_but, alignment=Qt.AlignRight)
        reverse_layout.addWidget(QLabel("the above"), alignment=Qt.AlignLeft)
        reverse_layout.addStretch(1)
        self.blend_colors_box = QCheckBox("Blend colors")
        self.blend_colors_box.setChecked(key.color_treatment == key.CT_BLENDED)
        self.blend_colors_box.clicked.connect(self._color_treatment_changed)
        reverse_layout.addWidget(self.blend_colors_box)
        reverse_layout.addStretch(1)
        layout.addLayout(reverse_layout)

        palette_layout = QHBoxLayout()
        palette_layout.setSpacing(2)
        palette_layout.setContentsMargins(1,1,1,1)
        palette_layout.addStretch(1)
        self.palette_button = QPushButton("Apply")
        self.palette_button.clicked.connect(self._apply_palette)
        palette_layout.addWidget(self.palette_button, alignment=Qt.AlignRight)
        palette_layout.addWidget(QLabel("palette"))
        self.palette_menu_button = QPushButton()
        self.palette_menu = QMenu()
        self.palette_menu.triggered.connect(lambda act, *, mbut=self.palette_menu_button,
            abut=self.palette_button: (mbut.setText(act.text()),abut.setEnabled(True)))
        self.palette_menu_button.setMenu(self.palette_menu)
        palette_layout.addWidget(self.palette_menu_button, alignment=Qt.AlignLeft)
        palette_layout.addStretch(1)
        layout.addLayout(palette_layout)
        self._update_colors_layout() # which also updates palette menu

        global _mouse_mode
        if _mouse_mode is None:
            from .mouse_key import ColorKeyMouseMode
            _mouse_mode = ColorKeyMouseMode(session)
            session.ui.mouse_modes.add_mode(_mouse_mode)
        self.handlers.append(_mouse_mode.triggers.add_handler('drag finished', self._drag_finished))

        class ScreenFloatSpinBox(QDoubleSpinBox):
            def __init__(self, *args, minimum=0.0, maximum=1.0, **kw):
                super().__init__(*args, **kw)
                self.setDecimals(5)
                self.setMinimum(minimum)
                self.setMaximum(maximum)
                self.setSingleStep(0.01)

        position_layout = QHBoxLayout()
        position_layout.addStretch(1)
        pos_label = QLabel("Position: x")
        pos_label.setToolTip("Position of lower left corner of the colored part of the key")
        position_layout.addWidget(pos_label, alignment=Qt.AlignRight)
        self.pos_x_box = ScreenFloatSpinBox(minimum=-1.0, maximum=2.0)
        self.pos_x_box.setValue(self.key.pos[0])
        self.pos_x_box.valueChanged.connect(self._new_key_position)
        position_layout.addWidget(self.pos_x_box)
        position_layout.addWidget(QLabel(" y"))
        self.pos_y_box = ScreenFloatSpinBox(minimum=-1.0, maximum=2.0)
        self.pos_y_box.setValue(self.key.pos[1])
        self.pos_y_box.valueChanged.connect(self._new_key_position)
        position_layout.addWidget(self.pos_y_box, alignment=Qt.AlignLeft)
        position_layout.addStretch(1)
        layout.addLayout(position_layout)

        size_layout = QHBoxLayout()
        size_layout.addStretch(1)
        pos_label = QLabel("Size: width")
        pos_label.setToolTip("Size of the colored part of they key, from 0-1 (fraction of screen size)")
        size_layout.addWidget(pos_label, alignment=Qt.AlignRight)
        self.size_w_box = ScreenFloatSpinBox()
        self.size_w_box.setValue(self.key.size[0])
        self.size_w_box.valueChanged.connect(self._new_key_size)
        size_layout.addWidget(self.size_w_box)
        size_layout.addWidget(QLabel(" height"))
        self.size_h_box = ScreenFloatSpinBox()
        self.size_h_box.setValue(self.key.size[1])
        self.size_h_box.valueChanged.connect(self._new_key_size)
        size_layout.addWidget(self.size_h_box, alignment=Qt.AlignLeft)
        size_layout.addStretch(1)
        layout.addLayout(size_layout)

        mouse_layout = QHBoxLayout()
        mouse_layout.addStretch(1)
        self.mouse_on_button = QCheckBox("Adjust key with")
        self.mouse_on_button.setChecked(True)
        self.mouse_on_button.clicked.connect(self._mouse_on_changed)
        mouse_layout.addWidget(self.mouse_on_button, alignment=Qt.AlignRight)
        self.mouse_button_button = QPushButton("left")
        menu = QMenu()
        for but in ["left", "middle", "right"]:
            menu.addAction(but)
        menu.triggered.connect(self._mouse_button_changed)
        self.mouse_button_button.setMenu(menu)
        mouse_layout.addWidget(self.mouse_button_button)
        mouse_layout.addWidget(QLabel("mouse button"), alignment=Qt.AlignLeft)
        self._mouse_on_changed(True)
        mouse_layout.addStretch(1)
        layout.addLayout(mouse_layout)

        from chimerax.ui.options import CategorizedOptionsPanel, EnumOption, OptionalColorOption, \
            FloatOption, IntOption, BooleanOption, FontOption
        class LabelSideOption(EnumOption):
            values = self.key.label_sides
        class AutoColorOption(OptionalColorOption):
            def get_value(self):
                val = super().get_value()
                if val is None:
                    return auto_color_strings[0]
                return val
            def set_value(self, val):
                if val in auto_color_strings:
                    val = None
                super().set_value(val)
        class LabelJustificationOption(EnumOption):
            values = self.key.justifications
        class NumLabelSpacingOption(EnumOption):
            values = self.key.numeric_label_spacings
        options_data = [
            ("Labels", [
                ("Color", 'label_rgba', AutoColorOption),
                ('Font size', 'font_size', (IntOption, {'min': 1})),
                ('Font', 'font', FontOption),
                ('Bold', 'bold', BooleanOption),
                ('Italic', 'italic', BooleanOption),
                ("Numeric spacing", 'numeric_label_spacing', NumLabelSpacingOption),
                ("Side", 'label_side', LabelSideOption),
                ("Justification", 'justification', LabelJustificationOption),
                ('Offset', 'label_offset', (FloatOption, {'decimal_places': 1, 'step': 1})),
                ]),
            ("Border", [
                ("Show border", 'border', BooleanOption),
                ("Color", 'border_rgba', AutoColorOption),
                ('Width', 'border_width', (FloatOption, {'decimal_places': 1, 'step': 1, 'min': 0})),
                ]),
            ("Tick Marks", [
                ("Show tick marks", 'ticks', BooleanOption),
                ('Length', 'tick_length', (FloatOption, {'decimal_places': 1, 'step': 2, 'min': 0})),
                ('Width', 'tick_thickness', (FloatOption, {'decimal_places': 1, 'step': 1, 'min': 0})),
            ]),
        ]
        scrolling = { x[0]: False for x in options_data }
        scrolling = { }
        self._options = {}
        options = CategorizedOptionsPanel(option_sorting=False, category_scrolled=scrolling)
        for cat, opt_info in options_data:
            for label, attr_name, opt_class in opt_info:
                if isinstance(opt_class, tuple):
                    opt_class, kw = opt_class
                else:
                    kw = {}
                opt = opt_class(label, getattr(self.key, attr_name), self._opt_cb, attr_name=attr_name, **kw)
                self._options[attr_name] = opt
                options.add_option(cat, opt)
        layout.addWidget(options)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Close | qbbox.Help)
        bbox.rejected.connect(self.delete)
        delete_button = bbox.addButton("Delete/Close", qbbox.DestructiveRole)
        delete_button.clicked.connect(self._delete_key)
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def delete(self):
        if self._mouse_handler:
            button = self.mouse_button_button.text()
            if self._prev_mouse_mode:
                new_mode = self._prev_mouse_mode.name
            else:
                new_mode = 'none'
            self._self_mm_change = True
            run(self.session, "ui mousemode %s %s" % (button, StringArg.unparse(new_mode)))
            self._self_mm_change = False
        for handler in self.handlers:
            handler.remove()
        self.key = None
        super().delete()

    def _apply_palette(self):
        run(self.session, "key " + StringArg.unparse(self.palette_menu_button.text())
            + " " + " ".join([StringArg.unparse(':' + label.text()) for label in reversed(self.labels)]))

    def _color_treatment_changed(self, blend):
        run(self.session, "key colorTreatment " + (self.key.CT_BLENDED if blend else self.key.CT_DISTINCT))

    def _colors_changed(self, num_colors):
        if num_colors > len(self.wells):
            arg = self._colors_labels_arg() + ' ' + " ".join(["white: "] * (num_colors - len(self.wells)))
        elif num_colors < len(self.wells):
            skip = len(self.wells) - num_colors
            arg = self._colors_labels_arg(wells=self.wells[skip:], labels=self.labels[skip:])
        else:
            arg = self._colors_labels_arg()
        run(self.session, "key " + arg)

    def _colors_labels_arg(self, wells=None, labels=None):
        if wells is None:
            wells = self.wells
        if labels is None:
            labels = self.labels
        # iterators and reverseiterators are not directly zippable or reverseable
        wells = list(reversed(list(wells)))
        labels = list(reversed(list(labels)))
        palette = palette_name([rgba8_to_rgba(w.color) for w in wells])
        if palette is None:
            return " ".join([StringArg.unparse("%s:%s" % (color_name(well.color), label.text()))
                for well, label in zip(wells, labels)])
        return StringArg.unparse(palette) + ' ' + ' '.join([StringArg.unparse(':' + l.text())
            for l in labels])

    def _delete_key(self):
        run(self.session, "key delete")

    def _drag_finished(self, *args):
        self.pos_x_box.blockSignals(True)
        self.pos_y_box.blockSignals(True)
        self.pos_x_box.setValue(self.key.pos[0])
        self.pos_y_box.setValue(self.key.pos[1])
        self.pos_x_box.blockSignals(False)
        self.pos_y_box.blockSignals(False)
        self.size_w_box.blockSignals(True)
        self.size_h_box.blockSignals(True)
        self.size_w_box.setValue(self.key.size[0])
        self.size_h_box.setValue(self.key.size[1])
        self.size_w_box.blockSignals(False)
        self.size_h_box.blockSignals(False)

    def _key_changed(self, trig_name, what_changed):
        if what_changed == "rgbas_and_labels":
            self._update_colors_layout()
        elif what_changed == "pos":
            if _mouse_mode.mouse_down_position:
                return
            self.pos_x_box.blockSignals(True)
            self.pos_y_box.blockSignals(True)
            self.pos_x_box.setValue(self.key.pos[0])
            self.pos_y_box.setValue(self.key.pos[1])
            self.pos_x_box.blockSignals(False)
            self.pos_y_box.blockSignals(False)
        elif what_changed == "size":
            if _mouse_mode.mouse_down_position:
                return
            self.size_w_box.blockSignals(True)
            self.size_h_box.blockSignals(True)
            self.size_w_box.setValue(self.key.size[0])
            self.size_h_box.setValue(self.key.size[1])
            self.size_w_box.blockSignals(False)
            self.size_h_box.blockSignals(False)
        elif what_changed == "color_treatment":
            self.blend_colors_box.setChecked(self.key.color_treatment == self.key.CT_BLENDED)
        elif what_changed in self._options:
            self._options[what_changed].value = getattr(self.key, what_changed)

    def _key_closed(self, *args):
        self.delete()

    def _mm_changed(self, trig_name, data):
        if self._self_mm_change:
            return
        button, modifiers, mode = data
        key_button = self.mouse_button_button.text()
        if button != key_button or modifiers:
            return
        self.mouse_on_button.setChecked(False)
        self._prev_mouse_mode = None
        self.handlers.remove(self._mouse_handler)
        self._mouse_handler.remove()
        self._mouse_handler = None

    def _mouse_button_changed(self, action):
        old_button = self.mouse_button_button.text()
        new_button = action.text()
        self.mouse_button_button.setText(new_button)
        if self._mouse_handler:
            if self._prev_mouse_mode:
                restore_mode = self._prev_mouse_mode.name
            else:
                restore_mode = 'none'
            mm = self.session.ui.mouse_modes
            self._prev_mouse_mode = mm.mode(button=new_button)
            self._self_mm_change = True
            run(self.session, "ui mousemode %s %s" % (old_button, StringArg.unparse(restore_mode)))
            run(self.session, "ui mousemode %s 'color key'" % new_button)
            self._self_mm_change = False

    def _mouse_on_changed(self, mouse_on):
        button = self.mouse_button_button.text()
        if mouse_on:
            self._prev_mouse_mode = self.session.ui.mouse_modes.mode(button=button)
            new_mode = 'color key'
            self._mouse_handler = self.session.triggers.add_handler('set mouse mode', self._mm_changed)
            self.handlers.append(self._mouse_handler)
        else:
            if self._prev_mouse_mode:
                new_mode = self._prev_mouse_mode.name
            else:
                new_mode = 'none'
            self._prev_mouse_mode = None
            self.handlers.remove(self._mouse_handler)
            self._mouse_handler.remove()
            self._mouse_handler = None
        self._self_mm_change = True
        run(self.session, "ui mousemode %s %s" % (button, StringArg.unparse(new_mode)))
        self._self_mm_change = False

    def _new_key_data(self, *args):
        run(self.session, "key " + self._colors_labels_arg())

    def _new_key_position(self):
        run(self.session, "key pos %s,%s" % (self.pos_x_box.cleanText(), self.pos_y_box.cleanText()))

    def _new_key_size(self):
        run(self.session, "key size %s,%s" % (self.size_w_box.cleanText(), self.size_h_box.cleanText()))

    def _opt_cb(self, opt):
        attr_name = opt.attr_name
        if attr_name.endswith('rgba'):
            attr_name = attr_name[:-4] + 'color'
            val = opt.value
            if val is None:
                val = auto_color_strings[0]
            else:
                from chimerax.core.colors import color_name
                val = color_name(opt.value)
        elif attr_name == 'font':
            val = StringArg.unparse(opt.value)
        else:
            val = str(opt.value).split()[0]
        run(self.session, "key %s %s" % (camel_case(attr_name), val))

    def _reverse_data(self, *args):
        run(self.session, "key " + self._colors_labels_arg(wells=reversed(self.wells),
            labels=reversed(self.labels)))

    def _set_palette_name(self, rgbas):
        palette = palette_name(rgbas)
        if palette is None:
            palette = "custom"
            enabled = False
        else:
            enabled = True
        self.palette_menu_button.setText(palette)
        self.palette_button.setEnabled(enabled)

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
        self._set_palette_name([rgba for rgba, text in rgbas_and_labels])

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
