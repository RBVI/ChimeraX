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
from Qt.QtWidgets import QVBoxLayout, QGridLayout, QLabel
from Qt.QtCore import Qt
from chimerax.core.commands import run

class AltlocExplorerTool(ToolInstance):

    #help = "help:user/tools/colorkey.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        self._layout = layout = QVBoxLayout()
        layout.setSpacing(2)
        parent.setLayout(layout)

        from chimerax.atomic.widgets import AtomicStructureMenuButton as ASMB
        self._structure_button = button = ASMB(session)
        button.value_changed.connect(self._structure_change)
        layout.addWidget(button)
        self._no_structure_label = QLabel("No atomic models open")
        layout.addWidget(self._no_structure_label)
        self._structure_layout = None
        self._structure_change()
        #TODO: react to alt loc changes/additions/subtractions
        """
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
        """

        tw.manage(placement='side')

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

    def _make_structure_layout(self, structure):
        layout = QGridLayout()
        for r in structure.residues:
            #TODO: need a Residue function to return available alt locs
            if max(r.atoms.num_alt_locs) < 2:
                continue
            
        if not layout.children():
        return layout

    def _structure_change(self):
        if self._structure_layout:
            self._layout.removeLayout(self._structure_layout)
            self._structure_layout.destroy()

        structure = self._structure_button.value
        if structure:
            self._no_structure_label.hide()
            self._structure_layout = self._make_structure_layout(structure)
            self._layout.addLayout(self._structure_layout)
        else:
            self._no_structure_label.show()
            self._structure_layout = None

