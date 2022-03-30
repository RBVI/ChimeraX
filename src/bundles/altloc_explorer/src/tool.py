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
from Qt.QtWidgets import QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QButtonGroup, QRadioButton, QWidget
from Qt.QtWidgets import QPushButton, QScrollArea
from Qt.QtCore import Qt
from chimerax.core.commands import run

class AltlocExplorerTool(ToolInstance):

    help = "help:user/tools/altlocexplorer.html"

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
        self._no_structure_label = QLabel("No atomic model chosen")
        layout.addWidget(self._no_structure_label)
        self._structure_widget = None
        self._changes_handler = None
        self._button_lookup = {}
        #TODO: react to alt loc additions/subtractions

        tw.manage(placement='side')

    def delete(self):
        if self._changes_handler:
            self._changes_handler.remove()
        self._structure_button.destroy()
        super().delete()

    def _atomic_changes(self, trig_name, trig_data):
        if "alt_loc changed" in trig_data.atom_reasons():
            for r, but_map in self._button_lookup.items():
                r_al = r.alt_loc
                for al, but in but_map.items():
                    if but.isChecked() and al != r_al:
                        but.setChecked(False)
                    elif not but.isChecked() and al == r_al:
                        but.setChecked(True)

    def _make_structure_widget(self, structure):
        scroll_area = QScrollArea()
        widget = QWidget()
        layout = QGridLayout()
        layout.setSpacing(2)
        widget.setLayout(layout)
        from itertools import count
        rows = count()
        from chimerax.core.commands import run
        self._button_groups = []
        alt_loc_rs = [r for r in structure.residues if r.alt_locs]
        col_offset = 0
        for r in alt_loc_rs:
            row = next(rows)
            button = QPushButton(r.string(omit_structure=True))
            button.clicked.connect(lambda *args, ses=self.session, run=run, spec=r.atomspec:
                run(ses, "show %s; view %s" % (spec, spec)))
            layout.addWidget(button, row, 0 + col_offset, alignment=Qt.AlignRight)
            button_group = QButtonGroup()
            self._button_groups.append(button_group)
            but_layout = QHBoxLayout()
            layout.addLayout(but_layout, row, 1 + col_offset, alignment=Qt.AlignLeft)
            for alt_loc in sorted(list(r.alt_locs)):
                but = QRadioButton(alt_loc)
                self._button_lookup.setdefault(r, {})[alt_loc] = but
                but.setChecked(r.alt_loc == alt_loc)
                but.clicked.connect(lambda *args, ses=self.session, run=run, spec=r.atomspec, loc=alt_loc:
                    run(ses, "altlocs change %s %s" % (loc, spec)))
                button_group.addButton(but)
                but_layout.addWidget(but, alignment=Qt.AlignCenter)
            if row < len(alt_loc_rs)-1 and row >= int(len(alt_loc_rs)/2):
                layout.setColumnStretch(2+col_offset, 1)
                layout.setColumnMinimumWidth(2+col_offset, 5)
                col_offset += 3
                rows = count()

        if not alt_loc_rs:
            layout.addWidget(QLabel("No alternate locations in this structure"), 0, 0)
        scroll_area.setWidget(widget)
        return scroll_area

    def _structure_change(self):
        if self._structure_widget:
            self._layout.removeWidget(self._structure_widget)
            self._structure_widget.hide()
            self._structure_widget.destroy()
            self._button_groups.clear()
            self._changes_handler.remove()
            self._changes_handler = None
            self._button_lookup.clear()

        structure = self._structure_button.value
        if structure:
            self._no_structure_label.hide()
            self._structure_widget = self._make_structure_widget(structure)
            self._layout.addWidget(self._structure_widget, alignment=Qt.AlignCenter)
            from chimerax.atomic import get_triggers
            self._changes_handler = get_triggers().add_handler('changes', self._atomic_changes)
        else:
            self._no_structure_label.show()
            self._structure_widget = None

