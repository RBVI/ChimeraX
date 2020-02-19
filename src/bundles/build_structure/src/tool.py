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
from chimerax.core.errors import UserError
from chimerax.core.commands import run
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QMenu, QStackedWidget, QWidget, QLabel, QFrame
from PyQt5.QtWidgets import QGridLayout, QRadioButton, QHBoxLayout, QLineEdit, QCheckBox, QGroupBox
from PyQt5.QtWidgets import QButtonGroup, QAbstractButton
from PyQt5.QtCore import Qt

class BuildStructureTool(ToolInstance):

    help = "help:user/tools/buildstructure.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(3)
        parent.setLayout(layout)

        session.logger.status("Build Structure is a work in progress, more functions coming soon...",
            color="blue")

        self.category_button = QPushButton()
        layout.addWidget(self.category_button, alignment=Qt.AlignCenter)
        cat_menu = QMenu(parent)
        self.category_button.setMenu(cat_menu)
        cat_menu.triggered.connect(self._cat_menu_cb)

        self.category_areas = QStackedWidget()
        layout.addWidget(self.category_areas)

        self.handlers = []
        self.category_widgets = {}
        for category in ["Start Structure", "Modify Structure"]:
            self.category_widgets[category] = widget = QFrame()
            widget.setLineWidth(2)
            widget.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            getattr(self, "_layout_" + category.lower().replace(' ', '_'))(widget)
            self.category_areas.addWidget(widget)
            cat_menu.addAction(category)
        initial_category = "Start Structure"
        self.category_button.setText(initial_category)
        self.category_areas.setCurrentWidget(self.category_widgets[initial_category])

        tw.manage(placement="side")

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        super().delete()

    def _cat_menu_cb(self, action):
        self.category_areas.setCurrentWidget(self.category_widgets[action.text()])
        self.category_button.setText(action.text())

    def _layout_modify_structure(self, parent):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        layout.addWidget(QLabel("Change selected atom to..."), alignment=Qt.AlignHCenter | Qt.AlignBottom)
        frame = QFrame()
        layout.addWidget(frame, alignment=Qt.AlignHCenter | Qt.AlignTop)
        frame.setLineWidth(1)
        frame.setFrameStyle(QFrame.Panel | QFrame.Plain)
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(0,0,0,0)
        frame_layout.setSpacing(0)
        frame.setLayout(frame_layout)
        params_layout = QGridLayout()
        params_layout.setHorizontalSpacing(10)
        params_layout.setVerticalSpacing(0)
        frame_layout.addLayout(params_layout)
        for col, title in enumerate(["Element", "Bonds", "Geometry"]):
            params_layout.addWidget(QLabel(title), 0, col, alignment=Qt.AlignHCenter | Qt.AlignBottom)
        self.ms_elements_button = ebut = QPushButton()
        from chimerax.atomic.widgets import make_elements_menu
        elements_menu = make_elements_menu(parent)
        elements_menu.triggered.connect(lambda act, but=ebut: but.setText(act.text()))
        ebut.setMenu(elements_menu)
        ebut.setText("C")
        params_layout.addWidget(ebut, 1, 0)

        self.ms_bonds_button = bbut = QPushButton()
        bonds_menu = QMenu(parent)
        for nb in range(5):
            bonds_menu.addAction(str(nb))
        bonds_menu.triggered.connect(lambda act, but=bbut: but.setText(act.text()))
        bbut.setMenu(bonds_menu)
        bbut.setText("4")
        params_layout.addWidget(bbut, 1, 1)

        self.ms_geom_button = gbut = QPushButton()
        geom_menu = QMenu(parent)
        geom_menu.triggered.connect(lambda act, but=gbut: but.setText(act.text()))
        bonds_menu.triggered.connect(lambda act: self._ms_geom_menu_update())
        gbut.setMenu(geom_menu)
        params_layout.addWidget(gbut, 1, 2)
        self._ms_geom_menu_update()

        atom_name_area = QWidget()
        frame_layout.addWidget(atom_name_area, alignment=Qt.AlignCenter)
        atom_name_layout = QGridLayout()
        atom_name_layout.setContentsMargins(0,0,0,0)
        atom_name_layout.setSpacing(0)
        atom_name_area.setLayout(atom_name_layout)
        self.ms_retain_atom_name = rbut = QRadioButton("Retain current atom name")
        rbut.setChecked(True)
        atom_name_layout.setColumnStretch(1, 1)
        atom_name_layout.addWidget(rbut, 0, 0, 1, 2, alignment=Qt.AlignLeft)
        self.ms_change_atom_name = QRadioButton("Set atom name to:")
        atom_name_layout.addWidget(self.ms_change_atom_name, 1, 0)
        self.ms_atom_name = name_edit = QLineEdit()
        name_edit.setFixedWidth(50)
        name_edit.setText(ebut.text())
        elements_menu.triggered.connect(lambda act: self._ms_update_atom_name())
        atom_name_layout.addWidget(name_edit, 1, 1, alignment=Qt.AlignLeft)

        apply_but = QPushButton("Apply")
        apply_but.clicked.connect(lambda checked: self._ms_apply_cb())
        layout.addWidget(apply_but, alignment=Qt.AlignCenter)

        checkbox_area = QWidget()
        layout.addWidget(checkbox_area, alignment=Qt.AlignCenter)
        checkbox_layout = QVBoxLayout()
        checkbox_area.setLayout(checkbox_layout)
        self.ms_connect_back = connect = QCheckBox("Connect to pre-existing atoms if appropriate")
        connect.setChecked(True)
        checkbox_layout.addWidget(connect, alignment=Qt.AlignLeft)
        self.ms_focus = focus = QCheckBox("Focus view on modified residue")
        focus.setChecked(False)
        checkbox_layout.addWidget(focus, alignment=Qt.AlignLeft)
        self.ms_element_color = color = QCheckBox("Color new atoms by element")
        color.setChecked(True)
        checkbox_layout.addWidget(color, alignment=Qt.AlignLeft)

        res_group = QGroupBox("Residue Name")
        layout.addWidget(res_group, alignment=Qt.AlignCenter)
        group_layout = QGridLayout()
        group_layout.setContentsMargins(0,0,0,0)
        group_layout.setSpacing(0)
        res_group.setLayout(group_layout)
        self.ms_res_unchanged = QRadioButton("Leave unchanged")
        group_layout.addWidget(self.ms_res_unchanged, 0, 0, 1, 3, alignment=Qt.AlignLeft)
        self.ms_res_mod = QRadioButton("Change modified residue's name to")
        group_layout.addWidget(self.ms_res_mod, 1, 0, 1, 1, alignment=Qt.AlignLeft)
        self.ms_mod_edit = QLineEdit()
        self.ms_mod_edit.setFixedWidth(50)
        self.ms_mod_edit.setText("UNL")
        group_layout.addWidget(self.ms_mod_edit, 1, 1, 1, 2, alignment=Qt.AlignLeft)
        self.ms_res_new = QRadioButton("Put just changed atoms in new residue named")
        group_layout.addWidget(self.ms_res_new, 2, 0, 1, 2, alignment=Qt.AlignLeft)
        self.ms_res_new_name = QLineEdit()
        self.ms_res_new_name.setFixedWidth(50)
        self.ms_res_new_name.setText("UNL")
        group_layout.addWidget(self.ms_res_new_name, 2, 2, 1, 1)

        self.ms_res_mod.setChecked(True)

        from chimerax.core.selection import SELECTION_CHANGED
        self.handlers.append(self.session.triggers.add_handler(SELECTION_CHANGED, self._ms_sel_changed))
        self._ms_sel_changed()

        sep = QFrame()
        sep.setFrameStyle(QFrame.HLine)
        layout.addWidget(sep, stretch=1)

        delete_area = QWidget()
        layout.addWidget(delete_area, alignment=Qt.AlignCenter)
        delete_layout = QHBoxLayout()
        delete_area.setLayout(delete_layout)
        del_but = QPushButton("Delete")
        del_but.clicked.connect(self._ms_del_cb)
        delete_layout.addWidget(del_but, alignment=Qt.AlignRight)
        delete_layout.addWidget(QLabel("selected atoms/bonds"), alignment=Qt.AlignLeft)

    def _layout_start_structure(self, parent):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        from .manager import manager
        # until "lazy" managers are supported, 'manager' cannot be None at this point
        self.ss_u_to_p_names = { manager.ui_name(pn):pn for pn in manager.provider_names }
        ui_names = list(self.ss_u_to_p_names.keys())
        ui_names.sort(key=lambda x: x.lower())
        provider_layout = QGridLayout()
        layout.addLayout(provider_layout)
        provider_layout.addWidget(QLabel("Add "), 0, 0, len(ui_names)+2, 1)

        self.parameter_widgets = QStackedWidget()
        provider_layout.addWidget(self.parameter_widgets, 0, 2, len(ui_names)+2, 1)
        self.ss_widgets = {}
        self.ss_button_group = QButtonGroup()
        self.ss_button_group.buttonClicked[QAbstractButton].connect(self._ss_provider_changed)
        provider_layout.setRowStretch(0, 1)
        for row, ui_name in enumerate(ui_names):
            but = QRadioButton(ui_name)
            self.ss_button_group.addButton(but)
            provider_layout.addWidget(but, row+1, 1, alignment=Qt.AlignLeft)
            params_title = " ".join([x.capitalize()
                if x.islower() else x for x in ui_name.split()]) + " Parameters"
            self.ss_widgets[ui_name] = widget = QGroupBox(params_title)
            manager.fill_parameters_widget(self.ss_u_to_p_names[ui_name], widget)
            self.parameter_widgets.addWidget(widget)
            if row == 0:
                but.setChecked(True)
                self.parameter_widgets.setCurrentWidget(widget)
        provider_layout.setRowStretch(len(ui_names)+1, 1)

        model_area = QWidget()
        layout.addWidget(model_area, alignment=Qt.AlignCenter)
        model_layout = QHBoxLayout()
        model_layout.setSpacing(2)
        model_area.setLayout(model_layout)
        self.ss_struct_widgets= [QLabel("Put atoms in")]
        model_layout.addWidget(self.ss_struct_widgets[0])
        from chimerax.atomic.widgets import StructureMenuButton
        self.ss_struct_menu = StructureMenuButton(self.session, special_items=["new model"])
        self.ss_struct_menu.value = "new model"
        self.ss_struct_menu.value_changed.connect(self._ss_struct_changed)
        self.ss_struct_widgets.append(self.ss_struct_menu)
        model_layout.addWidget(self.ss_struct_menu)
        self.ss_model_name_label = QLabel("named:")
        model_layout.addWidget(self.ss_model_name_label)
        self.ss_struct_widgets.append(self.ss_model_name_label)
        self.ss_model_name_edit = edit = QLineEdit()
        edit.setText("custom built")
        self.ss_struct_widgets.append(edit)
        model_layout.addWidget(edit)

        self.ss_apply_button = apply_but = QPushButton("Apply")
        apply_but.clicked.connect(lambda checked: self._ss_apply_cb())
        layout.addWidget(apply_but, alignment=Qt.AlignCenter)

        layout.addStretch(1)

    def _ms_apply_cb(self):
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        num_selected = len(sel_atoms)
        if num_selected != 1:
            raise UserError("You must select exactly one atom to modify.")
        a = sel_atoms[0]

        element_name = self.ms_elements_button.text()
        num_bonds = self.ms_bonds_button.text()

        cmd = "build modify %s %s %s" % (a.atomspec, element_name, num_bonds)

        geometry = self.ms_geom_button.text()
        if geometry != "N/A":
            cmd += " geometry " + geometry

        if not self.ms_retain_atom_name.isChecked():
            new_name = self.ms_atom_name.text().strip()
            if not new_name:
                raise UserError("Must provide a name for the modified atom")
            if new_name != a.name:
                cmd += " name " + new_name

        if not self.ms_connect_back.isChecked():
            cmd += " connectBack false"

        if not self.ms_element_color.isChecked():
            cmd += " colorByElement false"

        if self.ms_res_mod.isChecked():
            res_name = self.ms_mod_edit.text().strip()
            if not res_name:
                raise UserError("Must provided modified residue name")
            if res_name != a.residue.name:
                cmd += " resName " + res_name
        elif self.ms_res_new.isChecked():
            res_name = self.ms_res_new_name.text().strip()
            if not res_name:
                raise UserError("Must provided new residue name")
            cmd += " resNewOnly true resName " + res_name

        run(self.session, cmd)

        if self.ms_focus.isChecked():
            run(self.session, "view " + a.residue.atomspec)

    def _ms_del_cb(self, *args):
        from chimerax.atomic import selected_atoms, selected_bonds
        if not selected_atoms(self.session) and not selected_bonds(self.session):
            raise UserError("No atoms or bonds selected")
        run(self.session, "del atoms sel; del bonds sel")

    def _ms_geom_menu_update(self):
        num_bonds = int(self.ms_bonds_button.text())
        but = self.ms_geom_button
        if num_bonds < 2:
            but.setEnabled(False)
            but.setText("N/A")
            return
        but.setEnabled(True)
        menu = but.menu()
        menu.clear()
        from chimerax.atomic.bond_geom import geometry_name
        for gname in geometry_name[num_bonds:]:
            menu.addAction(gname)
        but.setText(gname)

    def _ms_sel_changed(self, *args):
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if len(sel_atoms) != 1:
            return
        a = sel_atoms[0]
        self._ms_update_atom_name(a)
        from .mod import unknown_res_name
        res_name = unknown_res_name(a.residue)
        self.ms_mod_edit.setText(res_name)
        self.ms_res_new_name.setText(res_name)

    def _ms_update_atom_name(self, a=None):
        if a is None:
            from chimerax.atomic import selected_atoms
            sel_atoms = selected_atoms(self.session)
            if len(sel_atoms) != 1:
                return
            a = sel_atoms[0]
        new_element = self.ms_elements_button.text()
        from .mod import default_changed_name
        new_name = default_changed_name(a, new_element)
        self.ms_atom_name.setText(new_name)
        if new_name == a.name:
            self.ms_retain_atom_name.setChecked(True)
        else:
            self.ms_change_atom_name.setChecked(True)

    def _new_start_providers(self, new_providers):
        pass

    def _ss_apply_cb(self):
        ui_name = self.ss_button_group.checkedButton().text()
        provider_name = self.ss_u_to_p_names[ui_name]

        from .manager import manager
        subcmd_string = manager.get_command_substring(provider_name, self.ss_widgets[ui_name])
        if manager.new_model_only(provider_name):
            # provider needs to provide its own command in this case
            run(self.session, subcmd_string)
        else:
            struct_info = self.ss_struct_menu.value
            if isinstance(struct_info, str):
                model_name = self.ss_model_name_edit.text().strip()
                if not model_name:
                    raise UserError("New structure name must not be blank")
                from chimerax.core.commands import StringArg
                struct_arg = StringArg.unparse(model_name)
            else:
                struct_arg = struct_info.atomspec
            run(self.session, " ".join(["build start", provider_name, struct_arg, subcmd_string]))

    def _ss_provider_changed(self, button):
        ui_name = button.text()
        self.parameter_widgets.setCurrentWidget(self.ss_widgets[ui_name])
        from .manager import manager
        provider_name = self.ss_u_to_p_names[ui_name]
        hidden = manager.new_model_only(provider_name)
        if manager.is_indirect(provider_name):
            self.ss_apply_button.setHidden(True)
            hidden = True
        else:
            self.ss_apply_button.setHidden(False)
        for widget in self.ss_struct_widgets:
            widget.setHidden(hidden)

    def _ss_struct_changed(self):
        show = self.ss_struct_menu.value == "new model"
        self.ss_model_name_label.setHidden(not show)
        self.ss_model_name_edit.setHidden(not show)
