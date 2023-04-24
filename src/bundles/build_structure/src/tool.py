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
from Qt.QtWidgets import QVBoxLayout, QPushButton, QMenu, QStackedWidget, QWidget, QLabel, QFrame
from Qt.QtWidgets import QGridLayout, QRadioButton, QHBoxLayout, QLineEdit, QCheckBox, QGroupBox
from Qt.QtWidgets import QButtonGroup, QAbstractButton, QStyle, QToolButton, QDoubleSpinBox, QDial
from Qt.QtGui import QAction
from Qt.QtCore import Qt

class BuildStructureTool(ToolInstance):

    help = "help:user/tools/buildstructure.html"
    SESSION_SAVE = True

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(3)
        parent.setLayout(layout)

        self.category_button = QPushButton()
        layout.addWidget(self.category_button, alignment=Qt.AlignCenter)
        cat_menu = QMenu(parent)
        self.category_button.setMenu(cat_menu)
        cat_menu.triggered.connect(self._cat_menu_cb)

        self.category_areas = QStackedWidget()
        layout.addWidget(self.category_areas)

        self.handlers = {}
        self.category_widgets = {}
        for category in ["Start Structure", "Modify Structure", "Adjust Bonds", "Adjust Torsions",
                "Join Models", "Invert", "Adjust Angles"]:
            self.category_widgets[category] = widget = QFrame()
            widget.setLineWidth(2)
            widget.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            getattr(self, "_layout_" + category.lower().replace(' ', '_'))(widget)
            self.category_areas.addWidget(widget)
            cat_menu.addAction(category)
        self.show_category("Start Structure")

        tw.manage(placement="side")

    def delete(self):
        for handler_list in self.handlers.values():
            for handler in handler_list:
                handler.remove()
        for rotater in self.torsion_data.keys():
            self.session.bond_rotations.delete_rotation(rotater)
        super().delete()

    def show_category(self, category):
        self.category_button.setText(category)
        self.category_areas.setCurrentWidget(self.category_widgets[category])

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'tab': self.category_button.text()
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        if data['tab'] in inst.category_widgets:
            inst.show_category(data['tab'])
        return inst

    def _aa_add_angle(self):
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if len(sel_atoms) != 3:
            raise UserError("Must select exactly 3 atoms in graphics window")
        from chimerax.std_commands.angle import angle_atoms_check, SetAngleError
        try:
            moving, fixed, moving_atoms = angle_atoms_check(*sel_atoms)
        except SetAngleError as e:
            raise UserError(str(e))
        center = sel_atoms[1]
        canonical = (fixed, center, moving)
        if canonical in self.angle_data:
            raise UserError("The angle formed by the 3 selected atoms is already active.")

        self.aa_no_angles_label.hide()
        for header in self.aa_header_widgets:
            header.show()
        grid = self.aa_angles_layout
        row = grid.rowCount()
        widgets = []
        initial_angle = self._aa_angle_value(canonical)
        self.angle_data[canonical] = [row, fixed, center, moving, initial_angle, None, widgets]
        close_button = QToolButton()
        close_action = QAction(close_button)
        close_action.triggered.connect(lambda *args, angle=canonical, f=self._aa_remove_angle: f(angle))
        close_action.setIcon(self.session.ui.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        close_button.setDefaultAction(close_action)
        grid.addWidget(close_button, row, 0)
        widgets.append(close_button)

        for i, atom in enumerate(canonical):
            relative_to = None if i == 0 else canonical[i-1]
            text = atom.string(minimal=True, relative_to=relative_to)
            if i == 1:
                widget = QPushButton(text)
                menu = QMenu(widget)
                widget.setMenu(menu)
                reset = QAction("Reset to initial angle", menu)
                reset.triggered.connect(lambda *args, angle=canonical, f=self._aa_reset_angle: f(angle))
                menu.addAction(reset)
                swap = QAction("Swap moving/fixed sides", menu)
                swap.triggered.connect(lambda *args, angle=canonical, f=self._aa_swap_angle_sides: f(angle))
                menu.addAction(swap)
            else:
                widget = QLabel(text)
            grid.addWidget(widget, row, i+1, alignment=Qt.AlignCenter)
            widgets.append(widget)
        self._aa_set_widget_texts(canonical)
        angle_text = QDoubleSpinBox()
        angle_text.setDecimals(1)
        angle_text.setRange(0.0, 180.0)
        angle_text.setSingleStep(1.0)
        angle_text.setAlignment(Qt.AlignRight)
        angle_text.setValue(initial_angle)
        angle_text.valueChanged.connect(
            lambda val, *, angle=canonical, f=self._aa_issue_command: f(angle, val))
        angle_text.setKeyboardTracking(False) # don't get a signal for _every_ keystroke
        grid.addWidget(angle_text, row, 4, alignment=Qt.AlignLeft)
        widgets.append(angle_text)
        # QDial is integer, so x10...
        dial = QDial()
        dial.setMinimum(0)
        dial.setMaximum(1800)
        dial.setSingleStep(10)
        dial.setValue(int(initial_angle * 10 + 0.5))
        dial.valueChanged.connect(lambda val, *, angle=canonical, f=self._aa_dial_changed: f(angle, val))
        dial.sliderReleased.connect(
            lambda *, angle=canonical, dial=dial, f=self._aa_issue_command: f(angle, dial.value()/10))
        grid.addWidget(dial, row, 5, alignment=Qt.AlignCenter)
        self.session.dial = dial
        widgets.append(dial)
        self._aa_resize_dials()
        if len(self.angle_data) == 1:
            self.handlers['adjust angles'] = handlers = []
            from chimerax.atomic import get_triggers
            handlers.append(get_triggers().add_handler('changes', self._aa_atomic_changes_cb))
            from chimerax.core.models import MODEL_NAME_CHANGED, MODEL_ID_CHANGED, ADD_MODELS, \
                REMOVE_MODELS, MODEL_POSITION_CHANGED
            triggers = self.session.triggers
            handlers.append(triggers.add_handler(MODEL_NAME_CHANGED, self._aa_atomic_changes_cb))
            handlers.append(triggers.add_handler(MODEL_ID_CHANGED, self._aa_atomic_changes_cb))
            handlers.append(triggers.add_handler(ADD_MODELS, self._aa_atomic_changes_cb))
            handlers.append(triggers.add_handler(REMOVE_MODELS, self._aa_atomic_changes_cb))
            handlers.append(triggers.add_handler(MODEL_POSITION_CHANGED, self._aa_atomic_changes_cb))

    def _aa_angle_cmd(self, canonical):
        cmd = "angle "
        atoms = self.angle_data[canonical][1:4]
        prev = None
        for a in atoms:
            cmd += a.string(style="command", minimal=True, relative_to=prev)
            prev = a
        cmd += " %g"
        if atoms[0] != canonical[0]:
            cmd += " move large"
        return cmd

    def _aa_angle_value(self, canonical):
        from chimerax import geometry
        return geometry.angle(*[x.scene_coord for x in canonical])

    def _aa_atomic_changes_cb(self, trig_name, trig_data):
        update_values = False
        check_legal = False
        changed_structures = set()
        if trig_name == 'changes':
            update_names = 'name changed' in trig_data.atom_reasons() \
                or 'name changed' in trig_data.residue_reasons() \
                or 'name changed' in trig_data.chain_reasons()
            check_death = trig_data.num_deleted_atoms() > 0
            check_legal = trig_data.created_bonds(include_new_structures=False)

            if 'active_coordset changed' in trig_data.structure_reasons():
                changed_structures.update(trig_data.modified_structures())
            if 'coordset changed' in trig_data.coordset_reasons():
                changed_structures.update(trig_data.modified_coordsets().unique_structures)
            if 'coord changed' in trig_data.atom_reasons():
                changed_structures.update(trig_data.modified_atoms().unique_structures)
        else:
            from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, MODEL_POSITION_CHANGED
            if trig_name == ADD_MODELS:
                update_names = (len(self.session.models) - len(trig_data)) < 2
                check_death = False
            elif trig_name == REMOVE_MODELS:
                update_names = len(self.session.models)  < 2
                check_death = True
            elif trig_name == MODEL_POSITION_CHANGED:
                update_names = False
                check_death = False
                changed_structures.add(trig_data)
            else:
                update_names = True
                check_death = False

        death_row = []
        for canonical in self.angle_data.keys():
            if check_death:
                execute = False
                for atom in canonical:
                    if atom.deleted:
                        execute = True
                        break
                if execute:
                    death_row.append(canonical)
                    continue
            if check_legal:
                from chimera.std_commands.angle import angle_atoms_check, SetAngleError
                try:
                    angle_atoms_check(*canonical)
                except SetAngleError:
                    death_row.append(canonical)
                    continue
            if update_names:
                self._aa_set_widget_texts(canonical)
        for canonical in death_row:
            self._aa_remove_angle(canonical)
        if changed_structures:
            for canonical in self.angle_data.keys():
                for a in canonical:
                    if a.structure in changed_structures:
                        self._aa_update_angle_value(canonical)
                        break

    def _aa_dial_changed(self, canonical, val):
        dial_val = val/10
        from chimerax import geometry
        cur_angle = geometry.angle(*[x.scene_coord for x in canonical])
        if dial_val == cur_angle:
            return
        data = self.angle_data[canonical]
        atoms = data[1:4]
        move_smaller = atoms[0] == canonical[0]
        axis = data[-2]
        from chimerax.std_commands.angle import set_angle
        axis = set_angle(*atoms, dial_val, move_smaller=move_smaller, prev_axis=axis)
        data[-2] = axis

    def _aa_issue_command(self, canonical, val):
        angle_cmd_template = self._aa_angle_cmd(canonical)
        from chimerax.core.commands import run
        run(self.session, angle_cmd_template % val)

    def _aa_remove_angle(self, canonical):
        for widget in self.angle_data[canonical][-1]:
            widget.hide()
        del self.angle_data[canonical]
        if not self.angle_data:
            self.aa_no_angles_label.show()
            for header in self.aa_header_widgets:
                header.hide()
            handlers = self.handlers['adjust angles']
            for handler in handlers:
                handler.remove()
            handlers.clear()
        else:
            self._aa_resize_dials()

    def _aa_resize_dials(self):
        if not self.angle_data:
            return
        num_angles = len(self.angle_data)
        target_height = 250 / num_angles
        dial_size = int(min(100, max(target_height, 30)))
        for *args, widgets in self.torsion_data.values():
            widgets[-1].setFixedSize(dial_size, dial_size)

    def _aa_reset_angle(self, canonical):
        initial_angle = self.angle_data[canonical][-3]
        self._aa_issue_command(canonical, initial_angle)

    def _aa_swap_angle_sides(self, canonical):
        row, fixed, center, moving, start_angle, axis, widgets = self.angle_data[canonical]
        self.angle_data[canonical] = [row, moving, center, fixed, start_angle, axis, widgets]
        self._aa_set_widget_texts(canonical)

    def _aa_set_widget_texts(self, canonical):
        row, fixed, center, moving, start_angle, axis, widgets = self.angle_data[canonical]
        atoms = (fixed, center, moving)
        for i, atom in enumerate(atoms):
            relative_to = None if i == 0 else atoms[i-1]
            text = atom.string(minimal=True, relative_to=relative_to)
            widgets[i+1].setText(text)

    def _aa_update_angle_value(self, canonical, value=None):
        if value is None:
            angle_value = self._aa_angle_value(canonical)
        else:
            angle_value = value
        widgets = self.angle_data[canonical][-1]
        spin_box, dial = widgets[-2:]
        spin_box.blockSignals(True)
        spin_box.setValue(angle_value)
        spin_box.blockSignals(False)
        dial.blockSignals(True)
        dial.setValue(int(10 * angle_value + 0.5))
        dial.blockSignals(False)

    def _ab_len_cb(self, opt):
        self.bond_len_slider.blockSignals(True)
        self.bond_len_slider.setValue(opt.value)
        self.bond_len_slider.blockSignals(False)
        if not self._initial_bond_lengths:
            raise UserError("No bonds selected")
        if self.bond_len_side_button.text() == "larger side":
            arg = " move large"
        else:
            arg = ""
        from chimerax.core.commands import run
        for b in self._initial_bond_lengths.keys():
            run(self.session, ("bond length %s %g" + arg) % (b.atomspec, opt.value))

    def _ab_sel_changed(self, *args):
        seen = set()
        for bonds in self.session.selection.items('bonds'):
            seen.update(bonds)
        from chimerax.atomic import Atoms, Bonds
        for atoms in self.session.selection.items('atoms'):
            if not isinstance(atoms, Atoms):
                atoms = Atoms(atoms)
            seen.update(atoms.intra_bonds)
        from weakref import WeakKeyDictionary
        self._initial_bond_lengths = WeakKeyDictionary({b:b.length for b in seen})
        if not seen:
            return
        import numpy
        val = numpy.mean(Bonds(seen).lengths)
        self.bond_len_opt.value = val
        self.bond_len_slider.blockSignals(True)
        self.bond_len_slider.setValue(val)
        self.bond_len_slider.blockSignals(False)

    def _at_activate(self):
        from chimerax.atomic import selected_bonds
        sel_bonds = selected_bonds(self.session)
        if len(sel_bonds) != 1:
            raise UserError("Exactly one bond must be selected in graphics window")
        bond = sel_bonds[0]
        for end_pt in bond.atoms:
            if len(end_pt.neighbors) == 1:
                raise UserError("Bond must have other atoms bonded to both ends to form torsion")
        try:
            self.session.bond_rotations.new_rotation(bond, one_shot=False)
        except self.session.bond_rotations.BondRotationError as e:
            raise UserError(str(e))

    def _at_add_torsion(self, rotater):
        # _at_activate notifies the user about bond rotations that can't be torsions,
        # we have to check here again to silently drop non-torsion rotations coming from other sources...
        bond = rotater.bond
        self.at_no_torsions_label.hide()
        for header in self.at_header_widgets:
            header.show()
        moving = rotater.moving_side
        fixed = bond.other_atom(moving)
        torsion_atoms = []
        for end1, end2 in [(fixed, moving), (moving, fixed)]:
            for nb in end1.neighbors:
                if nb != end2:
                    torsion_atoms.append(nb)
                    break
        grid = self.at_torsions_layout
        row = grid.rowCount()
        widgets = []
        self.torsion_data[rotater] = (row, bond, torsion_atoms, widgets)
        close_button = QToolButton()
        close_action = QAction(close_button)
        close_action.triggered.connect(lambda *args, rotater=rotater, manager=self.session.bond_rotations:
            manager.delete_rotation(rotater))
        close_action.setIcon(self.session.ui.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        close_button.setDefaultAction(close_action)
        grid.addWidget(close_button, row, 0)

        def multi_name(terminus, bonded, excluded):
            for nb in bonded.neighbors:
                if nb == terminus or nb == excluded:
                    continue
                if nb.name == terminus.name:
                    return terminus.string(relative_to=bonded)
            return terminus.name
        for col, torsion_index, bonded in [(1, 0, fixed), (3, 1, moving)]:
            terminus = torsion_atoms[torsion_index]
            single_widget = QLabel(terminus.name)
            multi_widget = QPushButton(multi_name(terminus, bonded, bond.other_atom(bonded)))
            multi_menu = QMenu(multi_widget)
            multi_menu.aboutToShow.connect(lambda *args, menu=multi_menu, end=bonded, bond=bond,
                torsion_atoms=torsion_atoms, index=torsion_index, rotater=rotater:
                self._at_compose_menu(menu, end, bond, index, rotater))
            multi_widget.setMenu(multi_menu)
            show_multi = bonded.num_bonds > 2
            grid.addWidget(single_widget, row, col, alignment=Qt.AlignCenter)
            grid.addWidget(multi_widget, row, col, alignment=Qt.AlignCenter)
            single_widget.setHidden(show_multi)
            multi_widget.setHidden(not show_multi)
            widgets.extend([single_widget, multi_widget])
        bond_widget = QPushButton(self._at_bond_text(rotater))
        grid.addWidget(bond_widget, row, 2, alignment=Qt.AlignCenter)
        bond_menu = QMenu(bond_widget)
        bond_widget.setMenu(bond_menu)
        reset = QAction("Reset to initial torsion angle", bond_menu)
        reset.triggered.connect(lambda *args, rotater=rotater, f=self._at_log_command:
            (setattr(rotater, 'angle', 0), f(rotater)))
        bond_menu.addAction(reset)
        swap = QAction("Swap moving/fixed sides", bond_menu)
        swap.triggered.connect(lambda *args, rotater=rotater: rotater.swap_sides())
        bond_menu.addAction(swap)
        widgets.append(bond_widget)
        widgets.append(close_button)
        angle_text = QDoubleSpinBox()
        angle_text.setDecimals(1)
        angle_text.setRange(-180.0, 180.0)
        angle_text.setSingleStep(1.0)
        angle_text.setWrapping(True)
        angle_text.setAlignment(Qt.AlignRight)
        angle_text.valueChanged.connect(
            lambda val, *, rotater=rotater, f=self._at_issue_command: f(rotater, val))
        angle_text.setKeyboardTracking(False) # don't get a signal for _every_ keystroke
        grid.addWidget(angle_text, row, 4, alignment=Qt.AlignLeft)
        widgets.append(angle_text)
        # QDial is integer, so x10...
        dial = QDial()
        dial.setMinimum(-1800)
        dial.setMaximum(1800)
        dial.setSingleStep(10)
        dial.setWrapping(True)
        dial.valueChanged.connect(lambda val, *, rotater=rotater, f=self._at_dial_changed: f(rotater, val))
        dial.sliderReleased.connect(
            lambda *, rotater=rotater, dial=dial, f=self._at_issue_command: f(rotater, dial.value()/10))
        grid.addWidget(dial, row, 5, alignment=Qt.AlignCenter)
        self.session.dial = dial
        widgets.append(dial)
        self._at_resize_dials()
        if len(self.torsion_data) == 1:
            # first torsion; add handlers
            self.handlers['adjust torsions: per torsion'] = handlers = []
            from chimerax.atomic import get_triggers
            handlers.append(get_triggers().add_handler('changes', self._at_atomic_changes_cb))
            from chimerax.core.models import MODEL_NAME_CHANGED, MODEL_ID_CHANGED, ADD_MODELS, REMOVE_MODELS
            triggers = self.session.triggers
            handlers.append(triggers.add_handler(MODEL_NAME_CHANGED, self._at_atomic_changes_cb))
            handlers.append(triggers.add_handler(MODEL_ID_CHANGED, self._at_atomic_changes_cb))
            handlers.append(triggers.add_handler(ADD_MODELS, self._at_atomic_changes_cb))
            handlers.append(triggers.add_handler(REMOVE_MODELS, self._at_atomic_changes_cb))

        self._at_update_torsion_value(rotater)

    def _at_atomic_changes_cb(self, trig_name, trig_data):
        if trig_name == 'changes':
            update_names = 'name changed' in trig_data.atom_reasons() \
                or 'name changed' in trig_data.residue_reasons() \
                or 'name changed' in trig_data.chain_reasons()
            update_ends = trig_data.created_atoms() or trig_data.num_deleted_atoms() > 0
        else:
            from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
            if trig_name == ADD_MODELS:
                update_names = (len(self.session.models) - len(trig_data)) < 2
            elif trig_name == REMOVE_MODELS:
                update_names = len(self.session.models)  < 2
            else:
                update_names = True
            update_ends = False
        if not update_names and not update_ends:
            return

        death_row = []
        for rotater, data in self.torsion_data.items():
            row, bond, torsion_atoms, widgets = data
            if bond.deleted:
                # torsion manager will call back to us for this
                continue
            bond_widget = widgets[4]
            moving = rotater.moving_side
            reversed = bond.atoms[0] == moving
            if update_ends:
                for end in bond.atoms:
                    if len(end.neighbors) < 2:
                        death_row.append(rotater)
                        break
                if rotater in death_row:
                    continue
                for torsion_index, torsion_atom in enumerate(torsion_atoms):
                    end = moving if torsion_index == 1 else bond.other_atom(moving)
                    if torsion_atom.deleted:
                        for nb in end.neighbors:
                            if nb != bond.other_atom(end):
                                self._at_change_torsion_atom(nb, rotater, torsion_index)
                                break
                    offset = 2 * torsion_index
                    single_widget, multi_widget = widgets[offset:offset+2]
                    is_single = len(end.neighbors) == 2
                    single_widget.setHidden(not is_single)
                    multi_widget.setHidden(is_single)
                if update_names:
                    bond_widget.setText(bond.string(minimal=True, reversed=reversed))
            elif update_names:
                bond_widget.setText(bond.string(minimal=True, reversed=reversed))
                for torsion_index, torsion_atom in enumerate(torsion_atoms):
                    end = moving if torsion_index == 1 else bond.other_atom(moving)
                    offset = 2 * torsion_index
                    single_widget, multi_widget = widgets[offset:offset+2]
                    text = torsion_atom.string(relative_to=end, minimal=True)
                    single_widget.setText(text)
                    multi_widget.setText(text)
        for rotater in death_row:
            self.session.bond_rotations.delete_rotation(rotater)

    def _at_bond_text(self, rotater):
        (row, bond, torsion_atoms, widgets) = self.torsion_data[rotater]
        return bond.string(minimal=True, reversed=(rotater.moving_side == bond.atoms[0]))

    def _at_change_torsion_atom(self, torsion_atom, rotater, torsion_index, text=None, update_value=True):
        (row, bond, torsion_atoms, widgets) = self.torsion_data[rotater]
        torsion_atoms[torsion_index] = torsion_atom
        offset = 2 * torsion_index
        if text is None:
            if torsion_index == 0:
                end = bond.other_atom(rotater.moving_side)
            else:
                end = rotater.moving_side
            text = torsion_atom.string(relative_to=end, minimal=True)
        for widget in widgets[offset:offset+2]:
            widget.setText(text)
        if update_value and not torsion_atoms[1-torsion_index].deleted:
            self._at_update_torsion_value(rotater)

    def _at_compose_menu(self, menu, end_atom, bond, torsion_index, rotater):
        menu.clear()
        other_atom = bond.other_atom(end_atom)
        for nb in end_atom.neighbors:
            if nb == other_atom:
                continue
            action = QAction(menu)
            action.setText(nb.string(relative_to=end_atom, minimal=True))
            action.triggered.connect(lambda *args, ta=nb, torsion_index=torsion_index, rotater=rotater,
                text=action.text(), f=self._at_change_torsion_atom: f(ta, rotater, torsion_index, text))
            menu.addAction(action)

    def _at_dial_changed(self, rotater, val):
        delta = val/10 - self._at_torsion_value(rotater)
        if delta != 0:
            rotater.angle += delta

    def _at_issue_command(self, rotater, val):
        torsion_cmd_template = self._at_torsion_cmd(rotater)
        from chimerax.core.commands import run
        run(self.session, torsion_cmd_template % val)

    def _at_log_command(self, rotater):
        torsion_cmd_template = self._at_torsion_cmd(rotater)
        from chimerax.core.commands import Command
        Command(self.session).run(torsion_cmd_template % self._at_torsion_value(rotater), log_only=True)

    def _at_remove_torsion(self, rotater):
        for widget in self.torsion_data[rotater][-1]:
            widget.hide()
        del self.torsion_data[rotater]
        if not self.torsion_data:
            self.at_no_torsions_label.show()
            for header in self.at_header_widgets:
                header.hide()
            handlers = self.handlers['adjust torsions: per torsion']
            for handler in handlers:
                handler.remove()
            handlers.clear()
        else:
            self._at_resize_dials()

    def _at_resize_dials(self):
        if not self.torsion_data:
            return
        num_torsions = len(self.torsion_data)
        target_height = 250 / num_torsions
        dial_size = int(min(100, max(target_height, 30)))
        for row, bond, torsion_atoms, widgets in self.torsion_data.values():
            widgets[-1].setFixedSize(dial_size, dial_size)

    def _at_swap_sides(self, rotater):
        (row, bond, torsion_atoms, widgets) = self.torsion_data[rotater]
        torsion_atoms[:] = [torsion_atoms[1], torsion_atoms[0]]
        fixed_widgets, moving_widgets = widgets[:2], widgets[2:4]
        for widget in widgets[:4]:
            self.at_torsions_layout.removeWidget(widget)
        widgets[:2] = moving_widgets
        widgets[2:4] = fixed_widgets
        for col, torsion_widgets in [(3, fixed_widgets), (1, moving_widgets)]:
            for torsion_widget in torsion_widgets:
                self.at_torsions_layout.addWidget(torsion_widget, row, col, alignment=Qt.AlignCenter)

        bond_widget = widgets[4]
        bond_widget.setText(self._at_bond_text(rotater))
        self._at_update_torsion_value(rotater)

    def _at_torsion_atoms(self, rotater):
        (row, bond, torsion_atoms, widgets) = self.torsion_data[rotater]
        moving = rotater.moving_side
        fixed = bond.other_atom(moving)
        t1, t2 = torsion_atoms
        return t1, fixed, moving, t2

    def _at_torsion_cmd(self, rotater):
        cmd = "torsion "
        prev = None
        for a in self._at_torsion_atoms(rotater):
            cmd += a.string(style="command", minimal=True, relative_to=prev)
            prev = a
        cmd += " %g"
        if rotater.moving_side != rotater.bond.smaller_side:
            cmd += " move large"
        return cmd

    def _at_torsion_value(self, rotater):
        (row, bond, torsion_atoms, widgets) = self.torsion_data[rotater]
        spin_box, dial = widgets[-2:]
        t1, fixed, moving, t2 = self._at_torsion_atoms(rotater)
        from chimerax.geometry import dihedral
        return dihedral(t1.coord, fixed.coord, moving.coord, t2.coord)

    def _at_update_torsion_value(self, rotater):
        torsion_value = self._at_torsion_value(rotater)
        (row, bond, torsion_atoms, widgets) = self.torsion_data[rotater]
        spin_box, dial = widgets[-2:]
        spin_box.blockSignals(True)
        spin_box.setValue(torsion_value)
        spin_box.blockSignals(False)
        dial.blockSignals(True)
        dial.setValue(10 * torsion_value)
        dial.blockSignals(False)

    def _cat_menu_cb(self, action):
        self.category_areas.setCurrentWidget(self.category_widgets[action.text()])
        self.category_button.setText(action.text())

    def _jm_apply_cb(self):
        from chimerax.atomic import selected_atoms
        if not selected_atoms(self.session):
            raise UserError("No atoms selected")
        length = self.jp_bond_len_opt.value
        omega = self.jp_omega_opt.value
        phi = self.jp_phi_opt.value
        side = self.jp_side_button.text()
        if side.endswith("smaller"):
            side = side[:-2]
        elif side.endswith("larger"):
            side = side[:-1]
        elif side.startswith("selected"):
            side = side[9]
        from .mod import BindError
        try:
            run(self.session, "build join peptide sel length %g omega %g phi %g move %s"
                % (length, omega, phi, side))
        except BindError as e:
            raise UserError(e)

    def _invert_swap_cb(self):
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if len(sel_atoms) not in [1,2]:
            raise UserError("You must select 1 or 2 atoms; you selected %d" % len(sel_atoms))
        run(self.session, "build invert sel")

    def _layout_adjust_bonds(self, parent):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        res_group = QGroupBox("Add/Delete")
        layout.addWidget(res_group, alignment=Qt.AlignHCenter|Qt.AlignTop)
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(0,0,0,0)
        group_layout.setSpacing(0)
        res_group.setLayout(group_layout)
        del_layout = QHBoxLayout()
        group_layout.addLayout(del_layout)
        del_button = QPushButton("Delete")
        del_button.clicked.connect(lambda *args, ses=self.session: run(ses, "~bond sel"))
        del_layout.addWidget(del_button)
        del_layout.addWidget(QLabel("selected bonds"), stretch=1, alignment=Qt.AlignLeft)
        add_layout = QHBoxLayout()
        group_layout.addLayout(add_layout)
        add_button = QPushButton("Add")
        add_layout.addWidget(add_button)
        type_button = QPushButton("reasonable")
        type_menu = QMenu(parent)
        type_menu.addAction("reasonable")
        type_menu.addAction("all possible")
        type_menu.triggered.connect(lambda act, but=type_button: but.setText(act.text()))
        type_button.setMenu(type_menu)
        def add_but_clicked(*args, but=type_button):
            if but.text() != "reasonable":
                from chimerax.core.commands import BoolArg
                kw = " reasonable %s" % BoolArg.unparse(False)
            else:
                kw = ""
            run(self.session, "bond sel" + kw)
        add_button.clicked.connect(add_but_clicked)
        add_layout.addWidget(type_button)
        add_layout.addWidget(QLabel("bonds between selected atoms"))

        len_group = QGroupBox("Set Length")
        layout.addWidget(len_group, alignment=Qt.AlignHCenter|Qt.AlignTop)
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(0,0,0,0)
        group_layout.setSpacing(0)
        len_group.setLayout(group_layout)
        numeric_area = QWidget()
        group_layout.addWidget(numeric_area, alignment=Qt.AlignCenter)
        numeric_layout = QHBoxLayout()
        numeric_layout.setContentsMargins(0,0,0,0)
        numeric_layout.setSpacing(0)
        numeric_area.setLayout(numeric_layout)
        from chimerax.ui.options import OptionsPanel, FloatOption
        precision = 3
        self.bond_len_opt = FloatOption("Set length of selected bonds to", 1.5, self._ab_len_cb,
            min="positive", max=99, decimal_places=precision)
        panel = OptionsPanel(scrolled=False)
        numeric_layout.addWidget(panel, alignment=Qt.AlignRight)
        panel.add_option(self.bond_len_opt)
        from chimerax.ui.widgets import FloatSlider
        self.bond_len_slider = FloatSlider(0.5, 4.5, 0.1, precision, True)
        self.bond_len_slider.set_left_text("0.5")
        self.bond_len_slider.set_right_text("4.5")
        self.bond_len_slider.setValue(1.5)
        numeric_layout.addWidget(self.bond_len_slider)
        self.bond_len_slider.valueChanged.connect(
            lambda val, *, opt=self.bond_len_opt: setattr(opt, "value", val) or opt.make_callback())
        side_area = QWidget()
        group_layout.addWidget(side_area, alignment=Qt.AlignCenter)
        side_layout = QHBoxLayout()
        side_layout.setContentsMargins(0,0,0,0)
        side_layout.setSpacing(0)
        side_area.setLayout(side_layout)
        side_layout.addWidget(QLabel("Move atoms on"), alignment=Qt.AlignRight)
        self.bond_len_side_button = QPushButton()
        menu = QMenu()
        self.bond_len_side_button.setMenu(menu)
        menu.addAction("smaller side")
        menu.addAction("larger side")
        menu.triggered.connect(lambda act, *, but=self.bond_len_side_button: but.setText(act.text()))
        self.bond_len_side_button.setText("smaller side")
        side_layout.addWidget(self.bond_len_side_button)
        revert_area = QWidget()
        group_layout.addWidget(revert_area, alignment=Qt.AlignCenter)
        revert_layout = QHBoxLayout()
        revert_layout.setContentsMargins(0,0,0,0)
        revert_layout.setSpacing(0)
        revert_area.setLayout(revert_layout)
        but = QPushButton()
        but.setText("Revert")
        but.clicked.connect(self._revert_lengths)
        revert_layout.addWidget(but, alignment=Qt.AlignRight)
        revert_layout.addWidget(QLabel("lengths"), alignment=Qt.AlignLeft)

        from chimerax.core.selection import SELECTION_CHANGED
        self.handlers['adjust bonds'] = [
            self.session.triggers.add_handler(SELECTION_CHANGED, self._ab_sel_changed)]
        self._ab_sel_changed()

    def _layout_adjust_angles(self, parent):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addStretch(1)
        parent.setLayout(layout)
        activate_layout = QHBoxLayout()
        activate_layout.addStretch(1)
        activate_button = QPushButton("Activate")
        activate_button.clicked.connect(self._aa_add_angle)
        activate_layout.addWidget(activate_button)
        activate_layout.addWidget(QLabel(" angle formed by 3 selected atoms"))
        activate_layout.addStretch(1)
        layout.addLayout(activate_layout)
        layout.addStretch(1)
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(5)
        grid_layout.addStretch(1)
        self.aa_angles_layout = grid = QGridLayout()
        self.aa_header_widgets = []
        for col, text in enumerate(["Fixed", "Middle", "Moving"]):
            label = QLabel(text)
            grid.addWidget(label, 0, col+1, alignment=Qt.AlignCenter)
            label.hide()
            self.aa_header_widgets.append(label)
        grid_layout.addLayout(grid)
        grid_layout.addStretch(1)
        layout.addLayout(grid_layout)
        layout.addStretch(1)
        self.aa_no_angles_label = QLabel("No angles active")
        self.aa_no_angles_label.setEnabled(False)
        layout.addWidget(self.aa_no_angles_label, alignment=Qt.AlignCenter)
        layout.addStretch(1)

        self.angle_data = {}

    def _layout_adjust_torsions(self, parent):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addStretch(1)
        parent.setLayout(layout)
        activate_layout = QHBoxLayout()
        activate_layout.addStretch(1)
        activate_button = QPushButton("Activate")
        activate_button.clicked.connect(self._at_activate)
        activate_layout.addWidget(activate_button)
        activate_layout.addWidget(QLabel(" selected bond as torsion"))
        activate_layout.addStretch(1)
        layout.addLayout(activate_layout)
        layout.addStretch(1)
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(5)
        grid_layout.addStretch(1)
        self.at_torsions_layout = grid = QGridLayout()
        self.at_header_widgets = []
        for col, text in enumerate(["Fixed", "Bond", "Moving"]):
            label = QLabel(text)
            grid.addWidget(label, 0, col+1, alignment=Qt.AlignCenter)
            label.hide()
            self.at_header_widgets.append(label)
        grid_layout.addLayout(grid)
        grid_layout.addStretch(1)
        layout.addLayout(grid_layout)
        layout.addStretch(1)
        self.at_no_torsions_label = QLabel("No torsions active")
        self.at_no_torsions_label.setEnabled(False)
        layout.addWidget(self.at_no_torsions_label, alignment=Qt.AlignCenter)
        layout.addStretch(1)

        self.handlers['adjust torsions: base'] = handlers = []
        manager = self.session.bond_rotations
        handlers.append(manager.triggers.add_handler(manager.CREATED,
            lambda trig_name, rotator, f=self._at_add_torsion: f(rotator)))
        handlers.append(manager.triggers.add_handler(manager.DELETED,
            lambda trig_name, rotator, f=self._at_remove_torsion: f(rotator)))
        handlers.append(manager.triggers.add_handler(manager.MODIFIED,
            lambda trig_name, rotator, f=self._at_update_torsion_value: f(rotator)))
        handlers.append(manager.triggers.add_handler(manager.REVERSED,
            lambda trig_name, rotator, f=self._at_swap_sides: f(rotator)))
        self.torsion_data = {}
        for bond, rotation in manager.bond_rotations.items():
            for end_pt in bond.atoms:
                if not end_pt.neighbors:
                    break
            else:
                for rotater in rotation.rotaters:
                    if not rotater.one_shot:
                        self._at_add_torsion(rotater)

    def _layout_invert(self, parent):
        layout = QVBoxLayout()
        parent.setLayout(layout)

        instructions = QLabel("Select one atom to swap the two smallest subsituents bonded to that atom,"
            " or select two atoms bonded to the same atom to swap those specific substituents",
            alignment=Qt.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        swap_button = QPushButton("Swap")
        swap_button.clicked.connect(lambda checked: self._invert_swap_cb())
        layout.addWidget(swap_button, alignment=Qt.AlignHCenter|Qt.AlignTop, stretch=1)

    def _layout_join_models(self, parent):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        self.peptide_group = QGroupBox("Peptide Parameters")
        layout.addWidget(self.peptide_group, alignment=Qt.AlignHCenter|Qt.AlignTop)
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(0,0,0,0)
        group_layout.setSpacing(0)
        self.peptide_group.setLayout(group_layout)

        peptide_instructions = QLabel("Form bond between selected C-terminal carbon and N-terminal nitrogen"
            " as follows:", alignment=Qt.AlignCenter)
        peptide_instructions.setWordWrap(True)
        group_layout.addWidget(peptide_instructions)
        from chimerax.ui.options import OptionsPanel, FloatOption
        panel = OptionsPanel(scrolled=False, sorting=False)
        group_layout.addWidget(panel, alignment=Qt.AlignCenter)
        self.jp_bond_len_opt = FloatOption("C-N length:", 1.33, None, min="positive", decimal_places=3)
        panel.add_option(self.jp_bond_len_opt)
        self.jp_omega_opt = FloatOption("C\N{GREEK SMALL LETTER ALPHA}-C-N-C\N{GREEK SMALL LETTER ALPHA}"
            " dihedral (\N{GREEK SMALL LETTER OMEGA} angle):", 180.0, None, decimal_places=1)
        panel.add_option(self.jp_omega_opt)
        self.jp_phi_opt = FloatOption("C-N-C\N{GREEK SMALL LETTER ALPHA}-C"
            " dihedral (\N{GREEK SMALL LETTER PHI} angle):", -120.0, None, decimal_places=1)
        panel.add_option(self.jp_phi_opt)
        side_layout = QHBoxLayout()
        group_layout.addLayout(side_layout)
        side_layout.addWidget(QLabel("Move atoms in ", alignment=Qt.AlignRight|Qt.AlignVCenter))
        self.jp_side_button = QPushButton("smaller")
        side_layout.addWidget(self.jp_side_button)
        side_menu = QMenu(self.jp_side_button)
        for side_text in ["selected N atom", "selected C atom", "smaller", "larger"]:
            side_menu.addAction(QAction(side_text, side_menu))
        side_menu.triggered.connect(lambda act, but=self.jp_side_button: but.setText(act.text()))
        self.jp_side_button.setMenu(side_menu)
        side_layout.addWidget(QLabel(" model", alignment=Qt.AlignLeft|Qt.AlignVCenter))
        peptide_disclaimer = QLabel("Selected N- and C-terminus must be in different models",
            alignment=Qt.AlignCenter)
        from chimerax.ui import shrink_font
        shrink_font(peptide_disclaimer)
        group_layout.addWidget(peptide_disclaimer)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(lambda checked: self._jm_apply_cb())
        layout.addWidget(apply_button, alignment=Qt.AlignHCenter|Qt.AlignTop)

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
        self._prev_mod_res = None
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
        self.handlers['modify structure'] = [
            self.session.triggers.add_handler(SELECTION_CHANGED, self._ms_sel_changed)]
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

        # manager may have alredy been started by command...
        self._ignore_new_providers = True
        from .manager import get_manager
        manager = get_manager(self.session)
        self.ss_u_to_p_names = { manager.ui_name(pn):pn for pn in manager.provider_names }
        ui_names = list(self.ss_u_to_p_names.keys())
        ui_names.sort(key=lambda x: x.lower())
        self._start_provider_layout = provider_layout = QGridLayout()
        provider_layout.setVerticalSpacing(5)
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
        self._ignore_new_providers = False

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

        self._prev_mod_res = None
        if self.ms_res_mod.isChecked():
            res_name = self.ms_mod_edit.text().strip()
            if not res_name:
                raise UserError("Must provided modified residue name")
            if res_name != a.residue.name:
                cmd += " resName " + res_name
            self._prev_mod_res = a.residue
        elif self.ms_res_new.isChecked():
            res_name = self.ms_res_new_name.text().strip()
            if not res_name:
                raise UserError("Must provided new residue name")
            cmd += " newRes true resName " + res_name

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
        if self._prev_mod_res != a.residue:
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

    def _new_start_providers(self, new_provider_names):
        if self._ignore_new_providers:
            return
        from .manager import get_manager
        manager = get_manager(self.session)
        num_prev = len(self.ss_u_to_p_names)
        new_u_to_p_names = { manager.ui_name(pn):pn for pn in new_provider_names }
        self.ss_u_to_p_names.update(new_u_to_p_names)
        ui_names = list(new_u_to_p_names.keys())
        ui_names.sort(key=lambda x: x.lower())

        for row, ui_name in enumerate(ui_names):
            row += num_prev
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

    def _revert_lengths(self):
        from chimerax.atomic.struct_edit import set_bond_length
        for b, l in self._initial_bond_lengths.items():
            if b.deleted:
                return
            set_bond_length(b, l)
        self._ab_sel_changed()

    def _ss_apply_cb(self):
        ui_name = self.ss_button_group.checkedButton().text()
        provider_name = self.ss_u_to_p_names[ui_name]

        from chimerax.core.errors import CancelOperation
        from .manager import get_manager
        manager = get_manager(self.session)
        try:
            subcmd_string = manager.get_command_substring(provider_name, self.ss_widgets[ui_name])
        except CancelOperation:
            return
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
        from .manager import get_manager
        manager = get_manager(self.session)
        provider_name = self.ss_u_to_p_names[ui_name]
        hide_model_choice = manager.new_model_only(provider_name)
        if manager.is_indirect(provider_name):
            self.ss_apply_button.setHidden(True)
            hide_model_choice = True
        else:
            self.ss_apply_button.setHidden(False)
        # hincky code to avoid flashing up widgets that end up hidden
        num_widgets = len(self.ss_struct_widgets)
        hiddens = [hide_model_choice] * num_widgets
        if not hide_model_choice and self.ss_struct_menu.value != "new model":
            hiddens[2:] = [True] * (num_widgets - 2)

        for widget, hidden in zip(self.ss_struct_widgets, hiddens):
            widget.setHidden(hidden)

    def _ss_struct_changed(self):
        show = self.ss_struct_menu.value == "new model"
        self.ss_model_name_label.setHidden(not show)
        self.ss_model_name_edit.setHidden(not show)
