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

from .settings import defaults
from PyQt5.QtWidgets import QWidget
from chimerax.ui.options import OptionsPanel, ColorOption, FloatOption, BooleanOption, IntOption, \
    StringOption, OptionalRGBAOption, make_optional

from chimerax.atomic import AtomicStructure
from chimerax.ui.options import Option

class AtomProximityGUI(QWidget):
    def __init__(self, session, name, prox_word, cmd_name, color, radius, hbond_allowance, overlap_cutoff,
            has_apply_button, *, settings_name="",

            # settings_name values:
            #   empty string: remembered across sessions and the same as for the main contacts/clashes
            #     GUI
            #   custom string (e.g. "rotamers"):  remembered across sessions and specific to your
            #     interface, not shared with other contacts/clashes GUIs.  The string "contacts GUI"
            #     or "clashes GUI" will be appended to the provided string to yield the final settings
            #     name.
            #   None: not remembered across sessions
            #
            # The settings will be saved when get_command() is called.  The defaults for the
            # settings will be the same as the values provided for the arguments below (i.e.
            # can be overriden by providing explicit values when calling this function).
            #
            # Controls configured to not show in the interface will not have their corresponding
            # keyword/value added to the command returned by get_command, nor will they have their
            # settings value changed/saved.  If needed, you will have to add the keyword/value to
            # the command yourself.
            #
            # Your tool needs to call the GUI's destroy() method when it's deleted
            atom_color=defaults["atom_color"], attr_name=defaults["attr_name"],
            bond_separation=defaults["bond_separation"], color_atoms=defaults["action_color"],
            continuous=False, dashes=None, distance_only=None, inter_model=True, inter_submodel=False,
            intra_mol=defaults["intra_mol"], intra_res=defaults["intra_res"], log=defaults["action_log"],
            make_pseudobonds=defaults["action_pseudobonds"], other_atom_color=defaults["other_atom_color"],
            res_separation=None, reveal=False, save_file=None, select=defaults["action_select"],
            set_attrs=defaults["action_attr"], show_dist=False, summary=True, test="others", test_atoms=None,

            # what controls to show in the interface
            # note that if 'test_atoms' is not None, then the "Atoms to Check" section will be omitted
            show_atom_color=True, show_attr_name=True, show_bond_separation=True,
            show_checking_frequency=True, show_color=True, show_color_atoms=True, show_dashes=True,
            show_distance_only=True, show_hbond_allowance=True, show_inter_model=True,
            show_inter_submodel=False, show_intra_mol=True, show_intra_res=True, show_log=True,
            show_make_pseudobonds=True, show_name=True, show_other_atom_color=True,
            show_overlap_cutoff=True, show_radius=True, show_res_separation=True, show_reveal=True,
            show_save_file=True, show_select=True, show_set_attrs=True, show_show_dist=True,
            show_summary=False, show_test=True):

        self.session = session

        from inspect import getargvalues, currentframe
        arg_names, var_args, var_kw, frame_dict = getargvalues(currentframe())
        settings_defaults = {}
        self.__show_values = {}
        for arg_name in arg_names:
            if not arg_name.startswith('show_') or 'show_' + arg_name in arg_names:
                settings_defaults[arg_name] = frame_dict[arg_name]
            else:
                self.__show_values[arg_name[5:]] = frame_dict[arg_name]
        if settings_name is None:
            self.__settings = settings = None
        else:
            self.__settings = settings = _get_settings(session, settings_name, settings_defaults, cmd_name)
        final_val = {}
        for def_name in settings_defaults.keys():
            final_val[def_name] = getattr(settings, def_name) if settings else frame_dict[def_name]

        super().__init__()
        from PyQt5.QtWidgets import QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QButtonGroup
        from PyQt5.QtWidgets import QRadioButton, QAbstractButton, QHBoxLayout, QDoubleSpinBox, QMenu
        from PyQt5.QtWidgets import QSpinBox, QCheckBox
        from PyQt5.QtCore import Qt
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.setLayout(layout)
        self.handlers = {}

        if test_atoms is None:
            self.desig1_atoms = None
            group = QGroupBox("Atoms to check")
            layout.addWidget(group)
            group_layout = QVBoxLayout()
            group_layout.setContentsMargins(0,0,0,0)
            #group_layout.setSpacing(0)
            group.setLayout(group_layout)
            desig1_layout = QGridLayout()
            desig1_layout.setContentsMargins(0,0,0,0)
            desig1_layout.setSpacing(0)
            desig1_layout.setColumnStretch(0, 1)
            desig1_layout.setColumnStretch(1, 0)
            desig1_layout.setColumnStretch(2, 0)
            desig1_layout.setColumnStretch(3, 1)
            group_layout.addLayout(desig1_layout)
            desig1_button = QPushButton("Designate")
            desig1_button.clicked.connect(self._designate1_cb)
            desig1_layout.addWidget(desig1_button, 0, 1, alignment=Qt.AlignRight)
            desig1_layout.addWidget(QLabel("currently selected atoms for checking"),
                0, 2, alignment=Qt.AlignLeft)
            self.desig1_status = QLabel()
            from chimerax.ui import shrink_font
            shrink_font(self.desig1_status)
            self._update_desig1_status()
            desig1_layout.addWidget(self.desig1_status, 1, 1, 1, 2, alignment=Qt.AlignCenter|Qt.AlignTop)

            if show_test:
                self.desig2_atoms = None
                test_layout = QGridLayout()
                test_layout.setContentsMargins(0,0,0,0)
                test_layout.setSpacing(5)
                test_layout.setColumnStretch(0, 1)
                test_layout.setColumnStretch(3, 1)
                group_layout.addLayout(test_layout)
                self.test_kw_to_label = {
                    'self': "themselves",
                    'others': "all other atoms",
                    None: "second set of designated atoms"
                }
                self.test_label_to_kw = { v:k for k,v in self.test_kw_to_label.items() }
                test_label = QLabel("Check designated\natoms against:")
                test_label.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
                test_layout.addWidget(test_label, 0, 1, len(self.test_kw_to_label), 1,
                    alignment=Qt.AlignRight)
                self.test_button_group = QButtonGroup()
                for i, kw in enumerate(['self', 'others', None]):
                    but = QRadioButton(self.test_kw_to_label[kw])
                    test_layout.addWidget(but, i, 2, alignment=Qt.AlignLeft)
                    if final_val['test'] == kw:
                        but.setChecked(True)
                    self.test_button_group.addButton(but, i)
                self.test_button_group.buttonClicked[QAbstractButton].connect(self._test_change)
                desig2_layout = QGridLayout()
                desig2_layout.setContentsMargins(0,0,0,0)
                desig2_layout.setSpacing(0)
                test_layout.addLayout(desig2_layout, 3, 1, 1, 2, alignment=Qt.AlignCenter)
                self.desig2_button = QPushButton("Designate")
                self.desig2_button.clicked.connect(self._designate2_cb)
                desig2_layout.addWidget(self.desig2_button, 0, 0, alignment=Qt.AlignRight)
                self.desig2_label = QLabel("selection as second set")
                desig2_layout.addWidget(self.desig2_label, 0, 1, alignment=Qt.AlignLeft)

                self.desig2_status = QLabel()
                shrink_font(self.desig2_status)
                self._update_desig2_status()
                desig2_layout.addWidget(self.desig2_status, 1, 0, 1, 2, alignment=Qt.AlignCenter)
                self._test_change(self.test_button_group.checkedButton())

        from chimerax.core.commands import plural_of
        prox_words = plural_of(prox_word)
        show_boolean_params = show_intra_mol or show_intra_res or show_inter_model or show_inter_submodel
        if show_overlap_cutoff or show_hbond_allowance or show_bond_separation or show_boolean_params:
            group = QGroupBox("%s parameters" % prox_word.capitalize())
            layout.addWidget(group)
            group_layout = QVBoxLayout()
            group_layout.setContentsMargins(0,0,0,0)
            group_layout.setSpacing(5)
            group.setLayout(group_layout)
            overlap_widgets = []
            distance_only_widgets = []
            if (show_overlap_cutoff or show_hbond_allowance) and show_distance_only:
                test_type_layout = QGridLayout()
                test_type_layout.setContentsMargins(0,0,0,0)
                test_type_layout.setSpacing(0)
                test_type_layout.setColumnStretch(1, 1)
                group_layout.addLayout(test_type_layout)
                overlap_rows = show_overlap_cutoff + show_hbond_allowance
                self.overlap_radio = QRadioButton("")
                test_type_layout.addWidget(self.overlap_radio, 0, 0, overlap_rows, 1)
                self.distance_radio = QRadioButton("")
                test_type_layout.addWidget(self.distance_radio, overlap_rows, 0)
                def enable_widgets(enabled, enabled_widgets, disabled_widgets):
                    for ew in enabled_widgets:
                        ew.setEnabled(enabled)
                    for dw in disabled_widgets:
                        dw.setEnabled(not enabled)
                self.overlap_radio.toggled.connect(lambda chk, *, ew=overlap_widgets,
                    dw=distance_only_widgets, f=enable_widgets: f(chk, ew, dw))
                self.distance_radio.toggled.connect(lambda chk, *, ew=distance_only_widgets,
                    dw=overlap_widgets, f=enable_widgets: f(chk, ew, dw))
                row = 0
                if show_overlap_cutoff:
                    cutoff_args = (row, 1)
                    row += 1
                if show_hbond_allowance:
                    allowance_args = (row, 1)
                    row += 1
                distance_args = (row, 1)
            else:
                test_type_layout = group_layout
                cutoff_args = allowance_args = distance_args = ()
            if show_overlap_cutoff:
                overlap_layout = QHBoxLayout()
                overlap_layout.setContentsMargins(0,0,0,0)
                #overlap_layout.setSpacing(0)
                test_type_layout.addLayout(overlap_layout, *cutoff_args)
                lab = QLabel("Find atoms with VDW overlap \N{GREATER-THAN OR EQUAL TO}")
                overlap_widgets.append(lab)
                overlap_layout.addWidget(lab)
                self.overlap_spinbox = QDoubleSpinBox()
                self.overlap_spinbox.setDecimals(2)
                self.overlap_spinbox.setSingleStep(0.1)
                self.overlap_spinbox.setValue(final_val['overlap_cutoff'])
                self.overlap_spinbox.setSuffix('\N{ANGSTROM SIGN}')
                overlap_widgets.append(self.overlap_spinbox)
                overlap_layout.addWidget(self.overlap_spinbox, stretch=1, alignment=Qt.AlignLeft)
            if show_hbond_allowance:
                hbond_layout = QHBoxLayout()
                hbond_layout.setContentsMargins(0,0,0,0)
                hbond_layout.setSpacing(0)
                test_type_layout.addLayout(hbond_layout, *allowance_args)
                lab = QLabel("Subtract")
                overlap_widgets.append(lab)
                hbond_layout.addWidget(lab)
                self.hbond_spinbox = QDoubleSpinBox()
                self.hbond_spinbox.setDecimals(2)
                self.hbond_spinbox.setSingleStep(0.1)
                self.hbond_spinbox.setValue(final_val['hbond_allowance'])
                self.hbond_spinbox.setSuffix('\N{ANGSTROM SIGN}')
                overlap_widgets.append(self.hbond_spinbox)
                hbond_layout.addWidget(self.hbond_spinbox)
                lab = QLabel("from overlap for potentially H-bonding pairs")
                overlap_widgets.append(lab)
                hbond_layout.addWidget(lab, stretch=1, alignment=Qt.AlignLeft)
            if show_distance_only:
                distance_layout = QHBoxLayout()
                distance_layout.setContentsMargins(0,0,0,0)
                distance_layout.setSpacing(0)
                test_type_layout.addLayout(distance_layout, *distance_args)
                lab = QLabel("Find atoms whose center-center distance is \N{LESS-THAN OR EQUAL TO}")
                distance_only_widgets.append(lab)
                distance_layout.addWidget(lab)
                self.dist_only_spinbox = QDoubleSpinBox()
                self.dist_only_spinbox.setDecimals(2)
                self.dist_only_spinbox.setSingleStep(0.1)
                val = 1.4 if final_val['distance_only'] is None else final_val['distance_only']
                self.dist_only_spinbox.setValue(val)
                self.dist_only_spinbox.setSuffix('\N{ANGSTROM SIGN}')
                distance_only_widgets.append(self.dist_only_spinbox)
                distance_layout.addWidget(self.dist_only_spinbox, stretch=1, alignment=Qt.AlignLeft)
            if overlap_widgets and distance_only_widgets:
                if distance_only is None:
                    self.overlap_radio.setChecked(True)
                    enable_widgets(False, distance_only_widgets, overlap_widgets)
                else:
                    self.distance_radio.setChecked(True)
                    enable_widgets(True, distance_only_widgets, overlap_widgets)
            if show_bond_separation:
                bond_sep_layout = QHBoxLayout()
                bond_sep_layout.setContentsMargins(0,0,0,0)
                bond_sep_layout.setSpacing(0)
                group_layout.addLayout(bond_sep_layout)
                bond_sep_layout.addWidget(QLabel("Ignore %s of pairs" % prox_words))
                self.bond_sep_button = QPushButton()
                bond_sep_layout.addWidget(self.bond_sep_button)
                bond_sep_menu = QMenu()
                for bond_sep in range(2, 6):
                    bond_sep_menu.addAction(str(bond_sep))
                bond_sep_menu.triggered.connect(lambda action, but=self.bond_sep_button:
                    but.setText(action.text()))
                self.bond_sep_button.setMenu(bond_sep_menu)
                self.bond_sep_button.setText(str(bond_separation))
                bond_sep_layout.addWidget(QLabel("or fewer bonds apart"), stretch=1,
                    alignment=Qt.AlignLeft)
            if show_res_separation:
                res_sep_layout = QHBoxLayout()
                res_sep_layout.setContentsMargins(0,0,0,0)
                #res_sep_layout.setSpacing(0)
                group_layout.addLayout(res_sep_layout)
                self.res_sep_checkbox = QCheckBox("Only find %s between residues at least" % prox_words)
                final_rs_val = final_val['res_separation']
                self.res_sep_checkbox.setChecked(final_rs_val is not None)
                res_sep_layout.addWidget(self.res_sep_checkbox)
                val = 5 if final_rs_val is None else final_rs_val
                self.res_sep_spinbox = rs_box = QSpinBox()
                rs_box.setMinimum(1)
                rs_box.setValue(val)
                res_sep_layout.addWidget(rs_box)
                res_sep_layout.addWidget(QLabel("apart in sequence"), stretch=1, alignment=Qt.AlignLeft)

            if show_boolean_params:
                self.bool_param_options = bool_param_options = OptionsPanel(sorting=False, scrolled=False,
                    contents_margins=(10,0,10,0))
                group_layout.addWidget(bool_param_options, alignment=Qt.AlignCenter)
                if show_inter_model:
                    self.inter_model_option = BooleanOption("Include inter-model %s" % prox_words,
                        None if settings else inter_model, None, attr_name="inter_model", settings=settings)
                    bool_param_options.add_option(self.inter_model_option)
                if show_inter_submodel:
                    self.inter_submodel_option = BooleanOption("Include inter-submodel %s" % prox_words,
                        None if settings else inter_submodel, None,
                        attr_name="inter_submodel", settings=settings)
                    bool_param_options.add_option(self.inter_submodel_option)
                if show_intra_res:
                    self.intra_res_option = BooleanOption("Include intra-residue %s" % prox_words,
                        None if settings else intra_res, None, attr_name="intra_res", settings=settings)
                    bool_param_options.add_option(self.intra_res_option)
                if show_intra_mol:
                    self.intra_mol_option = BooleanOption("Include intra-molecule %s" % prox_words,
                        None if settings else intra_mol, None, attr_name="intra_mol", settings=settings)
                    bool_param_options.add_option(self.intra_mol_option)

        if show_select or show_color_atoms or show_atom_color or show_other_atom_color \
        or show_make_pseudobonds or show_color or show_dashes or show_radius or show_name or show_reveal \
        or show_attr_name or show_set_attrs or show_log or show_save_file:
            group = QGroupBox("Treatment of %s atoms" % prox_word)
            layout.addWidget(group)
            group_layout = QVBoxLayout()
            group_layout.setContentsMargins(0,0,0,0)
            group_layout.setSpacing(5)
            group.setLayout(group_layout)
            treatment_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(10,0,10,0))
            group_layout.addWidget(treatment_options)
            if show_select:
                self.select_option = BooleanOption("Select",
                    None if settings else select, None, attr_name="select", settings=settings)
                treatment_options.add_option(self.select_option)
            if show_color_atoms:
                if show_atom_color or show_other_atom_color:
                    # checkable group
                    self.color_atoms_widget, sub_options = treatment_options.add_option_group(
                        group_label="Color atoms", checked=final_val['color_atoms'],
                        contents_margins=(10,0,10,0), sorting=False)
                    subgroup_layout = QVBoxLayout()
                    subgroup_layout.setContentsMargins(0,0,0,0)
                    subgroup_layout.setSpacing(5)
                    self.color_atoms_widget.setLayout(subgroup_layout)
                    subgroup_layout.addWidget(sub_options)
                    if show_atom_color:
                        self.atom_color_option = OptionalRGBAOption("Color",
                            None if settings else atom_color,
                            None, attr_name="atom_color", settings=settings)
                        sub_options.add_option(self.atom_color_option)
                    if show_other_atom_color:
                        self.other_atom_color_option = OptionalRGBAOption("Other atoms color",
                            None if settings else other_atom_color,
                            None, attr_name="other_atom_color", settings=settings)
                        sub_options.add_option(self.other_atom_color_option)
                else:
                    # boolean
                    self.color_atoms_widget = BooleanOption("Color atoms",
                        None if settings else color_atoms, None, attr_name="color_atoms", settings=settings)
                    treatment_options.add_option(self.color_atoms_widget)
            else:
                # booleans
                if show_atom_color:
                    self.atom_color_option = OptionalRGBAOption("Color", None if settings else atom_color,
                        None, attr_name="atom_color", settings=settings)
                    treatment_options.add_option(self.atom_color_option)
                if show_other_atom_color:
                    self.other_atom_color_option = OptionalRGBAOption("Other atoms color",
                        None if settings else other_atom_color,
                        None, attr_name="other_atom_color", settings=settings)
                    treatment_options.add_option(self.other_atom_color_option)
            if show_make_pseudobonds:
                if show_color or show_dashes or show_radius or show_name:
                    # checkable group
                    self.make_pseudobonds_widget, sub_options = treatment_options.add_option_group(
                        group_label="Draw pseudobonds", checked=final_val['make_pseudobonds'],
                        contents_margins=(10,0,10,0), sorting=False)
                    subgroup_layout = QVBoxLayout()
                    subgroup_layout.setContentsMargins(0,0,0,0)
                    subgroup_layout.setSpacing(5)
                    self.make_pseudobonds_widget.setLayout(subgroup_layout)
                    subgroup_layout.addWidget(sub_options)
                    if show_color:
                        self.color_option = ColorOption("Color", None if settings else color,
                            None, attr_name="color", settings=settings)
                        sub_options.add_option(self.color_option)
                    if show_dashes:
                        self.dashes_option = IntOption("Dashes", None if settings else dashes,
                            None, attr_name="dashes", min=0, settings=settings)
                        sub_options.add_option(self.dashes_option)
                    if show_radius:
                        self.radius_option = FloatOption("Radius", None if settings else radius,
                            None, attr_name="radius", settings=settings)
                        sub_options.add_option(self.radius_option)
                    if show_name:
                        self.name_option = StringOption("Group name", name, None)
                        sub_options.add_option(self.name_option)
                else:
                    # boolean
                    self.make_pseudobonds_widget = BooleanOption("Draw pseudobonds",
                        None if settings else make_pseudobonds, None, attr_name="make_pseudobonds",
                        settings=settings)
                    treatment_options.add_option(self.make_pseudobonds_widget)
            else:
                # booleans
                if show_color:
                    self.color_option = ColorOption("Color", None if settings else color,
                        None, attr_name="color", settings=settings)
                    treatment_options.add_option(self.color_option)
                if show_radius:
                    self.radius_option = FloatOption("Radius",
                        None if settings else radius,
                        None, attr_name="radius", settings=settings)
                    treatment_options.add_option(self.radius_option)
            if show_reveal:
                self.reveal_option = BooleanOption("If endpoint atom hidden, show endpoint residue",
                    None if settings else reveal, None, attr_name="reveal", settings=settings)
                treatment_options.add_option(self.reveal_option)
            if show_set_attrs:
                if show_attr_name:
                    combo_option = make_optional(StringOption)
                    self.set_attrs_option = self.attr_name_option = combo_option("Assign attribute named",
                        final_val['attr_name'], None)
                    if not final_val['set_attrs']:
                        # done this way so that attr name is not blank
                        self.set_attrs_option.value = None
                else:
                    self.set_attrs_option = BooleanOption("Assign attribute", final_val['set_attrs'], None)
                treatment_options.add_option(self.set_attrs_option)
            elif show_attr_name:
                self.attr_name_option = StringOption("Attribute name", final_val['attr_name'], None)
                treatment_options.add_option(self.attr_name_option)
            if show_log or show_save_file:
                group = QGroupBox("Write information to:")
                layout.addWidget(group)
                info_layout = QVBoxLayout()
                info_layout.setContentsMargins(0,0,0,0)
                info_layout.setSpacing(0)
                group.setLayout(info_layout)
                info_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(0,0,0,0))
                info_layout.addWidget(info_options)

                if show_log:
                    self.log_option = BooleanOption("Log", None if settings else log, None, attr_name="log",
                        settings=settings)
                    info_options.add_option(self.log_option)

                if show_save_file:
                    self.save_file_option = BooleanOption("File", False, None)
                    info_options.add_option(self.save_file_option)
            if show_summary:
                self.summary_option = BooleanOption("Log total number of %s" % prox_words,
                    None if settings else summary, None, attr_name="summary", settings=settings)
                treatment_options.add_option(self.summary_option)

        if show_checking_frequency:
            group = QGroupBox("Frequency of checking")
            layout.addWidget(group)
            group_layout = QGridLayout()
            group_layout.setContentsMargins(0,0,0,0)
            group_layout.setSpacing(5)
            group_layout.setColumnStretch(0, 1)
            group_layout.setColumnStretch(3, 1)
            group.setLayout(group_layout)
            group_layout.addWidget(QLabel("Check..."), 0, 1, 2, 1, alignment=Qt.AlignRight|Qt.AlignVCenter)
            self.ok_radio = QRadioButton("when OK%s clicked" % ("/Apply" if has_apply_button else ""))
            self.ok_radio.setChecked(True)
            group_layout.addWidget(self.ok_radio, 0, 2, alignment=Qt.AlignLeft)
            self.ok_radio.toggled.connect(self._checking_change)
            group_layout.addWidget(QRadioButton("continuously (until dialog closed)"), 1, 2,
                alignment=Qt.AlignLeft)

    def destroy(self):
        for handler in self.handlers.values():
            handler.remove()
        self.handlers.clear()
        #TODO: if continuous checking, issue '~' command
        super().destroy()

    def get_command(self):
        """Used to generate the 'hbonds' command that can be run to produce the requested H-bonds.
           Returns three strings:
              1) The command name
              2) The atom spec to provide just after the command name
              3) The keyword/value pairs that follow that
           The atom spec will be an empty string if no bond or model restriction was requested
           (or those controls weren't shown).
        """
        from chimerax.core.errors import UserError
        settings = {}
        command_values = {}

        # never saved in settings
        if self.__show_values['model_restrict']:
            models = self.__model_restrict_option.value
            if models is None:
                atom_spec = ""
            else:
                if not models:
                    raise UserError("Model restriction enabled but no models chosen")
                from chimerax.core.commands import concise_model_spec
                atom_spec = concise_model_spec(self.session, models)
        else:
            atom_spec = ""

        if self.__show_values['bond_restrict']:
            bond_restrict = self.__bond_restrict_option.value
            if bond_restrict is not None:
                command_values['restrict'] = bond_restrict
                if atom_spec:
                    atom_spec += " & sel"
                else:
                    atom_spec = "sel"

        if self.__show_values['save_file']:
            if self.__save_file_option.value:
                from PyQt5.QtWidgets import QFileDialog
                fname = QFileDialog.getSaveFileName(self, "Save H-Bonds File")[0]
                if fname:
                    command_values['save_file'] = fname
                else:
                    from chimerax.core.errors import CancelOperation
                    raise CancelOperation("H-bonds save file cancelled")
            else:
                command_values['save_file'] = None
        else:
            command_values['save_file'] = None

        # may be saved in settings
        if self.__show_values['color']:
            settings['color'] = self.__color_option.value
        else:
            settings['color'] = None

        if self.__show_values['radius']:
            settings['radius'] = self.__radius_option.value
        else:
            settings['radius'] = None

        if self.__show_values['dashes']:
            settings['dashes'] = self.__dashes_option.value
        else:
            settings['dashes'] = None

        if self.__show_values['show_dist']:
            settings['show_dist'] = self.__show_dist_option.value
        else:
            settings['show_dist'] = None

        if self.__show_values['inter_intra_model']:
            settings['inter_model'] = not self.__intra_model_only_option.value
            settings['intra_model'] = not self.__inter_model_only_option.value
        else:
            settings['inter_model'] = settings['intra_model'] = None

        if self.__show_values['relax']:
            settings['relax'] = self.__relax_group.isChecked()
            if self.__show_values['slop']:
                settings['dist_slop'] = self.__dist_slop_option.value
                settings['angle_slop'] = self.__angle_slop_option.value
            else:
                settings['dist_slop'] = settings['angle_slop'] = None
            if self.__show_values['slop_color']:
                slop_color_value = self.__slop_color_option.value
                if slop_color_value is None:
                    settings['two_colors'] = False
                    settings['slop_color'] = None
                else:
                    settings['two_colors'] = True
                    settings['slop_color'] = slop_color_value
            else:
                settings['two_colors'] = settings['slop_color'] = None
        else:
            settings['relax'] = settings['dist_slop'] = settings['angle_slop'] = None
            settings['two_colors'] = settings['slop_color'] = None

        if self.__show_values['salt_only']:
            settings['salt_only'] = self.__salt_only_option.value
        else:
            settings['salt_only'] = None

        if self.__show_values['intra_mol']:
            settings['intra_mol'] = self.__intra_mol_option.value
        else:
            settings['intra_mol'] = None

        if self.__show_values['intra_res']:
            settings['intra_res'] = self.__intra_res_option.value
        else:
            settings['intra_res'] = None

        if self.__show_values['inter_submodel']:
            settings['inter_submodel'] = self.__inter_submodel_option.value
        else:
            settings['inter_submodel'] = None

        if self.__show_values['reveal']:
            settings['reveal'] = self.__reveal_option.value
        else:
            settings['reveal'] = None

        if self.__show_values['retain_current']:
            settings['retain_current'] = self.__retain_current_option.value
        else:
            settings['retain_current'] = None

        if self.__show_values['log']:
            settings['log'] = self.__log_option.value
        else:
            settings['log'] = None

        if self.__settings:
            saveables = []
            for attr_name, value in settings.items():
                if value is not None:
                    setattr(self.__settings, attr_name, value)
                    saveables.append(attr_name)
            if saveables:
                self.__settings.save(settings=saveables)

        def val_to_str(ses, val, kw):
            from chimerax.core.commands import \
                BoolArg, IntArg, FloatArg, ColorArg, StringArg, SaveFileNameArg
            if type(val) == bool:
                return BoolArg.unparse(val, ses)
            if type(val) == int:
                return IntArg.unparse(val, ses)
            if type(val) == float:
                return FloatArg.unparse(val, ses)
            if kw.endswith('color'):
                from chimerax.core.colors import Color
                return ColorArg.unparse(Color(rgba=val), ses)
            if kw == 'save_file':
                return SaveFileNameArg.unparse(val, ses)
            return StringArg.unparse(str(val), ses)

        command_values.update(settings)
        from .cmd import cmd_hbonds
        kw_values = ""
        for kw, val in command_values.items():
            if val is None:
                continue
            if is_default(cmd_hbonds, kw, val):
                continue
            # 'dashes' default checking requires special handling
            if kw == 'dashes' and val == AtomicStructure.default_hbond_dashes \
            and not command_values['retain_current']:
                continue
            camel = ""
            next_upper = False
            for c in kw:
                if c == '_':
                    next_upper = True
                else:
                    if next_upper:
                        camel += c.upper()
                    else:
                        camel += c
                    next_upper = False
            kw_values += (" " if kw_values else "") + camel + " " + val_to_str(self.session, val, kw)
        return "hbonds", atom_spec, kw_values

    def _checking_change(self, ok_now_checked):
        #TODO: if continuous checking, issue '~' command
        pass

    def _designate1_cb(self):
        from chimerax.atomic import selected_atoms, get_triggers
        if 'desig1' not in self.handlers:
            self.handlers['desig1'] = get_triggers().add_handler('changes', lambda trig_name, changes:
                changes.num_destroyed_atoms() > 0 and self._update_desig1_status())
        self.desig1_atoms = selected_atoms(self.session)
        self._update_desig1_status()

    def _designate2_cb(self):
        from chimerax.atomic import selected_atoms, get_triggers
        if 'desig2' not in self.handlers:
            self.handlers['desig2'] = get_triggers().add_handler('changes', lambda trig_name, changes:
                changes.num_destroyed_atoms() > 0 and self._update_desig2_status())
        self.desig2_atoms = selected_atoms(self.session)
        self._update_desig2_status()

    def _inter_model_cb(self, opt):
        if opt.value:
            self.__intra_model_only_option.value = False

    def _intra_model_cb(self, opt):
        if opt.value:
            self.__inter_model_only_option.value = False

    def _test_change(self, but):
        if but.text() == self.test_kw_to_label[None]:
            show = True
            color = "black" if self.desig1_atoms else "red"
            desig2_but = but
        else:
            show = False
            color = "black"
            desig2_but = self.test_button_group.button(2)
        self.desig2_button.setHidden(not show)
        self.desig2_label.setHidden(not show)
        self.desig2_status.setHidden(not show)
        desig2_but.setStyleSheet("color: %s" % color)

    def _update_desig1_status(self):
        if self.desig1_atoms:
            color = "black"
            msg = "%d atoms designated" % len(self.desig1_atoms)
        else:
            color = "red"
            msg = "No atoms designated"
            if 'desig1' in self.handlers:
                self.handlers['desig1'].remove()
                del self.handlers['desig1']
        self.desig1_status.setText(msg)
        self.desig1_status.setStyleSheet("color: %s" % color)

    def _update_desig2_status(self):
        if self.desig2_atoms:
            color = "black"
            msg = "%d atoms designated" % len(self.desig2_atoms)
        else:
            color = "red"
            msg = "No second set"
            if 'desig2' in self.handlers:
                self.handlers['desig2'].remove()
                del self.handlers['desig2']
        tbg = self.test_button_group
        if tbg.checkedButton() == tbg.button(2):
            tbg.checkedButton().setStyleSheet("color: %s" % color)
        self.desig2_status.setText(msg)
        self.desig2_status.setStyleSheet("color: %s" % color)

def is_default(func, kw, val):
    from inspect import signature
    sig = signature(func)
    param = sig.parameters[kw]
    if kw.endswith('color'):
        from chimerax.core.colors import Color
        if isinstance(val, Color):
            cval = val
        else:
            cval = Color(val)
        if isinstance(param.default, Color):
            pval = param.default
        else:
            pval = Color(param.default)
        return cval == pval
    return param.default == val

class ClashesGUI(AtomProximityGUI):
    def __init__(self, session, has_apply_button, *, name="clashes",
            hbond_allowance=defaults["clash_hbond_allowance"], overlap_cutoff=defaults["clash_threshold"],
            **kw):
        from .cmd import handle_clash_kw
        color, radius = handle_clash_kw(kw)
        if 'show_distance_only' not in kw:
            kw['show_distance_only'] = False
        super().__init__(session, name, "clash", "clashes", color, radius,
            hbond_allowance, overlap_cutoff, has_apply_button, **kw)

class ContactsGUI(AtomProximityGUI):
    def __init__(self, session, has_apply_button, *, name="contacts",
            hbond_allowance=defaults["clash_hbond_allowance"], overlap_cutoff=defaults["contact_threshold"],
            **kw):
        from .cmd import handle_contact_kw
        color, radius = handle_contact_kw(kw)
        if 'show_hbond_allowance' not in kw:
            kw['show_hbond_allowance'] = False
        super().__init__(session, name, "contact", "contacts", color, radius,
            hbond_allowance, overlap_cutoff, has_apply_button, **kw)

def _get_settings(session, base_name, settings_defaults, name_mod):
    if base_name:
        settings_name = base_name + " " + name_mod
    else:
        settings_name = name_mod
    from chimerax.core.settings import Settings
    class HBondGUISettings(Settings):
        EXPLICIT_SAVE = settings_defaults

    return HBondGUISettings(session, settings_name)

#TODO: settings that need explicit save: inter_model, intra_model, two_colors, slop_color
