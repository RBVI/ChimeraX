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
from Qt.QtWidgets import QWidget
from chimerax.ui.options import OptionsPanel, ColorOption, FloatOption, BooleanOption, IntOption, \
    StringOption, OptionalRGBAOption, make_optional
from chimerax.atomic.options import AtomPairRestrictOption

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
            attr_name=defaults["attr_name"], bond_separation=defaults["bond_separation"],
            continuous=False, dashes=None, distance_only=None, ignore_hidden_models=None, inter_model=True,
            inter_submodel=False, intra_model=True, intra_mol=defaults["intra_mol"],
            intra_res=defaults["intra_res"], log=defaults["action_log"],
            make_pseudobonds=defaults["action_pseudobonds"], res_separation=None, restrict=None,
            reveal=False, save_file=None, select=defaults["action_select"],
            set_attrs=defaults["action_attr"], show_dist=False, summary=True, test_atoms=None,

            # what controls to show in the interface
            #
            # note that if 'test_atoms' is not None, then the selection-restriction control will be omitted
            # regardless of the 'show_restrict' value
            #
            # Also, if 'ignore_hidden_models' is None and it's settings value is also None, then when
            # the command arguments are generated 'ignore_hidden_models' is treated as True if
            # 'show_ignore_hidden_models' is True, else False
            show_attr_name=True, show_bond_separation=True, show_checking_frequency=True, show_color=True,
            show_dashes=True, show_distance_only=True, show_hbond_allowance=True,
            show_ignore_hidden_models=False, show_inter_model=True, show_inter_submodel=False,
            show_intra_model=True, show_intra_mol=True, show_intra_res=True, show_log=True,
            show_make_pseudobonds=True, show_name=True, show_overlap_cutoff=True, show_radius=True,
            show_res_separation=True, show_restrict=True, show_reveal=True, show_save_file=True,
            show_select=True, show_set_attrs=True, show_show_dist=True, show_summary=False):

        self.session = session
        self.cmd_name = cmd_name
        self.default_group_name = name

        from inspect import getargvalues, currentframe
        arg_names, var_args, var_kw, frame_dict = getargvalues(currentframe())
        settings_defaults = {}
        self.show_values = {}
        from chimerax.core.colors import ColorValue
        for arg_name in arg_names:
            if not arg_name.startswith('show_') or 'show_' + arg_name in arg_names:
                if arg_name.endswith('color'):
                    value = ColorValue(frame_dict[arg_name])
                else:
                    value = frame_dict[arg_name]
                settings_defaults[arg_name] = value
            else:
                self.show_values[arg_name[5:]] = frame_dict[arg_name]
        if settings_name is None:
            self.settings = settings = None
        else:
            self.settings = settings = _get_settings(session, settings_name, settings_defaults, cmd_name)
        self.final_vals = final_val = {}
        for def_name in settings_defaults.keys():
            final_val[def_name] = getattr(settings, def_name) if settings else frame_dict[def_name]

        super().__init__()
        from Qt.QtWidgets import QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QButtonGroup
        from Qt.QtWidgets import QRadioButton, QAbstractButton, QHBoxLayout, QDoubleSpinBox, QMenu
        from Qt.QtWidgets import QSpinBox, QCheckBox
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.setLayout(layout)

        from chimerax.core.commands import plural_of
        self.prox_word = prox_word
        self.prox_words = prox_words = plural_of(prox_word)
        if test_atoms is not None:
            show_restrict = False
        show_bool_params = show_intra_model or show_intra_mol or show_intra_res or show_inter_model \
            or show_inter_submodel or show_ignore_hidden_models
        if show_overlap_cutoff or show_hbond_allowance or show_restrict or show_bond_separation \
        or show_bool_params:
            group = QGroupBox("Interaction parameters")
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
                overlap_layout.setSpacing(0)
                test_type_layout.addLayout(overlap_layout, *cutoff_args)
                lab = QLabel("Find pairs of atoms with VDW overlap \N{GREATER-THAN OR EQUAL TO}")
                overlap_widgets.append(lab)
                overlap_layout.addWidget(lab)
                self.overlap_spinbox = QDoubleSpinBox()
                self.overlap_spinbox.setDecimals(2)
                self.overlap_spinbox.setSingleStep(0.1)
                self.overlap_spinbox.setRange(-99, 99)
                self.overlap_spinbox.setValue(final_val['overlap_cutoff'])
                #self.overlap_spinbox.setSuffix('\N{ANGSTROM SIGN}')
                overlap_widgets.append(self.overlap_spinbox)
                overlap_layout.addWidget(self.overlap_spinbox)
                lab2 = QLabel("\N{ANGSTROM SIGN}")
                overlap_widgets.append(lab2)
                overlap_layout.addWidget(lab2, stretch=1, alignment=Qt.AlignLeft)
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
                self.hbond_spinbox.setRange(-99, 99)
                self.hbond_spinbox.setValue(final_val['hbond_allowance'])
                #self.hbond_spinbox.setSuffix('\N{ANGSTROM SIGN}')
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
                lab = QLabel("Find pairs of atoms with center-center distance \N{LESS-THAN OR EQUAL TO}")
                distance_only_widgets.append(lab)
                distance_layout.addWidget(lab)
                self.dist_only_spinbox = QDoubleSpinBox()
                self.dist_only_spinbox.setDecimals(2)
                self.dist_only_spinbox.setSingleStep(0.1)
                self.dist_only_spinbox.setRange(0, 1000000)
                val = 4.0 if final_val['distance_only'] is None else final_val['distance_only']
                self.dist_only_spinbox.setValue(val)
                #self.dist_only_spinbox.setSuffix('\N{ANGSTROM SIGN}')
                distance_only_widgets.append(self.dist_only_spinbox)
                distance_layout.addWidget(self.dist_only_spinbox, alignment=Qt.AlignLeft)
                lab2 = QLabel("\N{ANGSTROM SIGN}")
                distance_only_widgets.append(lab2)
                distance_layout.addWidget(lab2, stretch=1, alignment=Qt.AlignLeft)
            if overlap_widgets and distance_only_widgets:
                if distance_only is None:
                    self.overlap_radio.setChecked(True)
                    enable_widgets(False, distance_only_widgets, overlap_widgets)
                else:
                    self.distance_radio.setChecked(True)
                    enable_widgets(True, distance_only_widgets, overlap_widgets)
            if show_restrict:
                restrict_options = OptionsPanel(sorting=False, scrolled=False,
                    contents_margins=(0,0,0,0))
                group_layout.addWidget(restrict_options, alignment=Qt.AlignLeft)
                if show_restrict:
                    self.sel_restrict_option = make_optional(AtomPairRestrictOption)("Limit by selection",
                        None if settings else restrict, None, attr_name="restrict", settings=settings,
                        atom_word="end")
                    restrict_options.add_option(self.sel_restrict_option)
            if show_bond_separation:
                bond_sep_layout = QHBoxLayout()
                bond_sep_layout.setContentsMargins(0,0,0,0)
                bond_sep_layout.setSpacing(0)
                group_layout.addLayout(bond_sep_layout)
                bond_sep_layout.addWidget(QLabel("Ignore interactions between atoms"))
                self.bond_sep_button = QPushButton()
                bond_sep_layout.addWidget(self.bond_sep_button)
                bond_sep_menu = QMenu(self)
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
                self.res_sep_checkbox = QCheckBox("Ignore interactions between residues <")
                final_rs_val = final_val['res_separation']
                self.res_sep_checkbox.setChecked(final_rs_val is not None)
                res_sep_layout.addWidget(self.res_sep_checkbox)
                val = 5 if final_rs_val is None else final_rs_val
                self.res_sep_spinbox = rs_box = QSpinBox()
                rs_box.setMinimum(1)
                rs_box.setValue(val)
                res_sep_layout.addWidget(rs_box)
                res_sep_layout.addWidget(QLabel("apart in sequence"), stretch=1, alignment=Qt.AlignLeft)

            if show_bool_params:
                bool_param_options = OptionsPanel(sorting=False, scrolled=False,
                    contents_margins=(10,0,10,0), columns=2)
                group_layout.addWidget(bool_param_options, alignment=Qt.AlignCenter)
                if show_inter_model:
                    self.inter_model_option = BooleanOption("Include intermodel",
                        None if settings else inter_model, None, attr_name="inter_model", settings=settings)
                    bool_param_options.add_option(self.inter_model_option)
                if show_inter_submodel:
                    self.inter_submodel_option = BooleanOption("Include intersubmodel",
                        None if settings else inter_submodel, None,
                        attr_name="inter_submodel", settings=settings)
                    bool_param_options.add_option(self.inter_submodel_option)
                if show_intra_model:
                    self.intra_model_option = BooleanOption("Include intramodel",
                        None if settings else intra_model, None, attr_name="intra_model", settings=settings)
                    bool_param_options.add_option(self.intra_model_option)
                if show_intra_mol:
                    self.intra_mol_option = BooleanOption("Include intramolecule",
                        None if settings else intra_mol, None, attr_name="intra_mol", settings=settings)
                    bool_param_options.add_option(self.intra_mol_option)
                if show_intra_res:
                    self.intra_res_option = BooleanOption("Include intraresidue",
                        None if settings else intra_res, None, attr_name="intra_res", settings=settings)
                    bool_param_options.add_option(self.intra_res_option)
                if show_ignore_hidden_models:
                    self.ignore_hidden_option = BooleanOption("Ignore hidden models", None if settings else
                        (True if ignore_hidden_models is None else ignore_hidden_models), None,
                        attr_name="ignore_hidden_models", settings=settings)
                    bool_param_options.add_option(self.ignore_hidden_option)

        if show_select or show_make_pseudobonds or show_color or show_dashes or show_radius \
        or show_show_dist or show_name or show_reveal or show_attr_name or show_set_attrs or show_log \
        or show_save_file:
            group = QGroupBox("Treatment of results")
            layout.addWidget(group)
            group_layout = QVBoxLayout()
            group_layout.setContentsMargins(0,0,0,0)
            group_layout.setSpacing(5)
            group.setLayout(group_layout)
            treatment_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(10,0,10,0))
            group_layout.addWidget(treatment_options)
            if show_select:
                self.select_option = BooleanOption("Select atoms",
                    None if settings else select, None, attr_name="select", settings=settings)
                treatment_options.add_option(self.select_option)
            if show_make_pseudobonds:
                if show_color or show_dashes or show_radius or show_show_dist or show_name:
                    # checkable group
                    self.make_pseudobonds_widget, sub_options = treatment_options.add_option_group(
                        group_label="Display as pseudobonds", checked=final_val['make_pseudobonds'],
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
                    if show_radius:
                        self.radius_option = FloatOption("Radius", None if settings else radius,
                            None, attr_name="radius", settings=settings)
                        sub_options.add_option(self.radius_option)
                    if show_dashes:
                        self.dashes_option = IntOption("Dashes", None if settings else dashes,
                            None, attr_name="dashes", min=0, settings=settings)
                        sub_options.add_option(self.dashes_option)
                    if show_show_dist:
                        self.show_dist_option = BooleanOption("Distance label",
                            None if settings else show_dist, None, attr_name="show_dist", settings=settings)
                        sub_options.add_option(self.show_dist_option)
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
                if show_color:
                    self.color_option = ColorOption("Pseudobond color", None if settings else color,
                        None, attr_name="color", settings=settings)
                    treatment_options.add_option(self.color_option)
                if show_radius:
                    self.radius_option = FloatOption("Pseudobond radius",
                        None if settings else radius,
                        None, attr_name="radius", settings=settings)
                    treatment_options.add_option(self.radius_option)
                if show_dashes:
                    self.dashes_option = IntOption("Pseudobond dashes", None if settings else dashes,
                        None, attr_name="dashes", min=0, settings=settings)
                    sub_options.add_option(self.dashes_option)
                if show_show_dist:
                    self.show_dist_option = BooleanOption("Pseudobond distance label",
                        None if settings else show_dist, None, attr_name="show_dist", settings=settings)
                    sub_options.add_option(self.show_dist_option)
                if show_name:
                    self.name_option = StringOption("Pseudobond group name", name, None)
                    sub_options.add_option(self.name_option)
            if show_reveal:
                self.reveal_option = BooleanOption("Reveal atoms of interacting residues",
                    None if settings else reveal, None, attr_name="reveal", settings=settings)
                treatment_options.add_option(self.reveal_option)
            if show_set_attrs:
                if show_attr_name:
                    combo_option = make_optional(StringOption)
                    option = self.attr_name_option = combo_option("Assign atomic attribute named",
                        final_val['attr_name'], None)
                    if not final_val['set_attrs']:
                        # done this way so that attr name is not blank
                        option.value = None
                else:
                    option = self.set_attrs_option = BooleanOption("Assign attribute",
                        final_val['set_attrs'], None)
                treatment_options.add_option(option)
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
                info_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(0,0,0,0),
                    columns=2 if show_log and show_save_file else 1)
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
        if self.show_values['checking_frequency'] and not self.ok_radio.isChecked():
            # continuous monitoring is on, turn it off
            from chimerax.core.commands import run
            run(self.session, "~" + self.cmd_name)
        super().destroy()

    def get_command(self):
        """Used to generate the command that can be run to produce the requested action.
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
        atom_spec = ""
        if self.show_values['save_file']:
            if self.save_file_option.value:
                from Qt.QtWidgets import QFileDialog
                fname = QFileDialog.getSaveFileName(self, "Save %s File" % self.prox_words.capitalize())[0]
                if fname:
                    command_values['save_file'] = fname
                else:
                    from chimerax.core.errors import CancelOperation
                    raise CancelOperation("%s save file cancelled" % self.prox_words.capitalize())
            else:
                command_values['save_file'] = None
        else:
            command_values['save_file'] = None

        if self.show_values['set_attrs']:
            if self.show_values['attr_name']:
                combo_val = self.attr_name_option.value
                command_values['set_attrs'] = combo_val is not None
                command_values['attr_name'] = combo_val
            else:
                command_values['set_attrs'] = self.set_attrs_option.value
        else:
            command_values['set_attrs'] = None

        # may be saved in settings
        if self.show_values['restrict']:
            restrict = self.sel_restrict_option.value
            if restrict is not None:
                atom_spec = "sel"
            settings['restrict'] = restrict
            save_restrict = True
        else:
            save_restrict = False

        if (self.show_values['overlap_cutoff'] or self.show_values['hbond_allowance']) \
        and self.show_values['distance_only']:
            overlap_active = self.overlap_radio.isChecked()
            distance_active = not overlap_active
        else:
            overlap_active = distance_active = True

        if self.show_values['overlap_cutoff'] and overlap_active:
            settings['overlap_cutoff'] = self.overlap_spinbox.value()
        else:
            settings['overlap_cutoff'] = None

        if self.show_values['hbond_allowance'] and overlap_active:
            settings['hbond_allowance'] = self.hbond_spinbox.value()
        else:
            settings['hbond_allowance'] = None

        if self.show_values['distance_only'] and distance_active:
            settings['distance_only'] = self.dist_only_spinbox.value()
        else:
            settings['distance_only'] = None

        if self.show_values['bond_separation']:
            settings['bond_separation'] = int(self.bond_sep_button.text())
        else:
            settings['bond_separation'] = None

        if self.show_values['res_separation'] and self.res_sep_checkbox.isChecked():
            settings['res_separation'] = self.res_sep_spinbox.value()
        else:
            settings['res_separation'] = None

        if self.show_values['inter_model']:
            settings['inter_model'] = self.inter_model_option.value
        else:
            settings['inter_model'] = None

        if self.show_values['inter_submodel']:
            settings['inter_submodel'] = self.inter_submodel_option.value
        else:
            settings['inter_submodel'] = None

        if self.show_values['intra_res']:
            settings['intra_res'] = self.intra_res_option.value
        else:
            settings['intra_res'] = None

        if self.show_values['intra_model']:
            settings['intra_model'] = self.intra_model_option.value
        else:
            settings['intra_model'] = None

        if self.show_values['intra_mol']:
            settings['intra_mol'] = self.intra_mol_option.value
        else:
            settings['intra_mol'] = None

        if self.show_values['ignore_hidden_models']:
            settings['ignore_hidden_models'] = self.ignore_hidden_option.value
        else:
            # not the same default as the command, but don't save in settings
            final_val = self.final_vals['ignore_hidden_models']
            command_values['ignore_hidden_models'] = False if final_val is None else final_val

        if self.show_values['select']:
            settings['select'] = self.select_option.value
        else:
            settings['select'] = None

        if self.show_values['make_pseudobonds']:
            if isinstance(self.make_pseudobonds_widget, BooleanOption):
                settings['make_pseudobonds'] = self.make_pseudobonds_widget.value
            else:
                settings['make_pseudobonds'] = self.make_pseudobonds_widget.isChecked()
        else:
            settings['make_pseudobonds'] = None

        if self.show_values['color']:
            settings['color'] = self.color_option.value
        else:
            settings['color'] = None

        if self.show_values['dashes']:
            settings['dashes'] = self.dashes_option.value
        else:
            settings['dashes'] = None

        if self.show_values['radius']:
            settings['radius'] = self.radius_option.value
        else:
            settings['radius'] = None

        if self.show_values['show_dist']:
            settings['show_dist'] = self.show_dist_option.value
        else:
            settings['show_dist'] = None

        if self.show_values['name']:
            settings['name'] = self.name_option.value
        else:
            settings['name'] = None

        if self.show_values['reveal']:
            settings['reveal'] = self.reveal_option.value
        else:
            settings['reveal'] = None

        if self.show_values['log']:
            settings['log'] = self.log_option.value
        else:
            settings['log'] = None

        if self.settings:
            saveables = []
            for attr_name, value in settings.items():
                if value is not None or (attr_name == 'restrict' and save_restrict):
                    setattr(self.settings, attr_name, value)
                    saveables.append(attr_name)
            if saveables:
                self.settings.save(settings=saveables)

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
        from .cmd import cmd_clashes, cmd_contacts, _cmd, handle_clash_kw, handle_contact_kw
        if self.cmd_name == "clashes":
            specific_func = cmd_clashes
            special_kw = {}
            special_kw.update({ k:v for k,v in zip(('color', 'radius'), handle_clash_kw(special_kw)) })
        else:
            specific_func = cmd_contacts
            special_kw = {}
            special_kw.update({ k:v for k,v in zip(('color', 'radius'), handle_contact_kw(special_kw)) })
        kw_values = ""
        for kw, val in command_values.items():
            if val is None:
                continue
            if is_default(specific_func, kw, val, special_kw):
                continue
            if is_default(_cmd, kw, val, special_kw):
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
        return self.cmd_name, atom_spec, kw_values

    def reset(self):
        self.settings.reset()
        if (self.show_values['overlap_cutoff'] or self.show_values['hbond_allowance']) \
        and self.show_values['distance_only']:
            self.overlap_radio.setChecked(True)
        if self.show_values['overlap_cutoff']:
            self.overlap_spinbox.setValue(self.settings.overlap_cutoff)
        if self.show_values['hbond_allowance']:
            self.hbond_spinbox.setValue(self.settings.hbond_allowance)
        if self.show_values['distance_only']:
            val = 4.0 if self.settings.distance_only is None else self.settings.distance_only
            self.dist_only_spinbox.setValue(val)
        if self.show_values['restrict']:
            self.sel_restrict_option.value = self.sel_restrict_option.restrict_kw_vals[0]
            self.sel_restrict_option.value = None
        if self.show_values['bond_separation']:
            self.bond_sep_button.setText(str(self.settings.bond_separation))
        if self.show_values['res_separation']:
            self.res_sep_checkbox.setChecked(self.settings.res_separation is not None)
            val = 5 if self.settings.res_separation is None else self.settings.res_separation
            self.res_sep_spinbox.setValue(val)
        if self.show_values['set_attrs']:
            if self.show_values['attr_name']:
                self.attr_name_option.value = self.settings.attr_name
                self.attr_name_option.value = None
            else:
                self.set_attrs_option.value = self.settings.set_attrs
        if self.show_values['make_pseudobonds']:
            if isinstance(self.make_pseudobonds_widget, BooleanOption):
                self.make_pseudobonds_widget.value = self.settings.make_pseudobonds
            else:
                self.make_pseudobonds_widget.setChecked(self.settings.make_pseudobonds)
            if self.show_values['name']:
                self.name_option.value = self.default_group_name
        if self.show_values['save_file']:
            self.save_file_option.value = False
        if self.show_values['checking_frequency']:
            self.ok_radio.setChecked(True)

    def _checking_change(self, ok_now_checked):
        from chimerax.core.commands import run
        if ok_now_checked:
            # was continuous, issue '~' command
            run(self.session, "~" + self.cmd_name)
        else:
            # now continuous, issue command
            run(self.session, " ".join(self.get_command()) + " continuous true")

def is_default(func, kw, val, special_kw):
    from inspect import signature
    sig = signature(func)
    try:
        param = sig.parameters[kw]
    except KeyError:
        return False
    if kw in special_kw:
        pval = special_kw[kw]
    else:
        pval = param.default
    if pval == param.empty:
        if kw in special_kw:
            pval = special_kw[kw]
        else:
            return False
    if kw.endswith('color'):
        from chimerax.core.colors import Color
        if isinstance(val, Color):
            cval = val
        else:
            cval = Color(val)
        if not isinstance(pval, Color):
            pval = Color(param.default)
        return cval == pval
    return pval == val

class ClashesGUI(AtomProximityGUI):
    def __init__(self, session, has_apply_button, *, name="clashes",
            hbond_allowance=defaults["clash_hbond_allowance"], overlap_cutoff=defaults["clash_threshold"],
            **kw):
        from .cmd import handle_clash_kw
        color, radius = handle_clash_kw(kw)
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
    class AtomProximityGUISettings(Settings):
        EXPLICIT_SAVE = settings_defaults

    return AtomProximityGUISettings(session, settings_name)

