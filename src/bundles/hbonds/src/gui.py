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

from chimerax.atomic import AtomicStructure
from chimerax.core.colors import BuiltinColors
from .hbond import rec_dist_slop, rec_angle_slop
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from chimerax.ui.options import Option, OptionsPanel, ColorOption, FloatOption, BooleanOption, IntOption, \
    OptionalRGBAOption, make_optional

class HBondsGUI(QWidget):
    def __init__(self, session, tool_window, *, settings_name="",
            # settings_name values:
            #   empty string: remembered across sessions and the same as for the main H-bond GUI
            #   custom string (e.g. "rotamers"):  remembered across sessions and specific to your
            #     your interface, not shared with other H-bond GUIs.  The string " h-bond GUI" 
            #     will be appended to the provided string to yield the final settings name.
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
            angle_slop=rec_angle_slop, color=AtomicStructure.default_hbond_color,
            dashes=AtomicStructure.default_hbond_dashes, dist_slop=rec_dist_slop, inter_model=True,
            inter_submodel=False, intra_model=True, intra_mol=True, intra_res=True, log=False,
            radius=AtomicStructure.default_hbond_radius, relax=True, restrict="any", retain_current=False,
            reveal=False, salt_only=False, save_file=None, show_dist=False,
            slop_color=BuiltinColors["dark orange"], two_colors=False,

            # what controls to show in the interface
            show_bond_restrict=True, show_color=True, show_dashes=True, show_inter_intra_model=True,
            show_intra_mol=True, show_intra_res=True, show_log=True, show_model_restrict=True,
            show_radius=True, show_relax=True, show_restrict=True, show_retain_current=True,
            show_reveal=True, show_salt_only=True, show_save_file=True, show_show_dist=True,
            show_slop=True, show_slop_color=True, show_two_colors=True):

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
            self.__settings = settings = _get_settings(session, settings_name, settings_defaults)
        final_val = {}
        for name in settings_defaults.keys():
            final_val[name] = getattr(settings, name) if settings else frame_dict[name]

        super().__init__()
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QRadioButton
        layout = QVBoxLayout()
        self.setLayout(layout)

        top_layout = QHBoxLayout()
        layout.addLayout(top_layout)

        top_left_options = OptionsPanel(sorting=False, scrolled=False)
        top_layout.addWidget(top_left_options)
        if show_color:
            self.__color_option = ColorOption("H-bond color", None if settings else color, None,
                attr_name="color", settings=settings)
            top_left_options.add_option(self.__color_option)
        if show_radius:
            self.__radius_option = FloatOption("Radius", None if settings else radius, None,
                attr_name="radius", min='positive', settings=settings)
            self.__radius_option.widget.setSuffix("\N{ANGSTROM SIGN}")
            top_left_options.add_option(self.__radius_option)
        if show_dashes:
            self.__dashes_option = IntOption("Number of dashes", None if settings else dashes, None,
                attr_name="dashes", min=0, settings=settings)
            top_left_options.add_option(self.__dashes_option)
        if show_show_dist:
            self.__show_dist_option = BooleanOption("Label H-bond with distance",
                None if settings else show_dist, None, attr_name="show_dist", settings=settings)
            top_left_options.add_option(self.__show_dist_option)

        if show_inter_intra_model:
            group = QGroupBox("Find these bonds")
            top_layout.addWidget(group)
            group_layout = QVBoxLayout()
            group.setLayout(group_layout)
            self.__inter_model_button = QRadioButton("inter-model")
            group_layout.addWidget(self.__inter_model_button)
            self.__intra_model_button = QRadioButton("intra-model")
            group_layout.addWidget(self.__intra_model_button)
            self.__both_model_button = QRadioButton("both")
            group_layout.addWidget(self.__both_model_button)
            if final_val['inter_model'] and final_val['intra_model']:
                self.__both_model_button.setChecked(True)
            elif final_val['inter_model']:
                self.__inter_model_button.setChecked(True)
            else:
                self.__intra_model_button.setChecked(True)

        if show_relax:
            self.__relax_group = group = QGroupBox("Relax H-bond constraints")
            layout.addWidget(group)
            group.setCheckable(True)
            group.setChecked(final_val['relax'])
            relax_layout = QVBoxLayout()
            relax_layout.setContentsMargins(0,0,0,0)
            relax_layout.setSpacing(0)
            group.setLayout(relax_layout)
            if show_slop:
                slop_layout = QHBoxLayout()
                slop_layout.setContentsMargins(0,0,0,0)
                slop_layout.setSpacing(0)
                relax_layout.addLayout(slop_layout)
                slop_layout.addWidget(QLabel("Relax constraints by:"),
                    alignment=Qt.AlignRight | Qt.AlignVCenter)
                slop_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(0,0,0,0))
                slop_layout.addWidget(slop_options, alignment=Qt.AlignLeft | Qt.AlignVCenter)
                self.__dist_slop_option = FloatOption("", None if settings else dist_slop, None,
                    attr_name="dist_slop", settings=settings)
                self.__dist_slop_option.widget.setSuffix("\N{ANGSTROM SIGN}")
                slop_options.add_option(self.__dist_slop_option)
                self.__angle_slop_option = FloatOption("", None if settings else angle_slop, None,
                    attr_name="angle_slop", settings=settings)
                self.__angle_slop_option.widget.setSuffix("\N{DEGREE SIGN}")
                slop_options.add_option(self.__angle_slop_option)
            if show_slop_color:
                slop_color_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(0,0,0,0))
                relax_layout.addWidget(slop_color_options)
                if final_val['two_colors']:
                    default_value = final_val['slop_color']
                    kw = {}
                else:
                    default_value = None
                    kw = { 'initial_color': final_val['slop_color'] }
                self.__slop_color_option = OptionalRGBAOption("Color H-bonds not meeting precise criteria"
                    " differently", default_value, None, **kw)
                slop_color_options.add_option(self.__slop_color_option)

        self.__bottom_options = bottom_options = OptionsPanel(sorting=False, scrolled=False,
            contents_margins=(0,0,0,0))
        layout.addWidget(bottom_options)

        if show_model_restrict:
            self.__model_restrict_option = OptionalModelRestrictOption(session, tool_window,
                "Restrict to models...", None, self._model_restrict_cb, class_filter=AtomicStructure)
            bottom_options.add_option(self.__model_restrict_option)

        if show_bond_restrict:
            self.__bond_restrict_option = OptionalHBondRestrictOption(tool_window, "Only find H-bonds",
                None, None)
            bottom_options.add_option(self.__bond_restrict_option)

        if show_intra_mol:
            self.__intra_mol_option = BooleanOption("Include intra-molecule H-bonds",
                None if settings else intra_mol, None, attr_name="intra_mol", settings=settings)
            bottom_options.add_option(self.__intra_mol_option)

        if show_intra_res:
            self.__intra_res_option = BooleanOption("Include intra-residue H-bonds",
                None if settings else intra_res, None, attr_name="intra_res", settings=settings)
            bottom_options.add_option(self.__intra_res_option)

        if show_reveal:
            self.__reveal_option = BooleanOption("If endpoint atom hidden, show endpoint residue",
                None if settings else reveal, None, attr_name="reveal", settings=settings)
            bottom_options.add_option(self.__reveal_option)

        if show_retain_current:
            self.__retain_current_option = BooleanOption("Retain existing H-bonds",
                None if settings else retain_current, None, attr_name="retain_current", settings=settings)
            bottom_options.add_option(self.__retain_current_option)

        if show_save_file:
            self.__save_file_option = BooleanOption("Write information to file", False, None)
            bottom_options.add_option(self.__save_file_option)

        if show_log:
            self.__log_option = BooleanOption("Write information to log",
                None if settings else log, None, attr_name="log", settings=settings)
            bottom_options.add_option(self.__log_option)

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
                atom_spec = concise_model_spec(models)
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
            if self.__intra_model_button.isChecked():
                intra = True
                inter = False
            elif self.__inter_model_button.isChecked():
                intra = False
                inter = True
            else:
                intra = True
                inter = True
            settings['intra_model'] = intra
            settings['inter_model'] = inter
        else:
            settings['intra_model'] = settings['inter_model'] = None

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

        if self.__show_values['intra_mol']:
            settings['intra_mol'] = self.__intra_mol_option.value
        else:
            settings['intra_mol'] = None

        if self.__show_values['intra_res']:
            settings['intra_res'] = self.__intra_res_option.value
        else:
            settings['intra_res'] = None

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

    def _model_restrict_cb(self, opt):
        if opt.value is None:
            new_label = "Restrict to models..."
        else:
            new_label = "Restrict to models:"
        self.__bottom_options.change_label_for_option(opt, new_label)

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

class ModelRestrictOption(Option):
    def __init__(self, session, tool_window, *args, **kw):
        self.session = session
        self.tool_window = tool_window
        super().__init__(*args, **kw)

    def get_value(self):
        return self.widget.value

    def set_value(self, value):
        if value is None:
            self._model_chooser.hide()
            self.tool_window.shrink_to_fit()
        else:
            self._model_chooser.show()
            self.widget.value = value

    value = property(get_value, set_value)

    def make_callback(self):
        if self.value is None:
            self._model_chooser.hide()
            self.tool_window.shrink_to_fit()
        else:
            self._model_chooser.show()
        super().make_callback()

    def set_multiple(self):
        pass

    def _make_widget(self, **kw):
        from chimerax.ui.widgets import ModelListWidget
        self.widget = self._model_chooser = mc = ModelListWidget(self.session, **kw)
        self.widget.hide()

OptionalModelRestrictOption = make_optional(ModelRestrictOption)

class HBondRestrictOption(Option):
    restrict_kw_vals = ("any", "cross", "both")
    fixed_kw_menu_texts = ("with at least one end selected", "with exactly one end selected",
        "with both ends selected")
    atom_spec_menu_text = "between selection and atom spec..."

    def __init__(self, tool_window, *args, **kw):
        self.tool_window = tool_window
        super().__init__(*args, **kw)

    def get_value(self):
        text = self.__push_button.text()
        for val, val_text in zip(self.restrict_kw_vals, self.fixed_kw_menu_texts):
            if text == val_text:
                return val
        return self.__line_edit.text()

    def set_value(self, value):
        if value in self.restrict_kw_vals:
            self.__push_button.setText(self.fixed_kw_menu_texts[self.restrict_kw_vals.index(value)])
            self.__line_edit.hide()
            self.tool_window.shrink_to_fit()
        else:
            self.__push_button.setText(self.atom_spec_menu_text)
            self.__line_edit.setText(value)
            self.__line_edit.show()

    value = property(get_value, set_value)

    def set_multiple(self):
        self.__push_button.setText(self.multiple_value)

    def _make_widget(self, *, display_value=None, **kw):
        if display_value is None:
            display_value = self.fixed_kw_menu_texts[0]
        from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QMenu, QAction, QLineEdit
        self.widget = layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        self.__push_button = QPushButton(display_value, **kw)
        self.__push_button.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        menu = QMenu()
        self.__push_button.setMenu(menu)
        for label in self.fixed_kw_menu_texts + (self.atom_spec_menu_text,):
            action = QAction(label, self.__push_button)
            action.triggered.connect(lambda arg, s=self, lab=label: self._menu_cb(lab))
            menu.addAction(action)
        layout.addWidget(self.__push_button, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.__line_edit = QLineEdit()
        self.__line_edit.setMinimumWidth(72)
        layout.addWidget(self.__line_edit, alignment=Qt.AlignCenter)

    def _menu_cb(self, label):
        if label in self.fixed_kw_menu_texts:
            self.value = self.restrict_kw_vals[self.fixed_kw_menu_texts.index(label)]
        else:
            self.value = self.__line_edit.text()
        self.make_callback()
OptionalHBondRestrictOption = make_optional(HBondRestrictOption)

def _get_settings(session, base_name, settings_defaults):
    if base_name:
        settings_name = base_name + " H-bond GUI"
    else:
        settings_name = "H-bond GUI"
    from chimerax.core.settings import Settings
    class HBondGUISettings(Settings):
        EXPLICIT_SAVE = settings_defaults

    return HBondGUISettings(session, settings_name)

#TODO: settings that need explicit save: inter_model, intra_model, two_colors, slop_color
