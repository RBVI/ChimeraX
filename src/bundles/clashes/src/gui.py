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

from chimerax.atomic import AtomicStructure
from chimerax.core.colors import BuiltinColors
from .hbond import rec_dist_slop, rec_angle_slop
from PyQt5.QtCore import Qt
from chimerax.ui.options import Option, OptionsPanel, ColorOption, FloatOption, BooleanOption, IntOption, \
    OptionalRGBAOption, make_optional

class AtomProximityGUI(QWidget):
    def __init__(self, session, name, prox_word, cmd_name, color, radius, hbond_allowance, overlap_cutoff,
            *, settings_name="",

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
            atom_color=defaults["atom_color"],
            slop_color=BuiltinColors["dark orange"], two_colors=False,

            # what controls to show in the interface
            show_atom_color=True,
            show_color=True, show_hbond_allowance = True, show_name=False, show_overlap_cutoff=True,
            show_radius=True):

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

        if show_make_pseudobonds:
            self.__make_pb_group = group = QGroupBox("Display as pseudobonds")
            layout.addWidget(group)
            group.setCheckable(True)
            group.setChecked(final_val['make_pseudobonds'])
            make_pb_layout = QVBoxLayout()
            make_pb_layout.setContentsMargins(0,0,0,0)
            make_pb_layout.setSpacing(0)
            group.setLayout(make_pb_layout)
            make_pb_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(0,0,0,0))
            make_pb_layout.addWidget(make_pb_options)
            if show_color:
                self.__color_option = ColorOption("Color", None if settings else color, None,
                    attr_name="color", settings=settings)
                make_pb_options.add_option(self.__color_option)
            if show_radius:
                self.__radius_option = FloatOption("Radius", None if settings else radius, None,
                    attr_name="radius", min='positive', settings=settings)
                self.__radius_option.widget.setSuffix("\N{ANGSTROM SIGN}")
                make_pb_options.add_option(self.__radius_option)
            if show_dashes:
                self.__dashes_option = IntOption("Dashes", None if settings else dashes, None,
                    attr_name="dashes", min=0, settings=settings)
                make_pb_options.add_option(self.__dashes_option)
            if show_show_dist:
                self.__show_dist_option = BooleanOption("Distance label",
                    None if settings else show_dist, None, attr_name="show_dist", settings=settings)
                make_pb_options.add_option(self.__show_dist_option)
            if show_reveal:
                self.__reveal_option = BooleanOption("If endpoint atom hidden, show endpoint residue",
                    None if settings else reveal, None, attr_name="reveal", settings=settings)
                make_pb_options.add_option(self.__reveal_option)
            if show_retain_current:
                self.__retain_current_option = BooleanOption("Retain pre-existing H-bonds",
                    None if settings else retain_current, None, attr_name="retain_current",
                    settings=settings)
                make_pb_options.add_option(self.__retain_current_option)

        if show_relax:
            self.__relax_group = group = QGroupBox("Relax distance and angle criteria")
            layout.addWidget(group)
            group.setCheckable(True)
            group.setChecked(final_val['relax'])
            relax_layout = QVBoxLayout()
            relax_layout.setContentsMargins(0,0,0,0)
            relax_layout.setSpacing(0)
            group.setLayout(relax_layout)
            relax_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(0,0,0,0))
            relax_layout.addWidget(relax_options)
            if show_slop:
                self.__dist_slop_option = FloatOption("Distance tolerance", None if settings else dist_slop,
                    None, attr_name="dist_slop", settings=settings)
                self.__dist_slop_option.widget.setSuffix("\N{ANGSTROM SIGN}")
                relax_options.add_option(self.__dist_slop_option)
                self.__angle_slop_option = FloatOption("Angle tolerance", None if settings else angle_slop,
                    None, attr_name="angle_slop", settings=settings)
                self.__angle_slop_option.widget.setSuffix("\N{DEGREE SIGN}")
                relax_options.add_option(self.__angle_slop_option)
            if show_slop_color:
                if final_val['two_colors']:
                    default_value = final_val['slop_color']
                    kw = {}
                else:
                    default_value = None
                    kw = { 'initial_color': final_val['slop_color'] }
                self.__slop_color_option = OptionalRGBAOption("Color H-bonds not meeting precise criteria"
                    " differently", default_value, None, **kw)
                relax_options.add_option(self.__slop_color_option)

        if show_model_restrict or show_inter_intra_model or show_bond_restrict or show_salt_only \
        or show_intra_mol or show_intra_res or show_inter_submodel:
            group = QGroupBox("Limit results:")
            layout.addWidget(group)
            limit_layout = QVBoxLayout()
            limit_layout.setContentsMargins(0,0,0,0)
            limit_layout.setSpacing(0)
            group.setLayout(limit_layout)
            self.__limit_options = limit_options = OptionsPanel(sorting=False, scrolled=False,
                contents_margins=(0,0,0,0))
            limit_layout.addWidget(limit_options)

            if show_model_restrict:
                self.__model_restrict_option = OptionalModelRestrictOption(session,
                    "Choose specific models...", None, self._model_restrict_cb, class_filter=AtomicStructure)
                limit_options.add_option(self.__model_restrict_option)

            if show_inter_intra_model:
                inter_val = final_val['inter_model'] and not final_val['intra_model']
                self.__inter_model_only_option = BooleanOption("Intermodel only", inter_val,
                    self._inter_model_cb)
                limit_options.add_option(self.__inter_model_only_option)
                intra_val = final_val['intra_model'] and not final_val['inter_model']
                self.__intra_model_only_option = BooleanOption("Intramodel only", intra_val,
                    self._intra_model_cb, attr_name="intra_model", settings=settings)
                limit_options.add_option(self.__intra_model_only_option)

            if show_bond_restrict:
                self.__bond_restrict_option = OptionalHBondRestrictOption("Limit by selection",
                    None, None)
                limit_options.add_option(self.__bond_restrict_option)

            if show_salt_only:
                self.__salt_only_option = BooleanOption("Salt bridges only",
                    None if settings else salt_only, None, attr_name="salt_only", settings=settings)
                limit_options.add_option(self.__salt_only_option)

            if show_intra_mol:
                self.__intra_mol_option = BooleanOption("Include intramolecule",
                    None if settings else intra_mol, None, attr_name="intra_mol", settings=settings)
                limit_options.add_option(self.__intra_mol_option)

            if show_intra_res:
                self.__intra_res_option = BooleanOption("Include intraresidue",
                    None if settings else intra_res, None, attr_name="intra_res", settings=settings)
                limit_options.add_option(self.__intra_res_option)

            if show_inter_submodel:
                self.__intra_submodel_option = BooleanOption("Include inter-submodel",
                    None if settings else intra_submodel, None, attr_name="inter_submodel",
                    settings=settings)
                limit_options.add_option(self.__intra_submodel_option)

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
                self.__log_option = BooleanOption("Log", None if settings else log, None, attr_name="log",
                    settings=settings)
                info_options.add_option(self.__log_option)

            if show_save_file:
                self.__save_file_option = BooleanOption("File", False, None)
                info_options.add_option(self.__save_file_option)

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

    def _inter_model_cb(self, opt):
        if opt.value:
            self.__intra_model_only_option.value = False

    def _intra_model_cb(self, opt):
        if opt.value:
            self.__inter_model_only_option.value = False

    def _model_restrict_cb(self, opt):
        if opt.value is None:
            new_label = "Restrict to models..."
        else:
            new_label = "Restrict to models:"
        self.__limit_options.change_label_for_option(opt, new_label)

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
    def __init__(self, session, name="clashes", hbond_allowance=defaults["clash_hbond_allowance"],
            overlap_cutoff=defaults["clash_threshold"], **kw):
        from .cmd import handle_clash_kw
        color, radius = handle_clash_kw(kw)
        super().__init__(session, name, "clash", "clashes", color, radius,
            hbond_allowance, overlap_cutoff, **kw)

class ContactsGUI(AtomProximityGUI):
    def __init__(self, session, name="contacts", hbond_allowance=defaults["clash_hbond_allowance"],
            overlap_cutoff=defaults["contact_threshold"], **kw):
        from .cmd import handle_contact_kw
        color, radius = handle_contact_kw(kw)
        super().__init__(session, name, "contact", "contacts", color, radius,
            hbond_allowance, overlap_cutoff, **kw)

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
