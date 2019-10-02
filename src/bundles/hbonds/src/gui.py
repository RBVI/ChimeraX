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
from chimerax.ui.options import Option, OptionsPanel, ColorOption, FloatOption, BooleanOption, IntOption, \
    OptionalRGBAOption, make_optional

class HBondsGUI(QWidget):
    def __init__(self, session, *, settings_name="",
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

        from inspect import getargvalues, currentframe
        arg_names, var_args, var_kw, frame_dict = getargvalues(currentframe())
        settings_defaults = {}
        for arg_name in arg_names:
            if not arg_name.startswith('show_') or 'show_' + arg_name in arg_names:
                settings_defaults[arg_name] = frame_dict[arg_name]
        if settings_name is None:
            self.__settings = settings = None
        else:
            self.__settings = settings = _get_settings(session, settings_name, settings_defaults)
        final_val = {}
        for name in settings_defaults.keys():
            final_val[name] = getattr(settings, name) if settings else frame_dict[name]

        super().__init__()
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QRadioButton
        from PyQt5.QtCore import Qt
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
                slop_options = OptionsPanel(sorting=False, scrolled=False)
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
                slop_color_options = OptionsPanel(sorting=False, scrolled=False)
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

        self.__bottom_options = bottom_options = OptionsPanel(sorting=False, scrolled=False)
        layout.addWidget(bottom_options)

        if show_model_restrict:
            self.__model_restrict_option = OptionalModelRestrictOption(session, "Restrict to models...",
                None, self._model_restrict_cb, class_filter=AtomicStructure)
            bottom_options.add_option(self.__model_restrict_option)

        if show_bond_restrict:
            self.__bond_restrict_option = HBondRestrictOption("Only find H-bonds",
                None if settings else restrict, None, attr_name="restrict", settings=settings)
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

    def _model_restrict_cb(self, opt):
        if opt.value is None:
            new_label = "Restrict to models..."
        else:
            new_label = "Restrict to models:"
        self.__bottom_options.change_label_for_option(opt, new_label)

class ModelRestrictOption(Option):
    def __init__(self, session, *args, **kw):
        self.session = session
        super().__init__(*args, **kw)

    def get_value(self):
        return self.widget.value

    def set_value(self, value):
        self.widget.value = value

    value = property(get_value, set_value)

    def set_multiple(self):
        pass

    def _set_enabled(self, enabled):
        if enabled:
            self._model_chooser.show()
        else:
            self._model_chooser.hide()
        super().enabled = enabled

    enabled = property(Option.enabled.fget, _set_enabled)

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

    def get_value(self):
        text = self.__push_button.text()
        for val, val_text in zip(self.restrict_kw_vals, self.fixed_kw_menu_texts):
            if text == val_text:
                return val
        return self.__line_edit.text()

    def set_value(self, value):
        try:
            self.__push_button.setText(self.fixed_kw_menu_texts[0])
            self.__line_edit.hide()
        except KeyError:
            self.__push_button.setText(self.atom_spec_menu_text)
            self.__line_edit.setText(value)
            self.__line_edit.show()

    value = property(get_value, set_value)

    def set_multiple(self):
        self.__push_button.setText(self.multiple_value)

    def _make_widget(self, *, display_value=None, **kw):
        if display_value is None:
            display_value = self.fixed_kw_menu_texts[0]
        from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QMenu, QAction, QLineEdit
        self.widget = QWidget()
        layout = QHBoxLayout()
        self.widget.setLayout(layout)
        self.__push_button = QPushButton(display_value, **kw)
        menu = QMenu()
        self.__push_button.setMenu(menu)
        for label in self.fixed_kw_menu_texts + (self.atom_spec_menu_text,):
            action = QAction(label, self.__push_button)
            action.triggered.connect(lambda arg, s=self, lab=label: self._menu_cb(lab))
            menu.addAction(action)
        layout.addWidget(self.__push_button)
        self.__line_edit = QLineEdit()
        layout.addWidget(self.__line_edit)

    def _menu_cb(self, label):
        if label in self.fixed_kw_menu_texts:
            self.value = self.restrict_kw_vals[self.fixed_kw_menu_texts.index(label)]
        else:
            self.value = self.__line_edit.text()
        self.make_callback()

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
