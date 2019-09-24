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
    OptionalRGBAOption

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
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel
        from PyQt5.QtCore import Qt
        layout = QVBoxLayout()
        self.setLayout(layout)

        top_layout = QHBoxLayout()
        layout.addLayout(top_layout)

        top_left_options = OptionsPanel(sorting=False)
        top_layout.addWidget(top_left_options)
        if show_color:
            self.__color_option = ColorOption("H-bond color", None if settings else color, None,
                attr_name="color", settings=settings)
            top_left_options.add_option(self.__color_option)
        if show_radius:
            self.__radius_option = FloatOption("Radius", None if settings else radius, None,
                attr_name="radius", min='positive', trailing_text="\N{ANGSTROM SIGN}", settings=settings)
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
            group.setLayout(relax_layout)
            if show_slop:
                slop_layout = QHBoxLayout()
                relax_layout.addLayout(slop_layout)
                slop_layout.addWidget(QLabel("Relax constraints by:"), Qt.AlignRight | Qt.AlignVCenter)
                slop_options = OptionsPanel(sorting=False)
                slop_layout.addWidget(slop_options)
                self.__slop_dist_option = FloatOption("", None if settings else slop_dist, None,
                    attr_name="slop_dist", settings=settings, trailing_text="\N{ANGSTROM SIGN}")
                slop_options.add_option(self.__slop_dist_option)
                self.__slop_angle_option = FloatOption("", None if settings else slop_angle, None,
                    attr_name="slop_angle", settings=settings, trailing_text="\N{DEGREE SIGN}")
                slop_options.add_option(self.__slop_angle_option)
            if show_slop_color:
                slop_color_options = OptionsPanel(sorting=False)
                relax_layout.addWidget(slop_color_options)
                if final_val['two_colors']:
                    default_value = final_val['slop_color']
                    kw = {}
                else:
                    default_value = None
                    kw = { 'initial_color': final_val['slop_color']
                self.__slop_color_option = OptionalRGBAOption("Color H-bonds not meeting precise criteria"
                    " differently", default_value, None, **kw)
                slop_color_options.add_option(self.__slop_dist_option)

        bottom_options = OptionsPanel(sorting=False)
        layout.addWidget(bottom_options)
        if show_model_restrict:
            #TODO: have an actual callback that updates the form label
            self.__model_restrict_option = ModelRestrictOption(session, "Restrict to models...", None, None)
            top_left_options.add_option(self.__model_restrict_option)

class ModelRestrictOption(Option):
    def __init__(self, session, *args, **kw):
        self.session = session
        super().__init__(*args, **kw)

    def get_value(self):
        if self._check_box.isChecked():
            return self._model_chooser.value
        return None

    def set_value(self, value):
        if value is None:
            self._check_box.setChecked(False)
            self._model_chooser.hide()
        else:
            self._check_box.setChecked(True)
            self._model_chooser.show()

    value = property(get_value, set_value)

    def _make_widget(self, **kw):
        from chimerax.ui.widgets import ModelListWidget
        from PyQt5.QtWidgets import QWidget, QCheckBox, QHBoxLayout
        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self._check_box = cb = QCheckBox()
        cb.clicked.connect(lambda state, s=self: s._widget_change(True))
        layout.addWidget(cb)
        self._model_chooser = mc = ModelListWidget(self.session, **kw)
        mc.value_changed.connect(self._widget_change)
        layout.addWidget(mc)
        self.widget.setLayout(layout)

    def _widget_change(self, button_change=False):
        self._model_chooser.setHidden(not self._check_box.isChecked())
        if button_change or self._check_box.isChecked():
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
