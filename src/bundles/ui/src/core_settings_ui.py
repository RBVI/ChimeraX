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

"""
core_settings_ui: GUI to control core settings
==============================================

TODO
"""

from chimerax.core.core_settings import settings as core_settings
from .options import SymbolicEnumOption, ColorOption, BooleanOption, IntOption, FloatOption
from .widgets import hex_color_name

class AtomSpecOption(SymbolicEnumOption):
    values = ("command", "serial", "simple")
    labels = ("command line", "serial number", "simple")

class CoreSettingsPanel:

    # settings_info is keyed on setting name, and value is a tuple composed of:
    #
    # 1) Description to display in the gui
    # 2) Category (also for gui)
    # 3) Option class to use, or (Option-class, {additional __init__ keywords}) tuple
    # 4) Updater to use when option changed.  One of:
    #     a) None, if no update necessary
    #     b) string, a command to run (and see next tuple component)
    #     c) a function to call (also see next tuple component)
    # 5) Converter to use with updater.  Either None (don't provide arg to updater), or
    #     a function to convert the option's value to a form usable with the updater.
    #     If the updater is a command, then the converted value will be as the right
    #     side of the '%' string-formatting operator; otherwise it will be the only arg
    #     provided to the function call.
    # 6) Change notifier.  Function that accepts a session and a trigger-handler-style callback
    #     as a arg and calls it when the setting changes. Can be None if not relevant.
    # 7) Function that fetches the setting value in a form that can be used to set the option.
    #     The session is provided as an argument to the function.  Should be None if and only
    #     if #6 is None.
    # 8) Balloon help for option.  Can be None.
    # 9) Whether to automatically set the core setting.  If True, the setting will be changed
    #     before any updater is called.  Otherwise, the updater is in charge of setting the
    #     setting.  Usually only set to False if the updater needs to examine the old value.
    settings_info = {
        'atomspec_contents': (
            "Atomspec display style",
            "Labels",
            AtomSpecOption,
            None,
            None,
            None,
            None,
            """How to format display of atomic data<br>
            <table>
            <tr><td>simple</td><td>&nbsp;</td><td>Simple readable form</td></tr>
            <tr><td>command line</td><td>&nbsp;</td><td>Form used in commands</td></tr>
            <tr><td>serial number</td><td>&nbsp;</td><td>Atom serial number</td></tr>
            </table>""",
            True),
        'bg_color': (
            "Background color",
            "Background",
            ColorOption,
            "set bgColor %s",
            hex_color_name,
            lambda ses, cb: ses.triggers.add_handler("background color changed", cb),
            lambda ses: ses.main_view.background_color,
            "Background color of main graphics window",
            True),
        'clipping_surface_caps': (
            'Surface caps',
            'Clipping',
            BooleanOption,
            'surface cap %s',
            None,
            lambda ses, cb: ses.triggers.add_handler("clipping caps changed", cb),
            lambda ses: core_settings.clipping_surface_caps,
            'Whether to cap surface holes created by clipping',
            False),
        'distance_color': (
            "Color",
            "Distances",
            ColorOption,
            "distance style color %s",
            hex_color_name,
            lambda ses, cb: ses.triggers.add_handler("distance color changed", cb),
            lambda ses: core_settings.distance_color,
            "Color of atomic distance monitors",
            False),
        'distance_dashes': (
            "Number of dashes",
            "Distances",
            (IntOption, {'min': 0 }),
            "distance style dashes %d",
            None,
            lambda ses, cb: ses.triggers.add_handler("distance dashes changed", cb),
            lambda ses: core_settings.distance_dashes,
            "How many dashes when drawing distance monitor.  Zero means solid line.  "
            "Currently, even numbers act the same as the next odd number.",
            False),
        'distance_decimal_places': (
            "Decimal places",
            "Distances",
            (IntOption, {'min': 0 }),
            "distance style decimalPlaces %d",
            None,
            lambda ses, cb: ses.triggers.add_handler("distance decimal places changed", cb),
            lambda ses: ses.pb_dist_monitor.decimal_places,
            "How many digits after the decimal point to show for distances",
            False),
        'distance_radius': (
            "Radius",
            "Distances",
            (FloatOption, {'min': 'positive', 'decimal_places': 3 }),
            "distance style radius %g",
            None,
            lambda ses, cb: ses.triggers.add_handler("distance radius changed", cb),
            lambda ses: core_settings.distance_radius,
            "Radial line thickness of distance",
            False),
        'distance_show_units': (
            'Show angstrom symbol (\N{ANGSTROM SIGN})',
            'Distances',
            BooleanOption,
            'distance style symbol  %s',
            None,
            lambda ses, cb: ses.triggers.add_handler("distance show units changed", cb),
            lambda ses: ses.pb_dist_monitor.show_units,
            'Whether to show angstrom symbol after the distancee',
            False),
        'resize_window_on_session_restore': (
            'Resize window on session restore',
            'Sessions',
            BooleanOption,
            None,
            None,
            None,
            None,
            'Whether to resize main window when restoring a session to the size it had when the session was saved.',
            True),
    }

    def __init__(self, session, ui_area):
        from PyQt5.QtWidgets import QBoxLayout
        self.session = session
        from .options import CategorizedSettingsPanel
        self.options_widget = CategorizedSettingsPanel(core_settings, "ChimeraX core")

        for setting, setting_info in self.settings_info.items():
            opt_name, category, opt_class, updater, converter, notifier, fetcher, balloon, \
                set_setting = setting_info
            if isinstance(opt_class, tuple):
                opt_class, kw = opt_class
            else:
                kw = {}
            opt = opt_class(opt_name, getattr(core_settings, setting), self._opt_cb,
                attr_name=setting, balloon=balloon, **kw)
            self.options_widget.add_option(category, opt)
            """
            self.options[setting] = opt
            """
            if notifier is not None:
                notifier(session,
                    lambda tn, data, fetch=fetcher, ses=session, opt=opt: opt.set(fetch(ses)))

        layout = QBoxLayout(QBoxLayout.TopToBottom)
        layout.setSpacing(5)
        layout.addWidget(self.options_widget, 1)
        layout.setContentsMargins(0, 0, 0, 0)

        ui_area.setLayout(layout)

    def _opt_cb(self, opt):

        setting = opt.attr_name
        opt_name, category, opt_class, updater, converter, notifier, fetcher, balloon, set_setting \
            = self.settings_info[setting]
        if set_setting:
            opt.set_attribute(core_settings)
        if updater is None:
            return

        if isinstance(updater, str):
            # command to run
            val = opt.value
            if converter:
                val = converter(val)
            from chimerax.core.commands import run
            run(self.session, updater % val)
        else:
            updater(self.session, opt.value)

    """
    def _reset(self):
        from chimerax.core.configfile import Value
        all_categories = self.all_check.isChecked()
        if not all_categories:
            cur_cat = self.options_widget.tabText(self.options_widget.currentIndex())
        for setting, setting_info in self.settings_info.items():
            if not all_categories and setting_info[1] != cur_cat:
                continue
            default_val = core_settings.PROPERTY_INFO[setting]
            if isinstance(default_val, Value):
                default_val = default_val.default
            opt = self.options[setting]
            opt.set(default_val)
            self._opt_cb(opt)

    def _restore(self):
        all_categories = self.all_check.isChecked()
        if not all_categories:
            cur_cat = self.options_widget.tabText(self.options_widget.currentIndex())
        for setting, setting_info in self.settings_info.items():
            if not all_categories and setting_info[1] != cur_cat:
                continue
            restore_val = core_settings.saved_value(setting)
            opt = self.options[setting]
            opt.set(restore_val)
            self._opt_cb(opt)

    def _save(self):
        all_categories = self.all_check.isChecked()
        # need to ensure "current value" is up to date before saving...
        cur_cat = self.options_widget.tabText(self.options_widget.currentIndex())
        save_settings = []
        for setting, setting_info in self.settings_info.items():
            if not all_categories and setting_info[1] != cur_cat:
                continue
            opt = self.options[setting]
            setattr(core_settings, setting, opt.get())
            save_settings.append(setting)
        # We don't simply use core_settings.save() when all_categories is True
        # since there may be core settings that aren't presented in the GUI
        core_settings.save(settings=save_settings)
    """
