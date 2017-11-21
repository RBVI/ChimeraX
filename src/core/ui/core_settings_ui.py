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

from ..core_settings import settings as core_settings
from .options import SymbolicEnumOption, ColorOption
from .widgets import hex_color_name

class AtomSpecOption(SymbolicEnumOption):
    values = ("command", "serial", "simple")
    labels = ("command line", "serial number", "simple")

class CoreSettingsPanel:

    # settings_info is keyed on setting name, and value is a tuple composed of:
    #
    # 1) Description to display in the gui
    # 2) Category (also for gui)
    # 3) Option class to use
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
            </table>"""),
        'bg_color': (
            "Background color",
            "Background",
            ColorOption,
            "set bgColor %s",
            hex_color_name,
            lambda ses, cb: ses.triggers.add_handler("background color changed", cb),
            lambda ses: ses.main_view.background_color,
            "Background color of main graphics window"),
    }

    def __init__(self, session, ui_area):
        from PyQt5.QtWidgets import QBoxLayout
        self.session = session
        from .options import CategorizedSettingsPanel
        self.options_widget = CategorizedSettingsPanel(core_settings)

        for setting, setting_info in self.settings_info.items():
            opt_name, category, opt_class, updater, converter, notifier, fetcher, balloon \
                = setting_info
            opt = opt_class(opt_name, getattr(core_settings, setting), self._opt_cb,
                attr_name=setting, balloon=balloon)
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
        opt.set_attribute(core_settings)

        setting = opt.attr_name
        opt_name, category, opt_class, updater, converter, notifier, fetcher, balloon \
            = self.settings_info[setting]
        if updater is None:
            return

        if isinstance(updater, str):
            # command to run
            val = opt.value
            if converter:
                val = converter(val)
            from ..commands import run
            run(self.session, updater % val)
        else:
            updater(self.session, opt.value)

    """
    def _reset(self):
        from ..configfile import Value
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
