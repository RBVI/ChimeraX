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
            None),
        'bg_color': (
            "Background color",
            "Background",
            ColorOption,
            "set bgColor %s",
            lambda val: val.hex_with_alpha(),
            lambda ses, cb: ses.triggers.add_handler("background color changed", cb),
            lambda ses: ses.main_view.background_color,
            "Background color of main graphics window"),
    }

    def __init__(self, session, ui_area):
        from PyQt5.QtWidgets import QTabWidget
        from .options import OptionsPanel
        self.session = session
        panels = {}
        tab_widget = TabWidget(ui_area)
        categories = []

        for setting, setting_info in self.settings_info.items():
            opt_name, category, opt_class, updater, converter, notifier, fetcher = setting_info
            try:
                panel = panels[category]
            except KeyError:
                categories.append(category)
                panel = OptionsPanel(sorting=True)
                panels[category] = panel
            opt = opt_class(opt_name, getattr(core_settings, setting), self._opt_cb,
                attr_name=setting)
            panel.add_option(opt)
            if notifier is not None:
                notifier(ses, lambda tn, data, fetch=fetcher, opt=opt: opt.set(fetch()))

        categories.sort()
        for category in categories:
            tab_widget.addTab(panels[category], category)

    def _opt_cb(self, opt):
        setting = opt.attr_name
        setattr(core_settings, setting, opt.value)

        opt_name, category, opt_class, updater, converter = settings_info[setting]
        if updater is None:
            return

        if isinstance(updater, str):
            # command to run
            val = opt.value
            if converter:
                val = converter(val)
            from ..commands import run_command
            run_command(self.session, updater % val)
        else:
            updater(self.session, opt.value)

