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

from chimerax.core.core_settings import set_proxies, settings as core_settings
from .options import SymbolicEnumOption, ColorOption, BooleanOption, IntOption
from .options import StringOption, HostPortOption
from chimerax.core.colors import color_name

class UpdateIntervalOption(SymbolicEnumOption):
    values = ("day", "week", "month", "never")
    labels = ("every day", "every week", "every month", "never")

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
    #     side of the '%' string-formatting operator; otherwise it will be the second arg
    #     provided to the function call (session will be the first).
    # 6) Balloon help for option.  Can be None.
    # 7) Whether to automatically set the core setting.  If True, the setting will be changed
    #     before any updater is called.  Otherwise, the updater is in charge of setting the
    #     setting.  Usually only set to False if the updater needs to examine the old value.
    settings_info = {
        'background_color': (
            "Background color",
            "Background",
            ColorOption,
            "set bgColor %s",
            color_name,
            "Background color of main graphics window",
            True),
        'http_proxy': (
            'HTTP proxy',
            'Web Access',
            HostPortOption,
            lambda ses, val: set_proxies(),
            None,
            'HTTP proxy for ChimeraX to use when trying to reach web sites',
            True),
        'https_proxy': (
            'HTTPS proxy',
            'Web Access',
            HostPortOption,
            lambda ses, val: set_proxies(),
            None,
            'HTTPS proxy for ChimeraX to use when trying to reach web sites',
            True),
        'resize_window_on_session_restore': (
            'Resize graphics window on session restore',
            'Window',
            BooleanOption,
            None,
            None,
            'Whether to resize main window when restoring a session to the size it had when the session was saved.',
            True),
        'toolshed_update_interval': (
            "Toolshed update interval",
            "Toolshed",
            UpdateIntervalOption,
            None,
            None,
            'How frequently to check toolshed for new updates<br>',
            True),
    }

    def __init__(self, session, ui_area):
        from Qt.QtWidgets import QBoxLayout
        self.session = session
        from chimerax.core.commands import run
        from .options import CategorizedSettingsPanel
        self.options_widget = CategorizedSettingsPanel(help_cb=lambda *, category=None, ses=session, run=run:
            run(ses, "help help:user/preferences.html"
            + ("" if category is None else "#" + category.replace(' ', '').lower())))
        self.options = {}

        for setting, setting_info in self.settings_info.items():
            opt_name, category, opt_class, updater, converter, balloon, set_setting = setting_info
            if isinstance(opt_class, tuple):
                opt_class, kw = opt_class
                if 'session' in kw:
                    kw['session'] = self.session
            else:
                kw = {}
            opt = opt_class(opt_name, getattr(core_settings, setting), self._opt_cb,
                attr_name=setting, settings=core_settings, balloon=balloon, auto_set_attr=set_setting, **kw)
            self.options_widget.add_option(category, opt)
            self.options[setting] = opt

        core_settings.triggers.add_handler('setting changed', self._core_setting_changed)
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        layout.setSpacing(5)
        layout.addWidget(self.options_widget, 1)
        layout.setContentsMargins(0, 0, 0, 0)

        ui_area.setLayout(layout)

    def show_category(self, category):
        self.options_widget.show_category(category)

    def _core_setting_changed(self, trig_name, info):
        setting_name, old_val, new_val = info
        if setting_name in self.options:
            self.options[setting_name].value = new_val

    def _opt_cb(self, opt):

        setting = opt.attr_name
        opt_name, category, opt_class, updater, converter, balloon, set_setting = self.settings_info[setting]
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
