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

    settings_info = {
        'atomspec_contents': ("Atomspec display style", "Labels", AtomSpecOption, None, None),
        'bg_color': ("Background color", "Background", ColorOption, "set bgColor %s",
            lambda val: val.hex_with_alpha()),
    }

    def __init__(self, session, ui_area):
        from PyQt5.QtWidgets import QTabWidget
        from .options import OptionsPanel
        self.session = session
        panels = {}
        tab_widget = TabWidget(ui_area)
        categories = []

        for setting, setting_info in self.settings_info.items():
            opt_name, category, opt_class, updater, converter = setting_info
            try:
                panel = panels[category]
            except KeyError:
                categories.append(category)
                panel = OptionsPanel(sorting=True)
                panels[category] = panel
            opt = opt_class(opt_name, getattr(core_settings, setting), attr_name=setting,
                callback=self._opt_cb)
            panel.add_option(opt)

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

