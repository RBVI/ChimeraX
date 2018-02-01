# vim: set expandtab ts=4 sw=4:

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

class SettingsTool:

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window

        from .settings import defaults, APPEARANCE, SINGLE_PREFIX, ALIGNMENT_PREFIX
        from chimerax.core.ui.options import CategorizedSettingsPanel, CategorizedOptionsPanel
        class AppearanceOptionsPanel(CategorizedOptionsPanel):
            def options(self, category=None):
                if category is None:
                    # parent panel is asking for all options
                    options = []
                    for category in self.categories():
                        options.extend(self.options(category))
                    return options
                return super().options(category)
        settings_panel = CategorizedSettingsPanel(sv.settings)
        appearance_panel = AppearanceOptionsPanel()
        settings_panel.add_tab(APPEARANCE, appearance_panel)

        for attr_name, option_info in defaults.items():
            try:
                category, description, option_class, ctor_keywords, default = option_info
            except:
                #TODO: don't use try/except when finished
                continue
            val = getattr(sv.settings, attr_name)
            opt = option_class(description, val, lambda o, s=self, cat=category:
                self._setting_change_cb(cat, o), attr_name=attr_name, **ctor_keywords)
            if category == APPEARANCE:
                if attr_name.startswith(SINGLE_PREFIX):
                    app_cat = "Single Sequence"
                elif attr_name.startswith(ALIGNMENT_PREFIX) or \
                (SINGLE_PREFIX + attr_name) in defaults:
                    app_cat = "Alignment"
                else:
                    app_cat = "All"
                appearance_panel.add_option(app_cat, opt)
            else:
                settings_panel.add_option(category, opt)
        from PyQt5.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        layout.addWidget(settings_panel)
        tool_window.ui_area.setLayout(layout)

    def _setting_change_cb(self, category, opt):
        opt.set_attribute(self.sv.settings)
        from .settings import APPEARANCE, REGIONS
        if category == APPEARANCE:
            self.sv.seq_canvas._reformat()
        elif category == REGIONS:
            if opt.attr_name == "show_sel":
                self.sv.region_browser._show_sel_cb()
                return
            if opt.attr_name.startswith('new_region'):
                return
            if opt.attr_name.startswith('sel'):
                regions = [self.sv.region_browser.get_region("ChimeraX selection")]
            else:
                name_part = self.sv.ERROR_REGION_STRING \
                    if opt.attr_name.startswith("error_region") else self.sv.GAP_REGION_STRING
                regions = []
                for region in self.sv.region_browser.regions:
                    if region.name.startswith(name_part) \
                    or region.name.startswith("partial " + name_part):
                        regions.append(region)
            if opt.attr_name.endswith("shown"):
                shown = opt.get()
                for region in regions:
                    region.shown = shown
            else:
                if opt.attr_name.startswith('sel'):
                    colors = [opt.get(), None]
                else:
                    colors = opt.get()
                for i, color in enumerate(colors):
                    for region in regions:
                        if i == 0 and region.name.startswith("partial"):
                            continue
                        if i == 1 and not region.name.startswith("partial"):
                            continue
                        if 'border' in opt.attr_name:
                            region.border_rgba = color
                        else:
                            region.interior_rgba = color
