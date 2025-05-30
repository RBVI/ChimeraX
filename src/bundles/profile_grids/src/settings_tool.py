# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

class SettingsTool:

    def __init__(self, pg, tool_window):
        self.pg = pg
        self.tool_window = tool_window

        from .settings import defaults, APPEARANCE
        from chimerax.ui.options import CategorizedSettingsPanel, CategorizedOptionsPanel
        class AppearanceOptionsPanel(CategorizedOptionsPanel):
            def options(self, category=None):
                if category is None:
                    # parent panel is asking for all options
                    options = []
                    for category in self.categories():
                        options.extend(self.options(category))
                    return options
                return super().options(category)
        opt_sort_f = lambda opt: getattr(opt, 'pg_sort_val', opt.name)
        settings_panel = CategorizedSettingsPanel(option_sorting=opt_sort_f)

        for attr_name, option_info in defaults.items():
            category, description, sort_val, option_class, ctor_keywords, default = option_info
            val = getattr(pg.settings, attr_name)
            opt = option_class(description, val, lambda o, s=self, cat=category:
                self._setting_change_cb(cat, o), attr_name=attr_name, settings=pg.settings, **ctor_keywords)
            opt.pg_sort_val = sort_val
            settings_panel.add_option(category, opt)
        from Qt.QtWidgets import QVBoxLayout
        from Qt.QtCore import Qt
        for hdr in self.pg.alignment.headers:
            container, header_panel = settings_panel.add_option_group("Headers",
                group_label=hdr.ident.replace('_', ' ').title(), group_alignment=Qt.AlignLeft,
                contents_margins=(0,2,10,0), sorting=hdr.option_sorting)
            layout = QVBoxLayout()
            container.setLayout(layout)
            layout.addWidget(header_panel, alignment=Qt.AlignLeft)
            hdr.add_options(header_panel, verbose_labels=False)
        from Qt.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        layout.addWidget(settings_panel)
        tool_window.ui_area.setLayout(layout)

    def _setting_change_cb(self, category, opt):
        self.pg.grid_canvas._update_cell_texts()
