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

class ColorSchemeDialog:

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#context"

        from Qt.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        from chimerax.ui.options import OptionsPanel, InputFileOption, BooleanOption, StringOption
        options_panel = OptionsPanel(sorting=False, scrolled=False)
        layout.addWidget(options_panel)

        self.file_opt = InputFileOption("Color scheme file:", None, None)
        options_panel.add_option(self.file_opt)
        self.default_opt = BooleanOption("Make this scheme the default:", True, None)
        options_panel.add_option(self.default_opt)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.load_color_scheme)
        hide_self = lambda *args, tw=tool_window: setattr(tool_window, 'shown', False)
        bbox.rejected.connect(hide_self)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.sv.session, help=tool_window.help:
            run(ses, "help " + help))
        layout.addWidget(bbox)

        tool_window.ui_area.setLayout(layout)

    def load_color_scheme(self):
        file_name = self.file_opt.value
        if not file_name.strip():
            from chimerax.ui import tool_user_error
            tool_user_error("No file name specified", self.tool_window.ui_area,
                title="Load Color Scheme Dialog Error")
            return

        self.tool_window.shown = False

        from chimerax.core.commands import run, StringArg, BoolArg
        run(self.sv.session, "seq viewer %s loadColorScheme %s makeDefault %s"
            % (StringArg.unparse(self.sv.alignment.ident), StringArg.unparse(file_name),
            BoolArg.unparse(self.default_opt.value)))
