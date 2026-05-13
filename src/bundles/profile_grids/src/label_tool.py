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

from chimerax.ui import tool_user_error

class LabelTool:

    def __init__(self, grid, tool_window):
        self.grid = grid
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/profilegrid.html#gridlabel"

        from Qt.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QPushButton
        from Qt.QtCore import Qt
        from chimerax.ui.widgets import ColorButton
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.addWidget(QLabel("Label residues with residue-type prevalence information"),
            alignment=Qt.AlignCenter)

        chains_layout = QHBoxLayout()
        chains_layout.setSpacing(2)
        layout.addLayout(chains_layout)
        chains_layout.addStretch(1)
        chains_layout.addWidget(QLabel("Chains:"))
        from chimerax.atomic import Residues
        from chimerax.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(grid.pg.session, filter_func=lambda c, aln=grid.alignment:
            c in Residues(aln.associated_residues()).chains)
        # Update on association change completely handled by grid's alignment_notification routine
        chains_layout.addWidget(self.chain_list)
        chains_layout.addStretch(1)

        self.sel_restrict_box = QCheckBox("Also limit to selected residues, if any")
        self.sel_restrict_box.setChecked(grid.pg.settings.label_selected_only)
        layout.addWidget(self.sel_restrict_box, alignment=Qt.AlignCenter)

        layout.addSpacing(3)

        layout.addLayout(self._layout_main_colors(True))

        go_back_layout = QHBoxLayout()
        go_back_layout.setSpacing(0)
        layout.addLayout(go_back_layout)
        go_back_layout.addStretch(2)
        revert_but = QPushButton("Revert")
        revert_but.clicked.connect(self.revert_coloring)
        go_back_layout.addWidget(revert_but)
        go_back_layout.addWidget(QLabel(" to last-used settings"))
        go_back_layout.addStretch(1)
        go_back_layout.addSpacing(10)
        go_back_layout.addStretch(1)
        reset_but = QPushButton("Reset")
        reset_but.clicked.connect(self.reset_coloring)
        go_back_layout.addWidget(reset_but)
        go_back_layout.addWidget(QLabel(" to factory defaults"))
        go_back_layout.addStretch(2)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.label_residues)
        bbox.rejected.connect(lambda tw=self.tool_window: setattr(tw, 'shown', False))
        # Since ApplyRole is not AcceptRole, simply connecting to the Apply button won't dismiss the dialog
        bbox.button(qbbox.Apply).clicked.connect(lambda *args, fc=self.label_residues: fc(apply=True))
        if getattr(tool_window, 'help', None) is None:
            bbox.button(qbbox.Help).setEnabled(False)
        else:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=grid.pg.session, tw=tool_window:
                run(ses, "help " + tw.help))
        layout.addWidget(bbox)

        tool_window.ui_area.setLayout(layout)

    def label_residues(self, *, apply=False):
        from chimerax.ui import tool_user_error
        if not apply:
            self.tool_window.shown = False
            self.grid.pg.session.ui.processEvents()
        waypoints = self._gather_waypoints()
        if not waypoints:
            return
        chains = self.chain_list.value
        if not chains:
            self.tool_window.shown = True
            return tool_user_error("No chains chosen for labeling")
        sel_only = self.sel_restrict_box.isChecked()
        from chimerax.atomic import concise_residue_spec, Chains
        residues = set(Chains(chains).existing_residues)
        if sel_only:
            from chimerax.atomic import selected_residues
            sel_res = selected_residues(self.grid.pg.session)
            if sel_res:
                residues = set([r for r in sel_res if r in residues])
                if not residues:
                    self.tool_window.shown = True
                    return tool_user_error("No selected residues are in the chosen chains")
        bg_color = self.bg_color_button.color

        from chimerax.core.commands import StringArg, BoolArg, run
        from chimerax.core.colors import color_name
        bg_color_name = color_name(bg_color)
        waypoint_names = [(val, color_name(color)) for val, color in waypoints]
        cmd = ["sequence grid", StringArg.unparse(self.grid.alignment.ident)]
        cmd.extend(["label", concise_residue_spec(self.grid.pg.session, residues)])
        cmd.extend(["bgColor", StringArg.unparse(bg_color_name)])
        palette_arg = ':'.join(["%g,%s" % (val, cname) for val, cname in waypoint_names])
        cmd.extend(["palette", StringArg.unparse(palette_arg)])
        run(self.grid.pg.session, ' '.join(cmd))

        self.grid.pg.settings.label_palette = waypoint_names
        self.grid.pg.settings.label_background = bg_color_name
        self.grid.pg.settings.label_selected_only = sel_only

    def reset_coloring(self):
        from .settings import label_defaults
        self._coloring_from_settings(label_defaults)

    def revert_coloring(self):
        from .settings import label_defaults
        self._coloring_from_settings({ key: getattr(self.grid.pg.settings, key)
            for key in label_defaults.keys()
        })

    def _coloring_from_settings(self, coloring_info):
        palette_data = coloring_info["label_palette"]
        bg_color = coloring_info["label_background"]
        self.num_waypoints_box.setValue(len(palette_data))
        from Qt.QtWidgets import QLabel, QSpinBox, QPushButton
        for row_info, row_widgets in zip(palette_data, self._main_widgets):
            for widget, value in zip([rw for rw in row_widgets if not isinstance(rw, QLabel)], row_info):
                if isinstance(widget, QSpinBox):
                    widget.setValue(value)
                else:
                    widget.color = value
        self._update_palette_chooser()
        self.bg_color_button.color = bg_color
        self.sel_restrict_box.setChecked(coloring_info["label_selected_only"])

    def _gather_waypoints(self):
        from chimerax.ui import tool_user_error
        waypoint_info = {}
        for row_widgets in self._main_widgets:
            fraction = row_widgets.fraction_box.value()
            color = tuple(row_widgets.color_button.color)
            if fraction in waypoint_info and waypoint_info[fraction] != color:
                self.tool_window.shown = True
                return tool_user_error("Cannot assign two different colors to same fraction (%g)" % fraction)
            waypoint_info[fraction] = color
        if len(waypoint_info) < 2:
            self.tool_window.shown = True
            return tool_user_error("Less than 2 distinct fraction values")
        return sorted(list(waypoint_info.items()))

    def _layout_main_colors(self, first_time=False):
        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton
        from Qt.QtWidgets import QDoubleSpinBox, QGridLayout, QGroupBox, QMenu, QCheckBox
        from Qt.QtCore import Qt
        from chimerax.ui.widgets import ColorButton
        from collections import namedtuple
        PrevalenceTuple = namedtuple("PrevalenceTuple", ["fraction_box", "color_button"])
        box_range = (0, 1)
        box_decimals = 2
        box_step = 0.1
        if first_time:
            # first time setup; do_main is vestigial (do_main_box used to be checkable)
            palette_data = self.grid.pg.settings.label_palette
            bg_color = self.grid.pg.settings.label_background
            self.do_main_box = QGroupBox("Label color parameters")
            main_layout = QHBoxLayout()
            main_layout.addStretch(1)
            main_layout.addWidget(self.do_main_box)
            main_layout.addStretch(1)
            self._main_widgets = []
            layout = QVBoxLayout()
            layout.setSpacing(0)
            layout.setContentsMargins(0,0,0,0)
            self.do_main_box.setLayout(layout)

            num_waypoints_layout = QHBoxLayout()
            num_waypoints_layout.setSpacing(1)
            num_waypoints_layout.addStretch(1)
            num_waypoints_layout.addWidget(QLabel("Use "))
            self.num_waypoints_box = QSpinBox()
            self.num_waypoints_box.setRange(2,7)
            self.num_waypoints_box.setValue(len(palette_data))
            self.num_waypoints_box.valueChanged.connect(lambda *args, f=self._layout_main_colors: f())
            num_waypoints_layout.addWidget(self.num_waypoints_box)
            num_waypoints_layout.addWidget(QLabel(" colors/thresholds"))
            num_waypoints_layout.addStretch(1)
            layout.addLayout(num_waypoints_layout)

            centering_layout = QHBoxLayout()
            layout.addLayout(centering_layout)
            centering_layout.addStretch(1)
            self._dynamic_layout = QGridLayout()
            self._dynamic_layout.setSpacing(0)
            self._dynamic_layout.setColumnStretch(4, 1)

            for row, row_ci in enumerate(palette_data):
                threshold, color = row_ci
                fraction_box = QDoubleSpinBox()
                fraction_box.setRange(*box_range)
                fraction_box.setDecimals(box_decimals)
                fraction_box.setSingleStep(box_step)
                fraction_box.setValue(threshold)
                fraction_box.setAlignment(Qt.AlignRight)
                color_button = ColorButton(pause_delay=0.5)
                color_button.color = color
                color_button.color_pause.connect(self._update_palette_chooser)
                row_widgets = PrevalenceTuple(fraction_box, color_button)
                self._main_widgets.append(row_widgets)
                for col, widget in enumerate(row_widgets):
                    self._dynamic_layout.addWidget(widget, row, col)
            centering_layout.addLayout(self._dynamic_layout)
            centering_layout.addStretch(1)

            palette_layout = QHBoxLayout()
            palette_layout.setSpacing(0)
            layout.addLayout(palette_layout)
            palette_layout.addStretch(1)
            from chimerax.ui.widgets import PaletteChooser
            self.palette_chooser = PaletteChooser(self._palette_applied,
                label="Set colors from palette ")
            palette_layout.addWidget(self.palette_chooser)
            palette_layout.addStretch(1)
            self._update_palette_chooser()

            reverse_layout = QHBoxLayout()
            reverse_layout.setContentsMargins(0,0,0,0)
            reverse_layout.setSpacing(0)
            layout.addLayout(reverse_layout)
            reverse_layout.addStretch(1)
            rev_but = QPushButton("Reverse")
            rev_but.clicked.connect(self._reverse_colors)
            reverse_layout.addWidget(rev_but)
            reverse_layout.addWidget(QLabel(" colors"))
            reverse_layout.addStretch(1)

            bg_layout = QHBoxLayout()
            bg_layout.setSpacing(0)
            layout.addLayout(bg_layout)
            bg_layout.addStretch(1)
            bg_layout.addWidget(QLabel("Background color: "))
            self.bg_color_button = ColorButton(max_size=(16,16))
            self.bg_color_button.color = bg_color
            bg_layout.addWidget(self.bg_color_button)
            bg_layout.addStretch(1)

            return main_layout
        #reformatting
        layout = self.do_main_box.layout()
        num_waypoints = self.num_waypoints_box.value()
        prev_values = [widgets.fraction_box.value() for widgets in self._main_widgets]
        prev_min = min(prev_values)
        prev_max = max(prev_values)
        if num_waypoints < len(self._main_widgets):
            last_row = self._main_widgets.pop()
            for row_widgets in self._main_widgets[num_waypoints-1:]:
                for widget in row_widgets:
                    self._dynamic_layout.removeWidget(widget)
                    widget.deleteLater()
            for col, widget in enumerate(last_row):
                self._dynamic_layout.removeWidget(widget)
                self._dynamic_layout.addWidget(widget, num_waypoints-1, col)
            self._main_widgets = self._main_widgets[:num_waypoints-1]
            self._main_widgets.append(last_row)
        elif num_waypoints > len(self._main_widgets):
            while len(self._main_widgets) < num_waypoints:
                row = len(self._main_widgets)
                prev_row = self._main_widgets[row-1]
                fraction_box = QDoubleSpinBox()
                fraction_box.setRange(*box_range)
                fraction_box.setDecimals(box_decimals)
                fraction_box.setSingleStep(box_step)
                fraction_box.setAlignment(Qt.AlignRight)
                self._dynamic_layout.addWidget(fraction_box, row, 0)
                color_button = ColorButton(pause_delay=0.5)
                color_button.color = prev_row.color_button.color
                color_button.color_pause.connect(self._update_palette_chooser)
                self._dynamic_layout.addWidget(color_button, row, 1)
                row_widgets = PrevalenceTuple(fraction_box, color_button)
                self._main_widgets.append(row_widgets)
        else:
            return
        for i, row_widgets in enumerate(self._main_widgets):
            row_widgets.fraction_box.setValue(
                prev_min + (prev_max - prev_min) * i / (len(self._main_widgets) - 1))

        self._update_palette_chooser()

    def _palette_applied(self, palette_name):
        for row_widgets, rgba in zip(self._main_widgets, self.palette_chooser.rgbas):
            row_widgets.color_button.color = rgba

    def _reverse_colors(self):
        rgbas = []
        for row_widgets in self._main_widgets:
            rgbas.append([c for c in row_widgets.color_button.color])
        for row_widgets, rgba in zip(self._main_widgets, reversed(rgbas)):
            row_widgets.color_button.color = rgba
        self._update_palette_chooser()

    def _update_palette_chooser(self, *args):
        rgbas = []
        for row_widgets in self._main_widgets:
            rgbas.append([c/255.0 for c in row_widgets.color_button.color])
        self.palette_chooser.rgbas = rgbas
