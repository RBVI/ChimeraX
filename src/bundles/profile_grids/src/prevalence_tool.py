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

class PrevalenceTool:

    def __init__(self, grid, tool_window):
        self.grid = grid
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/profilegrid.html#context"
        self._prev_chosen_cells = None

        from Qt.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QPushButton
        from Qt.QtCore import Qt
        from chimerax.ui.widgets import ColorButton
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.addWidget(QLabel("Color by <i>N</i>-fold change in chosen sequences"
            " relative to entire alignment"), alignment=Qt.AlignCenter)

        layout.addLayout(self._layout_main_colors(True))

        chosen_layout = QHBoxLayout()
        chosen_layout.setContentsMargins(0,0,0,0)
        chosen_layout.addStretch(1)
        layout.addLayout(chosen_layout)
        inner_layout = QVBoxLayout()
        inner_layout.setSpacing(2)
        chosen_cells_layout = QHBoxLayout()
        chosen_cells_layout.setSpacing(0)
        inner_layout.addLayout(chosen_cells_layout)
        do_color, color = self.grid.pg.settings.prevalence_chosen_color_info
        self.do_color_chosen_box = QCheckBox("Color chosen cells:  ")
        self.do_color_chosen_box.setChecked(do_color)
        chosen_cells_layout.addWidget(self.do_color_chosen_box)
        self.chosen_color_button = ColorButton(max_size=(16,16))
        self.chosen_color_button.color = color
        chosen_cells_layout.addWidget(self.chosen_color_button)
        chosen_cells_layout.addStretch(1)
        unchosen_cells_layout = QHBoxLayout()
        unchosen_cells_layout.setSpacing(0)
        inner_layout.addLayout(unchosen_cells_layout)
        do_color, color = self.grid.pg.settings.prevalence_unchosen_color_info
        self.do_color_unchosen_box = QCheckBox("Color unchosen cells in chosen columns:  ")
        self.do_color_unchosen_box.setChecked(do_color)
        unchosen_cells_layout.addWidget(self.do_color_unchosen_box)
        self.unchosen_color_button = ColorButton(max_size=(16,16))
        self.unchosen_color_button.color = color
        unchosen_cells_layout.addWidget(self.unchosen_color_button)
        unchosen_cells_layout.addStretch(1)
        chosen_layout.addLayout(inner_layout)
        chosen_layout.addStretch(1)

        go_back_layout = QHBoxLayout()
        go_back_layout.setSpacing(0)
        layout.addLayout(go_back_layout)
        go_back_layout.addStretch(2)
        revert_but = QPushButton("Revert")
        revert_but.clicked.connect(self.revert_coloring)
        go_back_layout.addWidget(revert_but)
        go_back_layout.addWidget(QLabel(" to last-used settings"))
        go_back_layout.addStretch(1)
        go_back_layout.addSpacing(5)
        go_back_layout.addStretch(1)
        reset_but = QPushButton("Reset")
        reset_but.clicked.connect(self.reset_coloring)
        go_back_layout.addWidget(reset_but)
        go_back_layout.addWidget(QLabel(" to factory defaults"))
        go_back_layout.addStretch(2)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(0)
        layout.addLayout(button_layout)
        button_layout.addStretch(2)
        remove_but = QPushButton("Remove")
        remove_but.clicked.connect(self.remove_coloring)
        button_layout.addWidget(remove_but)
        button_layout.addWidget(QLabel(" prevalence coloring"))
        button_layout.addStretch(1)
        button_layout.addSpacing(5)
        button_layout.addStretch(1)
        apply_but = QPushButton("Apply")
        apply_but.clicked.connect(self.color_by_prevalence)
        button_layout.addWidget(apply_but)
        button_layout.addWidget(QLabel(" above settings"))
        button_layout.addStretch(2)

        tool_window.ui_area.setLayout(layout)

    def color_by_prevalence(self):
        from Qt.QtGui import QColor, QBrush
        waypoints = self._gather_waypoints()
        if not waypoints:
            return
        do_small = self.do_small_box.isChecked()
        small_percent = self.small_percent_box.value()
        small_color = self.small_color_button.color
        smooth_transitions = self.transition_button.text() == "smooth"
        # Use same color for text of main cells, and white only if all the colors are dark
        from chimerax.core.colors import contrast_with
        contrasts = set()
        for factor, rgba in waypoints:
            contrasts.add(contrast_with([c/255.0 for c in rgba[:3]]))
            if len(contrasts) > 1:
                break
        if len(contrasts) > 1:
            text_rgb = (0,0,0)
        else:
            text_rgb = contrasts.pop()
        text_brush = QBrush(QColor(*[int(round(c * 255.0)) for c in text_rgb]))
        seqs = self.grid.seqs_from_cells()
        if self._prev_chosen_cells is None \
        or self._prev_chosen_cells != list(self.grid.chosen_cells.keys()):
            self.grid.pg.status("Computing subgrid", secondary=True)
            self._grid, weights = self.grid.pg.compute_grid(seqs)
            self.grid.pg.status("", secondary=True)
            self._prev_chosen_cells = list(self.grid.chosen_cells.keys())
            self._chosen_cols = set([col for row, col in self._prev_chosen_cells])
        tot_sub_seqs = len(seqs)
        tot_orig_seqs = len(self.grid.alignment.seqs)
        orig_grid = self.grid.grid_data
        sub_grid = self._grid
        smooth_transitions = self.transition_button.text() == "smooth"
        if do_small:
            small_brush = QBrush(QColor(*small_color[:3]))
            small_cutoff = int(small_percent * tot_orig_seqs / 100.0)
        for row, rect_info in self.grid.cell_rects.items():
            grid_row, rects = rect_info
            for col, rect in enumerate(rects):
                if col in self._chosen_cols:
                    continue
                orig_num = orig_grid[grid_row][col]
                if do_small and orig_num <= small_cutoff:
                    rect.setBrush(small_brush)
                    text_info = self.grid.cell_text_infos.get((row,col), None)
                    if text_info is not None:
                        text_info[0].setBrush(text_brush)
                else:
                    if orig_num == 0:
                        cell_factor = 1.0
                    else:
                        sub_num = sub_grid[grid_row][col]
                        cell_factor = (sub_num / tot_sub_seqs) / (orig_num / tot_orig_seqs)
                    prev_factor, prev_color = None, None
                    for factor, rgba in waypoints:
                        if cell_factor <= factor:
                            if prev_factor is None:
                                cell_rgb = rgba[:3]
                            else:
                                fraction = (cell_factor - prev_factor) / (factor - prev_factor)
                                if smooth_transitions:
                                    cell_rgb = [int(round((1 - fraction) * prev_rgba[c])
                                        + fraction * rgba[c]) for c in range(3)]
                                else:
                                    cell_rgb = (prev_rgba if fraction < 0.5 else rgba)[:3]
                            rect.setBrush(QBrush(QColor(*cell_rgb)))
                            text_info = self.grid.cell_text_infos.get((row,col), None)
                            if text_info is not None:
                                text_info[0].setBrush(text_brush)
                            break
                        prev_factor = factor
                        prev_rgba = rgba
                    else:
                        rect.setBrush(QBrush(QColor(*prev_rgba[:3])))
                        text_info = self.grid.cell_text_infos.get((row,col), None)
                        if text_info is not None:
                            text_info[0].setBrush(text_brush)

        do_chosen_color = self.do_color_chosen_box.isChecked()
        chosen_color = self.chosen_color_button.color
        if do_chosen_color:
            rgb8 = chosen_color[:3]
            brush = QBrush(QColor(*rgb8))
            for row, col in self.grid.chosen_cells.keys():
                rect = self.grid.cell_rects[row][1][col]
                rect.setBrush(brush)

        do_unchosen_color = self.do_color_unchosen_box.isChecked()
        unchosen_color = self.unchosen_color_button.color
        if do_unchosen_color:
            rgb8 = unchosen_color[:3]
            brush = QBrush(QColor(*rgb8))
            for chosen_col in set([col for row, col in self.grid.chosen_cells]):
                for row in range(len(self.grid.cell_rects)):
                    if (row,chosen_col) in self.grid.chosen_cells:
                        continue
                    self.grid.cell_rects[row][1][chosen_col].setBrush(brush)

        self.grid.pg.settings.prevalence_main_color_info = (
            True, waypoints,
            do_small, small_percent, tuple(small_color),
            smooth_transitions
        )
        self.grid.pg.settings.prevalence_chosen_color_info = (do_chosen_color, tuple(chosen_color))
        self.grid.pg.settings.prevalence_unchosen_color_info = (do_unchosen_color, tuple(unchosen_color))

    def remove_coloring(self):
        from Qt.QtGui import QColor, QBrush
        from chimerax.core.colors import contrast_with
        divisor = sum(self.grid.weights)
        for row, rect_info in self.grid.cell_rects.items():
            grid_row, rects = rect_info
            for col, rect in enumerate(rects):
                val = self.grid.grid_data[grid_row,col]
                fraction = val / divisor
                non_blue = int(255 * (1.0 - fraction) + 0.5)
                rect.setBrush(QBrush(QColor(non_blue, non_blue, 255)))
                try:
                    cell_text = self.grid.cell_text_infos[(row,col)][0]
                except KeyError:
                    continue
                text_rgb = contrast_with((non_blue/255.0, non_blue/255.0, 1.0))
                cell_text.setBrush(QBrush(QColor(*[int(round(c * 255.0)) for c in text_rgb])))

    def reset_coloring(self):
        from .settings import prevalence_defaults
        self._coloring_from_settings(prevalence_defaults)

    def revert_coloring(self):
        from .settings import prevalence_defaults
        self._coloring_from_settings({ key: getattr(self.grid.pg.settings, key)
            for key in prevalence_defaults.keys()
        })

    def _coloring_from_settings(self, coloring_info):
        do_main, color_info, do_small, small_threshold, small_color, smooth_transitions \
            = coloring_info["prevalence_main_color_info"]
        self.num_waypoints_box.setValue(len(color_info))
        from Qt.QtWidgets import QLabel, QSpinBox, QPushButton
        for row_info, row_widgets in zip(color_info, self._main_widgets):
            for widget, value in zip([rw for rw in row_widgets if not isinstance(rw, QLabel)], row_info):
                if isinstance(widget, QSpinBox):
                    widget.setValue(value)
                else:
                    widget.color = value
        self._update_palette_chooser()
        self.do_small_box.setChecked(do_small)
        self.small_percent_box.setValue(small_threshold)
        self.small_color_button.color = small_color
        self.transition_button.setText("smooth" if smooth_transitions else "sharp")
        do_color, color = coloring_info["prevalence_chosen_color_info"]
        self.do_color_chosen_box.setChecked(do_color)
        self.chosen_color_button.color = color
        do_uncolor, uncolor = coloring_info["prevalence_unchosen_color_info"]
        self.do_color_unchosen_box.setChecked(do_uncolor)
        self.unchosen_color_button.color = uncolor

    def _gather_waypoints(self):
        waypoint_info = {}
        for row_widgets in self._main_widgets:
            factor = row_widgets.factor_box.value()
            color = tuple(row_widgets.color_button.color)
            if factor in waypoint_info and waypoint_info[factor] != color:
                return tool_user_error("Cannot assign two different colors to same factor (%gx)" % factor)
            waypoint_info[factor] = color
        if len(waypoint_info) < 2:
            return tool_user_error("Less than 2 distinct factor values")
        return sorted(list(waypoint_info.items()))

    def _layout_main_colors(self, first_time=False):
        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton
        from Qt.QtWidgets import QDoubleSpinBox, QGridLayout, QGroupBox, QMenu, QCheckBox
        from Qt.QtCore import Qt
        from chimerax.ui.widgets import ColorButton
        from collections import namedtuple
        PrevalenceTuple = namedtuple("PrevalenceTuple", ["factor_box", "color_button"])
        if first_time:
            # first time setup; do_main is vestigial (do_main_box used to be checkable)
            do_main, color_info, do_small, small_threshold, small_color, smooth_transitions \
                = self.grid.pg.settings.prevalence_main_color_info
            self.do_main_box = QGroupBox("Color other columns by prevalence change:")
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
            self.num_waypoints_box.setValue(len(color_info))
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

            for row, row_ci in enumerate(color_info):
                threshold, color = row_ci
                factor_box = QDoubleSpinBox()
                factor_box.setRange(0.0, 999.9)
                factor_box.setDecimals(1)
                factor_box.setSingleStep(0.5)
                factor_box.setValue(threshold)
                factor_box.setAlignment(Qt.AlignRight)
                factor_box.setSuffix("x")
                color_button = ColorButton(pause_delay=0.5)
                color_button.color = color
                color_button.color_pause.connect(self._update_palette_chooser)
                row_widgets = PrevalenceTuple(factor_box, color_button)
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

            small_layout = QHBoxLayout()
            small_layout.setSpacing(0)
            layout.addLayout(small_layout)
            small_layout.addStretch(1)
            self.do_small_box = QCheckBox("But color cells with less than ")
            self.do_small_box.setChecked(do_small)
            small_layout.addWidget(self.do_small_box)
            self.small_percent_box = QDoubleSpinBox()
            self.small_percent_box.setRange(0.0, 99.999)
            self.small_percent_box.setDecimals(3)
            self.small_percent_box.setSingleStep(0.5)
            self.small_percent_box.setValue(small_threshold)
            self.small_percent_box.setAlignment(Qt.AlignRight)
            self.small_percent_box.setSuffix("%")
            small_layout.addWidget(self.small_percent_box)
            small_layout.addWidget(QLabel(" of sequences originally: "))
            self.small_color_button = ColorButton(max_size=(16,16))
            self.small_color_button.color = small_color
            small_layout.addWidget(self.small_color_button)
            small_layout.addStretch(1)

            transition_layout = QHBoxLayout()
            transition_layout.setSpacing(0)
            layout.addLayout(transition_layout)
            transition_layout.addStretch(1)
            transition_layout.addWidget(QLabel("Color transitions: "))
            self.transition_button = QPushButton("smooth" if smooth_transitions else "sharp")
            menu = QMenu(self.transition_button)
            menu.addAction("sharp")
            menu.addAction("smooth")
            menu.triggered.connect(lambda act, but=self.transition_button: but.setText(act.text()))
            self.transition_button.setMenu(menu)
            transition_layout.addWidget(self.transition_button)
            transition_layout.addStretch(1)
            return main_layout
        #reformatting
        layout = self.do_main_box.layout()
        num_waypoints = self.num_waypoints_box.value()
        prev_values = [widgets.factor_box.value() for widgets in self._main_widgets]
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
                label1 = QLabel(prev_row.label1.text())
                self._dynamic_layout.addWidget(label1, row, 0)
                factor_box = QDoubleSpinBox()
                factor_box.setRange(0.0, 999.9)
                factor_box.setDecimals(1)
                factor_box.setSingleStep(0.5)
                factor_box.setAlignment(Qt.AlignRight)
                factor_box.setSuffix("x")
                self._dynamic_layout.addWidget(factor_box, row, 1)
                label2 = QLabel(prev_row.label2.text())
                self._dynamic_layout.addWidget(label2, row, 2)
                color_button = ColorButton(pause_delay=0.5)
                color_button.color = prev_row.color_button.color
                color_button.color_pause.connect(self._update_palette_chooser)
                self._dynamic_layout.addWidget(color_button, row, 3)
                row_widgets = PrevalenceTuple(label1, factor_box, label2, color_button)
                self._main_widgets.append(row_widgets)
        else:
            return
        for i, row_widgets in enumerate(self._main_widgets):
            row_widgets.factor_box.setValue(
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
