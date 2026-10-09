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

from chimerax.core.tools import ToolInstance
class MutationStructureColoring(ToolInstance):
    help = 'help:user/tools/mutationscores.html#coloring'

    def __init__(self, session, tool_name = 'Mutation Structure Coloring'):
        self._temporary_coloring_name = 'last_mutation_coloring'
        self._coloring_attribute_name = 'mutation_score'  # Temporary residue attribute name
        self._last_coloring_name = None

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys = False)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Add structure coloring controls
        cc = self._create_coloring_controls(parent)
        layout.addWidget(cc)
        
        # Color, Adjust, Previous buttons
        bf = self._create_buttons(parent)
        layout.addWidget(bf)

        # Allow naming a coloring for future use.
        nc = self._create_naming_controls(parent)
        layout.addWidget(nc)
        
        layout.addStretch(1)    # Extra space at end
                
        tw.manage(placement='side')

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Mutation Structure Coloring', create=create)

    def _create_coloring_controls(self, parent):
        from chimerax.ui.widgets import column_frame, EntriesRow
        frame, layout = column_frame(parent, spacing=0)
        from .ms_data import mutation_scores_names
        mset_names = mutation_scores_names(self.session) + ('set1', 'set2')
        controls = EntriesRow(frame,
                              'Score', ('score1', 'score2'),  # Will be replaced when menu posted
                              'subtract fit', ('none', 'score1'),
                              'Mutations', mset_names)
        controls2 = EntriesRow(frame,        
                               'filter mutations', ('all', 'drag box on scatterplot', 'define ranges...'),
                               'value', ('mean', 'count', 'sum absolute', 'sum', 'min', 'median', 'max', 'stddev'),
                               'palette', ('blue to red', 'red to blue', 'white to red', 'red to white',
                                           'white to blue', 'blue to white'))
        self._color_score_menu, self._subtract_fit_menu, self._mutation_set_menu = csm, sfm, msm = controls.values
        score_names = self._score_names()
        csm.shorten_text('middle', 200)
        csm.value = score_names[0] if score_names else 'none'
        self._mutation_set_menu_label = controls.labels[2]
        self._color_which_menu, self._color_score_type_menu, self._color_palette_menu = controls2.values

        cwmenu = self._color_which_menu.widget.menu()
        cwmenu.triggered.connect(self._color_which_chosen)
        cwmenu.aboutToShow.connect(lambda *,menu=cwmenu: self._menu_about_to_show(menu))
        msm.shorten_text('middle', 200)
        smenu = csm.widget.menu()
        smenu.aboutToShow.connect(lambda *,menu=smenu: self._menu_about_to_show(menu))
        msmenu = msm.widget.menu()
        msmenu.aboutToShow.connect(lambda *,menu=msmenu: self._menu_about_to_show(menu))
        msmenu.triggered.connect(self._mutation_set_changed)
        sfmenu = sfm.widget.menu()
        sfmenu.aboutToShow.connect(lambda *,menu=sfmenu: self._menu_about_to_show(menu))
        self._set_mutation_set_menu_visibility()

        _mutation_color_history(self.session, create = True)  # Start tracking mutation residue coloring
        return frame
    
    def _create_buttons(self, parent):
        buttons = [
            ('Color structures', self._color_structures),
            ('Adjust colors', self._show_render_by_attribute_gui),
            ('Help', self._show_help),
        ]
        from chimerax.ui.widgets import button_row
        f, buttons = button_row(parent, buttons, spacing = 5, button_list = True)
        return f
    
    def _create_naming_controls(self, parent):
        from chimerax.ui.widgets import EntriesRow
        nc = EntriesRow(parent,
                        ('Name coloring', self._name_coloring),
                        '',
                        ('Previous colorings', self._show_color_history_gui))
        self._coloring_name = cn = nc.values[0]
        cn.pixel_width = 200
        return nc.frame

    def _name_coloring(self):
        new_coloring_name = self._coloring_name.value.strip()
        if not new_coloring_name:
            from chimerax.core.errors import UserError
            raise UserError('Enter a name for the coloring then press the Name button')
        coloring_name = self._last_coloring_name
        if not _rename_coloring_and_attribute(self.session, coloring_name, new_coloring_name):
            from chimerax.core.errors import UserError
            raise UserError('Press the "Color structure" button, then you can name the coloring')
        self._last_coloring_name = new_coloring_name

    @property
    def _mutation_set(self):
        mutation_set_name = self._mutation_set_menu.value
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, mutation_set_name)
        return mset
        
    def set_mutation_set(self, mset):
        self._mutation_set_menu.value = mset.name

    def _set_mutation_set_menu_visibility(self):
        from .ms_data import mutation_scores_names
        visible = (len(mutation_scores_names(self.session)) > 1)
        self._mutation_set_menu.widget.setVisible(visible)
        self._mutation_set_menu_label.setVisible(visible)

    def _mutation_set_changed(self, action):
        score_name = self._color_score_menu.value
        score_names = self._score_names()
        if score_name not in score_names:
            self._color_score_menu.value = score_names[0]

    def set_coloring_score(self, score_name):
        self._color_score_menu.value = score_name

    def _color_which_chosen(self):
        if self._color_which_menu.value == 'define ranges...':
            show_score_ranges_gui(self.session)

    def _menu_about_to_show(self, menu):
        menu.clear()
        if menu is self._mutation_set_menu.widget.menu():
            from .ms_data import mutation_scores_names
            for ms_name in mutation_scores_names(self.session):
                menu.addAction(ms_name)
        elif menu is self._color_which_menu.widget.menu():
            filters = ('all', 'drag box on scatterplot', 'define ranges...')
            named_ranges = _named_score_ranges(self.session)
            if named_ranges:
                filters += tuple(named_ranges.names())
            self._mod_names = self._mutation_set.modification_names()
            filters += self._mod_names
            for filter in filters:
                menu.addAction(filter)
        else:
            if menu is self._subtract_fit_menu.widget.menu():
                menu.addAction('none')
            for name in self._score_names():
                menu.addAction(name)
        # Show or hide mutation set menu if mutation sets opened or closed.
        self._set_mutation_set_menu_visibility()
        
    def _score_names(self):
        return self._mutation_set.score_names()

    def _color_structures(self):
        mset = self._mutation_set
        mutation_set_name = self._mutation_set_menu.value
        if len(mset.associated_chains()) == 0:
            from chimerax.core.errors import UserError
            raise UserError(f'There are no structures associated with mutations {mutation_set_name}')

        score_name = self._color_score_menu.value
        which = self._color_which_menu.value # 'all' or 'drag box on scatterplot' or named filtering
        score_type = self._color_score_type_menu.value # 'mean', 'sum absolute', 'sum'
        score_type = score_type.replace(" ", "_")
        palette = self._color_palette_menu.value # 'blue to red', ...
        subtract_fit_name = self._subtract_fit_menu.value
        modifications = None
        
        if which == 'drag box on scatterplot':
            ranges = self._box_ranges(score_name)
        elif which == 'all':
            ranges = None
        elif which == 'define ranges...':
            ranges = _get_score_ranges_from_gui(self.session)
        elif which in getattr(self, '_mod_names', []):
            ranges = None
            modifications = which
        else:
            ranges = self._named_ranges(which)

        attr_name = self._coloring_attribute_name
        from chimerax.core.commands import quote_if_necessary
        cmd_score = f'mutationscores define {attr_name} from {quote_if_necessary(score_name)} combine {score_type}'
        if ranges:
            cmd_score += f' ranges "{ranges}"'
        if modifications:
            cmd_score += f' modifications {modifications}'
        if subtract_fit_name != 'none':
            cmd_score += f' subtractFit {subtract_fit_name}'
        from .ms_data import mutation_all_scores
        if len(mutation_all_scores(self.session)) > 1:
            cmd_score += f' mutationSet {quote_if_necessary(mutation_set_name)}'
        rvalues = self._run_command(cmd_score)
        values = [value for variant, value in rvalues.all_values()]

        chains = mset.associated_chains()
        from chimerax.atomic import concise_chain_spec
        chain_spec = concise_chain_spec(chains)
        palette_spec = self._palette_specifier(palette, values)

        cmd_color = f'color byattribute r:{attr_name} {chain_spec} palette {palette_spec} noValueColor gray'
        self._run_command(cmd_color)

        # Save in coloring history.
        coloring_info = {
            'attribute_name': attr_name,
            'source_score': score_name,
            'combine_method': score_type,
            'mutation_set_name': mutation_set_name,
            'palette': palette_spec,
            'noValueColor': 'gray',
            }
        if ranges:
            coloring_info['filtering'] = ranges
        if subtract_fit_name != 'none':
            coloring_info['subtract_fit'] = subtract_fit_name
        mch = _mutation_color_history(self.session)
        coloring_name = self._temporary_coloring_name
        mch.add_coloring(coloring_name, coloring_info)
        self._last_coloring_name = coloring_name
        
    def _run_command(self, command):
        from chimerax.core.commands import run
        return run(self.session, command)

    def _box_ranges(self, score_name):
        from .ms_scatter_plot import MutationScatterPlot
        plots = [scatter_plot for scatter_plot in self.session.tools
                 if isinstance(scatter_plot, MutationScatterPlot) and scatter_plot._last_drag_box]
        if len(plots) == 1:
            box = plots[0]._last_drag_box
            if box is None:
                return None, None
        elif len(plots) == 0:
            from chimerax.core.errors import UserError
            raise UserError('Drag a box on a scatter plot before filtering by box')
        else:
            from chimerax.core.errors import UserError
            raise UserError('Multiple scatter plots have dragged box.  Clear all but one scatterplot box.')

        score1, min1, max1, score2, min2, max2 = box
        range1 = f'{score1} >= {"%.3g"%min1} and {score1} <= {"%.3g"%max1}'
        range2 = f'{score2} >= {"%.3g"%min2} and {score2} <= {"%.3g"%max2}'
        ranges = f'{range1} and {range2}'
        return ranges

    def _named_ranges(self, ranges_name):
        named_ranges = _named_score_ranges(self.session)
        if named_ranges is None or ranges_name not in named_ranges.names():
            from chimerax.core.errors import UserError
            raise UserError(f'No named score ranges {ranges_name}')

        score_ranges = named_ranges.score_ranges(ranges_name)
        ranges = ' and '.join(
            f'{score_range.score_name} {score_range.compare} {score_range.threshold}'
            for score_range in score_ranges)
        return ranges if ranges else None
    
    def _attribute_name(self, score_name, range_name, score_type, subtract_fit_name):
        subtract_fit = '' if subtract_fit_name == 'none' else f'_subtract_fit_{subtract_fit_name}'
        if range_name:
            if range_name.startswith(score_name):
                attr_name = f'{range_name}{subtract_fit}_{score_type}'
            else:
                attr_name = f'{score_name}_{range_name}{subtract_fit}_{score_type}'
        else:
            attr_name = f'{score_name}{subtract_fit}_{score_type}'
        return attr_name

    def _palette_specifier(self, palette, values):
        palette_colors = {
            'blue to red': ('blue', 'white', 'white', 'red'),
            'red to blue': ('red', 'white', 'white', 'blue'),
            'white to red': ('white', 'red'),
            'red to white': ('red', 'white'),
            'white to blue': ('white', 'blue'),
            'blue to white': ('blue', 'white'),
        }
        colors = palette_colors[palette]
        ncolors = len(colors)
        from numpy import mean, std
        m, sd = mean(values), std(values)
        imid = (ncolors-1)/2
        sd_range = 4    # Number of standard deviations from first color to last.
        sd_step = sd * (sd_range / (ncolors-1))
        thresholds = tuple((m + (i-imid)*sd_step) for i in range(ncolors))
        palette_spec = ':'.join(f'{"%.3g"%thresh},{color}'for thresh, color in zip(thresholds, colors))
        return palette_spec

    def _show_render_by_attribute_gui(self):
        coloring_name = self._last_coloring_name
        mch = _mutation_color_history(self.session)
        coloring_info = mch.coloring_info(coloring_name)
        if coloring_info is None:
            self.session.logger.error('Color the structure first, then you can adjust colors')
            return

        mutation_set_name = coloring_info['mutation_set_name']
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, mutation_set_name)

        attr_name = coloring_info['attribute_name']
        
        palette = []
        for val_col in coloring_info['palette'].split(':'):
            threshold, color = val_col.split(',')
            palette.append((float(threshold), color))

        _show_render_by_attribute_panel(self.session, mset, attr_name, palette)

    def _show_color_history_gui(self):
        mchp = MutationColorHistoryPanel.get_singleton(self.session, create=True)
        mchp.display(True)
        return mchp

    def _show_help(self):
        self._run_command(f'help {self.help}')

def _show_render_by_attribute_panel(session, mutation_set, attribute_name,
                                    palette = None, no_value_color = None):
    from chimerax.core.commands import run
    rba_gui = run(session, 'ui tool show "Render/Select by Attribute"')

    chains = mutation_set.associated_chains()
    models = list(set(chain.structure for chain in chains))
    no_value_info = (True, no_value_color) if no_value_color is not None else None
    rba_gui.configure(models = models, target = 'residues', tab = 'render', attr_name = attribute_name,
                      level_info = palette, render_type = rba_gui.RENDER_COLORS,
                      no_value_info = no_value_info)

def mutation_scores_color(session, coloring_name):
    '''Color structure residues as they were last colored with the specified coloring name.'''
    mch = _mutation_color_history(session)
    if mch is None:
        from chimerax.core.errors import UserError
        raise UserError('No mutation score colorings have been saved.')
    if mch.coloring_info(coloring_name) is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No coloring named {coloring_name}.')
    mch.apply_coloring(coloring_name)

def _mutation_color_history(session, create = False):
    mch = getattr(session, 'mutation_color_history', None)
    if mch is None and create:
        session.mutation_color_history = mch = MutationColorHistory(session)
    return mch

def show_structure_coloring_gui(session):
    msc = MutationStructureColoring.get_singleton(session, create=True)
    msc.display(True)
    return msc

class ScoreRanges(ToolInstance):
    help = 'help:user/tools/mutationscores.html#coloring'

    def __init__(self, session, tool_name = 'Mutation Score Ranges'):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys = False)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Create score range controls
        cc = self._create_score_range_controls(parent)
        layout.addWidget(cc)

        # Allow naming score ranges for use in the coloring gui.
        nc = self._create_naming_controls(parent)
        layout.addWidget(nc)
        
        layout.addStretch(1)    # Extra space at end
                
        tw.manage(placement=None)	# Floating

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Mutation Score Ranges', create=create)

    def _create_score_range_controls(self, parent):
        from chimerax.ui.widgets import column_frame, EntriesRow, radio_buttons
        frame, layout = column_frame(parent, spacing=0)

        from .ms_data import mutation_scores_names
        mset_names = mutation_scores_names(self.session) + ('set1', 'set2')
        controls = EntriesRow(frame,
                              'Score ranges', True, 'high', False, 'low',
                              'Mutations', mset_names)
        self._high_ranges, self._low_ranges, self._mutation_set_menu = controls.values
        radio_buttons(self._high_ranges, self._low_ranges)
        for checkbutton in (self._high_ranges, self._low_ranges):
            checkbutton.changed.connect(self._range_chosen)
        self._mutation_set_menu_label = controls.labels[3]
        msmenu = self._mutation_set_menu.widget.menu()
        msmenu.triggered.connect(self._mutation_set_chosen)
        msmenu.aboutToShow.connect(lambda: self._menu_about_to_show(msmenu))
        self._set_mutation_set_menu_visibility()

        thresh = EntriesRow(frame, 'Threshold +/-', 2.0, 'synonymous standard deviations from mean')
        self._sdev_threshold_entry = th = thresh.values[0]
        th.pixel_width = 25

        suffix = EntriesRow(frame, 'Show only score names ending in', '')
        self._score_name_suffix = snsuffix = suffix.values[0]
        snsuffix.widget.returnPressed.connect(self._suffix_changed)
        snsuffix.pixel_width = 100

        score_names_frame, sn_layout = column_frame(parent, spacing=0)
        sn_layout.setContentsMargins(20,0,0,0)
        self._score_names_frame = score_names_frame
        layout.addWidget(score_names_frame, stretch = 1)
        self._score_checkbutton_rows = []
        self._create_score_checkbuttons()

        name_menu = EntriesRow(frame, 'Show named ranges',
                               ('new', 'name1'))  # Will be replaced by named ranges when menu posted
        self._name_menu = name_menu.values[0]
        nmmenu = self._name_menu.widget.menu()
        nmmenu.triggered.connect(self._name_chosen)
        nmmenu.aboutToShow.connect(lambda: self._menu_about_to_show(nmmenu))

        flines, fl_layout = column_frame(frame, spacing=0)
        self._range_lines_frame = flines
        layout.addWidget(flines)
        self._range_rows = []

        return frame

    def _create_score_checkbuttons(self):
        for row in self._score_checkbutton_rows:
            row.frame.deleteLater()
        self._score_checkbutton_rows.clear()

        max_name_chars_per_line = 50
        score_names = self._score_names()
        i = 0
        while i < len(score_names):
            row_score_names = []
            line_args = []
            row_chars = 0
            for score_name in score_names[i:]:
                nchar = len(score_name)
                if len(row_score_names) == 0 or row_chars + nchar <= max_name_chars_per_line:
                    row_score_names.append(score_name)
                    line_args.append(False)
                    line_args.append(score_name)
                    row_chars += nchar
                    i += 1
                else:
                    break
            from chimerax.ui.widgets import EntriesRow
            score_checkbuttons = EntriesRow(self._score_names_frame, *line_args)
            self._score_checkbutton_rows.append(score_checkbuttons)
            for score_name, checkbutton in zip(row_score_names,score_checkbuttons.values):
                checkbutton.changed.connect(lambda enabled, score_name=score_name:
                                            self._score_chosen(score_name, enabled))

    def _mutation_set_chosen(self):
        self._clear_range_rows()
        self._create_score_checkbuttons()
        
    def _range_chosen(self):
        high, low = self._high_ranges.value, self._low_ranges.value
        if (high and low) or (not high and not low):
            return
        self._update_score_checkmarks()

    def _update_score_checkmarks(self):        
        compare = ('>=' if self._high_ranges.value else '<=')
        enabled_scores = []
        for row in self._range_rows:
            score_name, row_compare = [label.text() for label in row.labels[:2]]
            if row_compare == compare:
                enabled_scores.append(score_name)
        for row in self._score_checkbutton_rows:
            score_names = [label.text() for label in row.labels]
            for score_name, checkbutton in zip(score_names, row.values):
                checkbutton.value = (score_name in enabled_scores)

    def _suffix_changed(self):
        self._create_score_checkbuttons()
        self._update_score_checkmarks()

    def _name_chosen(self):
        ranges_name = self._name_menu.value
        if ranges_name == 'new':
            self._clear_range_rows()
            self._range_chosen()
        else:
            named_ranges = _named_score_ranges(self.session)
            if named_ranges and ranges_name in named_ranges.names():
                self._show_named_ranges(ranges_name)

    def _show_named_ranges(self, ranges_name):
        self._clear_range_rows()
        named_ranges = _named_score_ranges(self.session)
        score_ranges = named_ranges.score_ranges(ranges_name)
        for score_range in score_ranges:
            self._add_score_range(score_range)
        self._range_chosen()	# Set score checkbuttons
        self._ranges_name.value = ranges_name

    def _clear_range_rows(self):
        for row in self._range_rows:
            row.frame.deleteLater()
        self._range_rows.clear()

    def _score_chosen(self, score_name, enabled):
        if enabled:
            if self._find_range_row(score_name) is None:
                mutation_set_name = self._mutation_set_menu.value
                high = self._high_ranges.value
                compare = ('>=' if high else '<=')
                tlow,thigh = self._score_thresholds(score_name, mutation_set_name)
                threshold = '%.3g' % (thigh if high else tlow)
                score_range = ScoreRange(mutation_set_name, score_name, compare, threshold)
                self._add_score_range(score_range)
        else:
            row = self._find_range_row(score_name)
            if row:
                row.frame.deleteLater()
                self._range_rows.remove(row)

    def _score_thresholds(self, score_name, mutation_set_name):
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, mutation_set_name)
        score_values = mset.score_values(score_name)
        mean, sdev = score_values.synonymous_mean_and_sdev()
        if mean is None or sdev is None:
            mean, sdev = 0,1
        num_sd = self._sdev_threshold
        return mean-num_sd*sdev, mean+num_sd*sdev

    @property
    def _sdev_threshold(self):
        t = self._sdev_threshold_entry.value
        return 0 if t is None else t
    
    def _add_score_range(self, score_range):
        from chimerax.ui.widgets import EntriesRow
        sr = EntriesRow(self._range_lines_frame,
                        score_range.score_name,
                        score_range.compare,
                        '')
        threshold = sr.values[0]
        threshold.value = score_range.threshold
        threshold.pixel_width = 50
        self._range_rows.append(sr)

    def _find_range_row(self, score_name):
        compare = ('>=' if self._high_ranges.value else '<=')
        for row in self._range_rows:
            row_score_name, row_compare = [label.text() for label in row.labels[:2]]
            if row_score_name == score_name and row_compare == compare:
                return row
        return None
    
    def _create_naming_controls(self, parent):
        from chimerax.ui.widgets import EntriesRow
        naming = EntriesRow(parent,
                            ('Define name', self._name_ranges),
                            '',
                            ('Delete', self._delete_named_ranges))
        self._ranges_name = fn = naming.values[0]
        fn.pixel_width = 200
        return naming.frame

    def _name_ranges(self):
        ranges_name = self._ranges_name.value.strip()
        if not ranges_name:
            from chimerax.core.errors import UserError
            raise UserError('Enter a name for the score ranges then press the Define Name button')
        score_ranges = self._score_ranges()
        if len(score_ranges) == 0:
            from chimerax.core.errors import UserError
            raise UserError('No high or low score ranges specified')
        named_ranges = _named_score_ranges(self.session, create=True)
        named_ranges.add_name(ranges_name, score_ranges)
        self._name_menu.value = ranges_name

    def _score_ranges(self):
        score_ranges = []
        mutation_set_name = self._mutation_set_menu.value
        for row in self._range_rows:
            score_name, compare = [label.text() for label in row.labels[:2]]
            threshold = row.values[0].value
            sr = ScoreRange(mutation_set_name, score_name, compare, threshold)
            score_ranges.append(sr)
        return score_ranges
        
    def _delete_named_ranges(self):
        named_ranges = _named_score_ranges(self.session)
        if named_ranges is None:
            return
        ranges_name = self._ranges_name.value.strip()
        if not ranges_name:
            from chimerax.core.errors import UserError
            raise UserError('Enter a ranges name and then press the Delete button')
        if ranges_name not in named_ranges.names():
            from chimerax.core.errors import UserError
            raise UserError(f'There is no named ranges {ranges_name}')
        named_ranges.delete_name(ranges_name)
        
    def _set_mutation_set_menu_visibility(self):
        from .ms_data import mutation_scores_names
        visible = (len(mutation_scores_names(self.session)) > 1)
        self._mutation_set_menu.widget.setVisible(visible)
        self._mutation_set_menu_label.setVisible(visible)
    
    def _menu_about_to_show(self, menu):
        menu.clear()
        if menu is self._mutation_set_menu.widget.menu():
            from .ms_data import mutation_scores_names
            for ms_name in mutation_scores_names(self.session):
                menu.addAction(ms_name)
        else:
            menu.addAction('new')
            named_ranges = _named_score_ranges(self.session)
            if named_ranges:
                for ranges_name in named_ranges.names():
                    menu.addAction(ranges_name)
            
    def _score_names(self):
        from .ms_data import mutation_scores
        ms_name = self._mutation_set_menu.value
        mset = mutation_scores(self.session, ms_name)
        score_names = mset.score_names()
        suffix = self._score_name_suffix.value
        score_names = [score_name for score_name in score_names if score_name.endswith(suffix)]
        return score_names

from chimerax.core.state import StateManager  # Handles session saving

class NamedScoreRanges(StateManager):
    def __init__(self):
        self._named_ranges = {}		# ranges name -> list of ScoreRange
    def names(self):
        return tuple(self._named_ranges.keys())
    def add_name(self, ranges_name, score_ranges):
        self._named_ranges[ranges_name] = score_ranges
    def delete_name(self, ranges_name):
        if ranges_name in self._named_ranges:
            del self._named_ranges[ranges_name]
    def score_ranges(self, ranges_name):
        return self._named_ranges.get(ranges_name, [])

    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    def take_snapshot(self, session, flags):
        data = {'named_ranges': self._named_ranges}
        return data
    @classmethod
    def restore_snapshot(cls, session, data):
        nsr = _named_score_ranges(session, create=True)
        nsr._named_ranges = data['named_ranges']
        return nsr
    def reset_state(self, session):
        self._named_ranges.clear()

def _named_score_ranges(session, create = False):
    msr = getattr(session, 'mutation_score_ranges', None)
    if msr is None and create:
        session.mutation_score_ranges = msr = NamedScoreRanges()
    return msr

from chimerax.core.state import State  # Handles session saving

class ScoreRange(State):
    def __init__(self, mutation_set_name, score_name, compare, threshold):
        self.mutation_set_name = mutation_set_name
        self.score_name = score_name
        self.compare = compare		# '>=' or '<='
        self.threshold = threshold	# String '2.0'

    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    save_attr_names = ('mutation_set_name', 'score_name', 'compare', 'threshold')
    def take_snapshot(self, session, flags):
        data = {attr:getattr(self, attr) for attr in self.save_attr_names}
        return data
    @classmethod
    def restore_snapshot(cls, session, data):
        return ScoreRange(**data)

def show_score_ranges_gui(session):
    nsr = ScoreRanges.get_singleton(session, create=True)
    nsr.display(True)
    return nsr

def _get_score_ranges_from_gui(session):
    nsr = ScoreRanges.get_singleton(session)
    if nsr is None:
        return None
    score_ranges = nsr._score_ranges()
    if len(score_ranges) == 0:
        return None
    ranges = ' and '.join(
        f'{score_range.score_name} {score_range.compare} {score_range.threshold}'
        for score_range in score_ranges)
    return ranges

from chimerax.core.state import StateManager  # Handles session saving

class MutationColorHistory(StateManager):
    def __init__(self, session):
        self._session = session
        self._coloring_parameters = {}  # Maps coloring name to palette and attribute settings
        self._ignore_color_command = False
        triggers = session.triggers
        triggers.add_handler('command finished', self._command_finished)
        if not triggers.has_trigger('new mutation coloring'):
            triggers.add_trigger('new mutation coloring')

    def add_coloring(self, coloring_name, coloring_info):
        cp = self._coloring_parameters
        cp[coloring_name] = coloring_info

    def rename_coloring(self, coloring_name, new_coloring_name):
        cp = self._coloring_parameters
        if coloring_name not in cp:
            return
        values = cp[coloring_name]
        del cp[coloring_name]
        cp[new_coloring_name] = values
        self._session.triggers.activate_trigger('new mutation coloring', new_coloring_name)
            
    def coloring_info(self, coloring_name):
        cp = self._coloring_parameters
        return cp.get(coloring_name)

    def mutation_set_for_coloring(self, coloring_name):
        cp = self._coloring_parameters
        if coloring_name in cp:
            mset_name = cp.get('mutation_set_name')
            from .ms_data import mutation_scores
            mset = mutation_scores(self._session, mset_name)
            return mset
        return None

    def coloring_names(self):
        self._remove_colorings_for_closed_mutation_sets()
        cp = self._coloring_parameters
        return list(cp.keys())

    def _remove_colorings_for_closed_mutation_sets(self):
        cp = self._coloring_parameters
        for coloring_name, values in tuple(cp.items()):
            if self.mutation_set_for_coloring(coloring_name) is None:
                del cp[coloring_name]
        
    def apply_coloring(self, coloring_name):
        cp = self._coloring_parameters        
        params = cp.get(coloring_name)
        if params is None:
            return
        attribute_name = params['attribute_name']

        mset = self.mutation_set_for_coloring(coloring_name)
        if mset is None:
            return

        chains = mset.associated_chains()
        from chimerax.atomic import concise_chain_spec
        chain_spec = concise_chain_spec(chains)
        options = ' '.join(f'{opt_name} {opt_value}'
                           for opt_name, opt_value in params.items()
                           if opt_name in ('palette', 'noValueColor'))
        cmd_color = f'color byattribute r:{attribute_name} {chain_spec} {options}'

        from chimerax.core.commands import run
        self._ignore_color_command = True
        run(self._session, cmd_color)
        self._ignore_color_command = False

    def remove_coloring(self, coloring_name):
        cp = self._coloring_parameters
        if coloring_name in cp:
            del cp[coloring_name]

    def _command_finished(self, trigger_name, cmd_text):
        '''
        If user changes coloring with color byattribute command then update
        the saved coloring settings.
        '''
        if self._ignore_color_command:
            return
        
        # Example: color byattribute a:bfactor #!1 target scab palette 63.64,blue:98.675,white:133.71,red
        if not cmd_text.startswith('color byattribute r:'):
            return
        
        fields = cmd_text.split()
        attr_name = fields[2][2:]
        coloring_name = self._coloring_name_for_attribute(attr_name)
        if coloring_name is None:
            return
        
        option_values = []
        for save_option in ['palette', 'noValueColor']:
            if save_option in fields:
                i = fields.index(save_option)+1
                if i < len(fields):
                    option_values.append((save_option, fields[i]))
        
        cp = self._coloring_parameters
        for opt_name, opt_value in option_values:
            cp[coloring_name][opt_name] = opt_value

    def _coloring_name_for_attribute(self, attribute_name):
        cp = self._coloring_parameters
        cnames = [coloring_name for coloring_name, values in cp.items()
                  if values.get('attribute_name') == attribute_name]
        if len(cnames) != 1:
            return None
        return cnames[0]

    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    def take_snapshot(self, session, flags):
        data = self._coloring_parameters.copy()
        data['version'] = 3
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        mch = _mutation_color_history(session, create=True)
        params = data.copy()
        version = params.pop('version', None)
        if version == 1:
            for attr_name, values in params.items():
                if 'coloring_name' not in values:
                    values['coloring_name'] = attr_name
                if 'attribute_name' not in values:
                    values['attribute_name'] = attr_name
        if version == 2:
            # Key was attribute name and coloring_name was saved as value.
            # Change to coloring name as key.
            new_params = {}
            for attr_name, values in params.items():
                cname = values.get('coloring_name', attr_name)
                new_params[cname] = values
            params = new_params
        mch._coloring_parameters = params
        return mch

    def reset_state(self, session):
        self._coloring_parameters.clear()

from chimerax.core.tools import ToolInstance
class MutationColorHistoryPanel(ToolInstance):
    help = 'help:user/tools/mutationscores.html#coloring'

    def __init__(self, session, tool_name = 'Mutation Coloring History'):
        mch = _mutation_color_history(session, create = True)
        self._mutation_color_history = mch

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys = False)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        from Qt.QtWidgets import QLabel
        h = QLabel(parent)
        h.setText('Click a list entry to color a structure.')
        layout.addWidget(h)
        
        from Qt.QtWidgets import QListWidget
        lw = QListWidget(parent)
        self._coloring_list = lw
        lw.setSortingEnabled(True)
        lw.setSelectionMode(lw.ExtendedSelection)
        lw.itemClicked.connect(self._coloring_clicked)
        layout.addWidget(lw)
        self._update_list()

        from chimerax.ui.widgets import EntriesRow
        br = EntriesRow(parent,
                        ('Adjust colors', self._adjust_colors),
                        ('Rename', self._rename_coloring),
                        '',
                        ('Save .csv', self._save_csv),
                        ('Delete', self._delete_coloring),
                        spacing = 5)
        self._rename_entry = re = br.values[0]
        re.pixel_width = 200
        layout.addWidget(br.frame)
                
        tw.manage(placement="side")

        session.triggers.add_handler('new mutation coloring', self._new_coloring)

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Mutation Coloring History', create=create)

    def _new_coloring(self, tname, value):
        if self.tool_window.tool_instance is None:
            return 'delete handler'	# GUI panel has been destroyed
        self._update_list()

    def _update_list(self):
        self._coloring_list.clear()
        coloring_names = self._mutation_color_history.coloring_names()
        if 'last_mutation_coloring' in coloring_names:
            coloring_names.remove('last_mutation_coloring')
        self._coloring_list.addItems(coloring_names)

    def _coloring_clicked(self, item):
        coloring_name = item.text()
        self._mutation_color_history.apply_coloring(coloring_name)
        self._rename_entry.value = coloring_name

    def _selected_coloring_names(self):
        return [item.text() for item in self._coloring_list.selectedItems()]

    def _all_coloring_names(self):
        al = self._coloring_list
        return [al.item(i).text() for i in range(al.count())]

    def _adjust_colors(self):
        coloring_names = self._selected_coloring_names()
        if len(coloring_names) != 1:
            self.session.logger.error('Select exactly one coloring in the list then press the "Adjust colors" button to show the Render by Attribute panel for adjusting the colors and color levels.')
            return

        coloring_name = coloring_names[0]
        mch = self._mutation_color_history
        options = mch.coloring_info(coloring_name)
        mset = mch.mutation_set_for_coloring(coloring_name)
        if 'palette' in options:
            palette = []
            for thresh_color in options['palette'].split(':'):
                thresh, color = thresh_color.split(',')
                palette.append((float(thresh), color))
        else:
            palette = None
        from chimerax.core.colors import Color
        no_value_color = Color(options.get('noValueColor')).uint8x4()

        _show_render_by_attribute_panel(self.session, mset, options['attribute_name'],
                                        palette = palette, no_value_color = no_value_color)

    def _delete_coloring(self):
        coloring_names = self._selected_coloring_names()
        if len(coloring_names) == 0:
            self.session.logger.error('Select an colorings in the list then press the Delete button.')
            return
            
        mch = self._mutation_color_history
        for coloring_name in coloring_names:
            mch.remove_coloring(coloring_name)

        self._update_list()

    def _save_csv(self, *, value_format = '%.4g'):
        coloring_names = self._selected_coloring_names()
        if len(coloring_names) == 0:
            coloring_names = self._all_coloring_names()

        text, dir = self._residue_scores_csv(coloring_names, value_format = value_format)
        
        from Qt.QtWidgets import QFileDialog
        parent = self.tool_window.ui_area
        path, ftype  = QFileDialog.getSaveFileName(parent,
                                                   caption = 'Save Residue Scores',
                                                   directory = dir,
                                                   filter = 'Comma separated values (.csv)')
        if path:
            with open(path, 'w') as f:
                f.write(text)

    def _residue_scores_csv(self, coloring_names, value_format = '%.4g'):
        res = set()
        values = []
        mch = self._mutation_color_history
        for coloring_name in coloring_names:
            options = mch.coloring_info(coloring_name)
            attribute_name = options['attribute_name']
            mset = mch.mutation_set_for_coloring(coloring_name)
            if mset is None:
                self.session.logger.error(f'No mutation set has an attribute "{attribute_name}".')
                continue

            scores = mset.computed_values(attribute_name)
            res.update(scores.residue_numbers_and_types())
            values.append((coloring_name, scores.values_by_residue_number))
 
        res_num_and_type = list(res)
        res_num_and_type.sort()
        header = ','.join(['#residue number', 'residue type'] + [coloring_name for coloring_name, scores in values])
        lines = [header]
        for rnum, rtype in res_num_and_type:
            row = [str(rnum), rtype]
            for attr_name, scores in values:
                score = ''
                if rnum in scores:
                    rscores = scores[rnum]
                    if len(rscores) == 1:
                        variant, value = rscores[0]
                        score = value_format % value
                row.append(score)
            lines.append(','.join(row))
        text = '\n'.join(lines)

        from os.path import dirname
        dir = dirname(mset.path) if mset else None

        return text, dir
    
    def _rename_coloring(self):
        coloring_names = self._selected_coloring_names()
        if len(coloring_names) != 1:
            self.session.logger.error('Select one coloring in the list, edit the name, then press the Rename button.')
            return
        coloring_name = coloring_names[0]

        new_name = self._rename_entry.value.strip()
        if not new_name:
            self.session.logger.error(f'New name is blank.')
            return

        if new_name != coloring_name:
            _rename_coloring_and_attribute(self.session, coloring_name, new_name)

def _rename_coloring_and_attribute(session, coloring_name, new_name):
    mch = _mutation_color_history(session)
    info = mch.coloring_info(coloring_name)
    if info is None:
        return False
    old_attr_name = info['attribute_name']
    mset = mch.mutation_set_for_coloring(coloring_name)
    mset.rename_computed_values(old_attr_name, new_name)
    _rename_residue_attribute(mset.associated_chains(), old_attr_name, new_name)
    info['attribute_name'] = new_name
    mch.rename_coloring(coloring_name, new_name)
    return True

def _rename_residue_attribute(chains, attribute_name, new_name):
    # Remove current residue attribute with new_name.
    for chain in chains:
        for r in chain.residues:
            if hasattr(r, new_name):
                delattr(r, new_name)
    # Set new attribute and remove old attribute.
    count = 0
    for chain in chains:
        for r in chain.residues:
            if hasattr(r, attribute_name):
                rvalue = getattr(r, attribute_name)
                delattr(r, attribute_name)
                setattr(r, new_name, rvalue)
                count += 1
    if count > 0:
        session = chain.structure.session
        atype = int if isinstance(rvalue, int) else float
        from chimerax.atomic import Residue
        Residue.register_attr(session, new_name, "Deep Mutational Scan", attr_type=atype, supercede=True)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(
        required = [('coloring_name', StringArg)],
        keyword = [],
        synopsis = 'Color structure residues as they were last colored with the specified attribute name.'
    )
    register('mutationscores color', desc, mutation_scores_color, logger=logger)
