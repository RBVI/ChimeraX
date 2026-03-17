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
    help = 'https://www.rbvi.ucsf.edu/chimerax/data/mutation-scores-oct2024/mutation_scores.html'

    def __init__(self, session, tool_name = 'Mutation Structure Coloring'):
        self._last_coloring_attribute = None

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
                
        tw.manage(placement="side")

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
                               'filter mutations', ('all', 'drag box on scatterplot'),
                               'value', ('mean', 'count', 'sum absolute', 'sum', 'min', 'median', 'max', 'stddev'),
                               'palette', ('blue to red', 'red to blue', 'white to red', 'red to white',
                                           'white to blue', 'blue to white'))
        self._color_score_menu, self._subtract_fit_menu, self._mutation_set_menu = controls.values
        score_names = self._score_names()
        self._color_score_menu.value = score_names[0] if score_names else 'none'
        self._mutation_set_menu_label = controls.labels[2]
        self._color_which_menu, self._color_score_type_menu, self._color_palette_menu = controls2.values

        smenu = self._color_score_menu.widget.menu()
        smenu.aboutToShow.connect(lambda *,menu=smenu: self._menu_about_to_show(menu))
        msmenu = self._mutation_set_menu.widget.menu()
        msmenu.aboutToShow.connect(lambda *,menu=smenu: self._menu_about_to_show(menu))
        sfmenu = self._subtract_fit_menu.widget.menu()
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
                        ('Name', self._name_coloring),
                        'coloring',
                        '',
                        ('Previous colorings', self._show_color_history_gui))
        self._coloring_name = cn = nc.values[0]
        cn.pixel_width = 200
        return nc.frame

    def _name_coloring(self):
        name = self._coloring_name.value.strip()
        if not name:
            from chimerax.core.errors import UserError
            raise UserError('Enter a name for the coloring then press the Name button')
        if self._last_coloring_attribute is None:
            from chimerax.core.errors import UserError
            raise UserError('Color a structure and then press the Name button')
        mch = _mutation_color_history(self.session)
        mch._set_coloring_name(self._last_coloring_attribute, name)
        mchp = MutationColorHistoryPanel.get_singleton(self.session, create=False)
        if mchp:
            mchp._update_list()

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
            if menu is self._subtract_fit_menu.widget.menu():
                menu.addAction('none')
            for name in self._score_names():
                menu.addAction(name)

    def _score_names(self):
        from .ms_data import mutation_scores
        ms_name = self._mutation_set_menu.value
        mset = mutation_scores(self.session, ms_name)
        return mset.score_names()

    def _color_structures(self):
        from .ms_data import mutation_all_scores, mutation_scores
        mutation_set_name = self._mutation_set_menu.value
        session = self.session
        scores = mutation_scores(session, mutation_set_name)
        if len(scores.associate_chains(session)) == 0:
            from chimerax.core.errors import UserError
            raise UserError(f'There are no structures associated with mutations {mutation_set_name}')

        score_name = self._color_score_menu.value
        which = self._color_which_menu.value # 'all' or 'drag box on scatterplot'
        score_type = self._color_score_type_menu.value # 'mean', 'sum absolute', 'sum'
        score_type = score_type.replace(" ", "_")
        palette = self._color_palette_menu.value # 'blue to red', ...
        subtract_fit_name = self._subtract_fit_menu.value

        ranges, range_name = self._box_ranges(score_name) if which == 'drag box on scatterplot' else (None,None)
        if ranges is None:
            which = 'all'

        attr_name = self._attribute_name(score_name, range_name, score_type, subtract_fit_name)
        self._last_define_attr_name = attr_name

        cmd_score = f'mutationscores define {attr_name} from {score_name} combine {score_type}'
        if ranges:
            cmd_score += f' ranges "{ranges}"'
        if subtract_fit_name != 'none':
            cmd_score += f' subtractFit {subtract_fit_name}'
        if len(mutation_all_scores(session)) > 1:
            cmd_score += f' mutationSet {mutation_set_name}'
        rvalues = self._run_command(cmd_score)
        values = [value for rnum, from_aa, to_aa, value in rvalues.all_values()]

        chains = scores.associated_chains()
        from chimerax.atomic import concise_chain_spec
        chain_spec = concise_chain_spec(chains)

        palette_spec = self._palette_specifier(palette, values)

        cmd_color = f'color byattribute r:{attr_name} {chain_spec} palette {palette_spec} noValueColor white'
        self._run_command(cmd_color)

        self._last_coloring_attribute = attr_name
    
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
        if score2 == score_name:
            range_name = f'{score2}_{"%.3g"%min2}_{"%.3g"%max2}_{score1}_{"%.3g"%min1}_{"%.3g"%max1}'
        else:
            range_name = f'{score1}_{"%.3g"%min1}_{"%.3g"%max1}_{score2}_{"%.3g"%min2}_{"%.3g"%max2}'
        return ranges, range_name
    
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

        '''
        min_score, max_score = min(values), max(values)
        step = (max_score - min_score) / (ncolors+1)
        thresholds = tuple((min_score + (i+1)*step) for i in range(ncolors))
        '''
        
        self._last_color_palette = tuple(zip(thresholds, colors))
        palette_spec = ':'.join(f'{"%.3g"%thresh},{color}'for thresh, color in zip(thresholds, colors))
        return palette_spec

    def _show_render_by_attribute_gui(self):
        if self._last_define_attr_name is None or self._last_color_palette is None:
            self.session.logger.error('Color the structure first, then you can adjust colors')
            return

        mutation_set_name = self._mutation_set_menu.value
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, mutation_set_name)

        _show_render_by_attribute_panel(self.session, mset, self._last_define_attr_name, self._last_color_palette)

    def _show_color_history_gui(self):
        mch = MutationColorHistoryPanel.get_singleton(self.session, create=True)
        mch.display(True)
        return mch

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
    mch.color_by_name(coloring_name)

def _mutation_color_history(session, create = False):
    mch = getattr(session, 'mutation_color_history', None)
    if mch is None and create:
        session.mutation_color_history = mch = MutationColorHistory(session)
    return mch

def show_structure_coloring_gui(session):
    msc = MutationStructureColoring.get_singleton(session, create=True)
    msc.display(True)
    return msc

from chimerax.core.state import StateManager  # Handles session saving

class MutationColorHistory(StateManager):
    def __init__(self, session):
        self._session = session
        self._attribute_coloring_parameters = {}
        self._ignore_color_command = False
        triggers = session.triggers
        triggers.add_handler('command finished', self._command_finished)
        if not triggers.has_trigger('new mutation coloring'):
            triggers.add_trigger('new mutation coloring')

    def _command_finished(self, trigger_name, cmd_text):
        if self._ignore_color_command:
            return
        
        # Example: color byattribute a:bfactor #!1 target scab palette 63.64,blue:98.675,white:133.71,red
        if not cmd_text.startswith('color byattribute r:'):
            return
        
        fields = cmd_text.split()
        attr_name = fields[2][2:]
        mset = self.mutation_set_for_attribute(attr_name)
        if mset is None:
            return
        
        option_values = []
        for save_option in ['palette', 'noValueColor']:
            if save_option in fields:
                i = fields.index(save_option)+1
                if i < len(fields):
                    option_values.append((save_option, fields[i]))

        acp = self._attribute_coloring_parameters
        new_attr = attr_name not in acp
        if new_attr:
            acp[attr_name] = {'attribute_name': attr_name}

        for opt_name, opt_value in option_values:
            acp[attr_name][opt_name] = opt_value

        if new_attr:
            self._session.triggers.activate_trigger('new mutation coloring', attr_name)

    def _set_coloring_name(self, attribute_name, coloring_name):
        acp = self._attribute_coloring_parameters
        if attribute_name in acp:
            acp[attribute_name]['coloring_name'] = coloring_name

    def mutation_set_for_attribute(self, attr_name):
        from .ms_data import mutation_all_scores
        for mset in mutation_all_scores(self._session):
            if attr_name in mset.computed_values_names():
                return mset
        return None

    def coloring_names(self):
        coloring_names = []
        acp = self._attribute_coloring_parameters
        for attr_name, values in tuple(acp.items()):
            if self.mutation_set_for_attribute(attr_name) is None:
                del acp[attr_name]	        # Remove deleted attributes
            elif 'coloring_name' in values:
                coloring_names.append(values['coloring_name'])
        return coloring_names

    def coloring_options(self, coloring_name):
        acp = self._attribute_coloring_parameters
        for attr_name, values in acp.items():
            if values.get('coloring_name') == coloring_name:
                return values
        return None
        
    def _color_by_attribute(self, attribute_name):
        acp = self._attribute_coloring_parameters        
        params = acp.get(attribute_name)
        if params is None:
            return

        mset = self.mutation_set_for_attribute(attribute_name)
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
        
    def color_by_name(self, coloring_name):
        acp = self._attribute_coloring_parameters
        for attr_name, values in acp.items():
            if values.get('coloring_name') == coloring_name:
                self._color_by_attribute(attr_name)
                return

    def rename_coloring(self, coloring_name, new_name):
        acp = self._attribute_coloring_parameters
        for attribute_name, values in acp.items():
            if values.get('coloring_name') == coloring_name:
                values['coloring_name'] = new_name

    def remove_coloring(self, coloring_name):
        acp = self._attribute_coloring_parameters
        for attribute_name, values in tuple(acp.items()):
            if values.get('coloring_name') == coloring_name:
                del acp[attribute_name]

    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    def take_snapshot(self, session, flags):
        data = self._attribute_coloring_parameters.copy()
        data['version'] = 2
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
        mch._attribute_coloring_parameters = params
        return mch

    def reset_state(self, session):
        self._attribute_coloring_parameters.clear()

from chimerax.core.tools import ToolInstance
class MutationColorHistoryPanel(ToolInstance):
    help = 'https://www.rbvi.ucsf.edu/chimerax/data/mutation-scores-oct2024/mutation_scores.html'

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
        self._coloring_list.addItems(coloring_names)

    def _coloring_clicked(self, item):
        coloring_name = item.text()
        self._mutation_color_history.color_by_name(coloring_name)
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
        options = mch.coloring_options(coloring_name)
        mset = mch.mutation_set_for_attribute(options['attribute_name'])
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
            options = mch.coloring_options(coloring_name)
            attribute_name = options['attribute_name']
            mset = mch.mutation_set_for_attribute(attribute_name)
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
                        from_aa, to_aa, value = rscores[0]
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
            mch = self._mutation_color_history
            mch.rename_coloring(coloring_name, new_name)
            self._update_list()

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(
        required = [('coloring_name', StringArg)],
        keyword = [],
        synopsis = 'Color structure residues as they were last colored with the specified attribute name.'
    )
    register('mutationscores color', desc, mutation_scores_color, logger=logger)
