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

def mutation_scores_color(session, attribute_name):
    '''Color structure residues as they were last colored with the specified attribute name.'''
    mch = _mutation_color_history(session)
    if mch is None:
        from chimerax.core.errors import UserError
        raise UserError('No mutation score attributes have been used for coloring residues.')
    mch.color_by_attribute(attribute_name)

def _mutation_color_history(session, create = False):
    mch = getattr(session, 'mutation_color_history', None)
    if mch is None and create:
        session.mutation_color_history = mch = MutationColorHistory(session)
    return mch

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
            acp[attr_name] = {}

        for opt_name, opt_value in option_values:
            acp[attr_name][opt_name] = opt_value

        if new_attr:
            self._session.triggers.activate_trigger('new mutation coloring', attr_name)
        
    def mutation_set_for_attribute(self, attr_name):
        from .ms_data import mutation_all_scores
        for mset in mutation_all_scores(self._session):
            if attr_name in mset.computed_values_names():
                return mset
        return None

    def attribute_names(self):
        return list(self._attribute_coloring_parameters.keys())

    def options(self, attribute_name):
        return self._attribute_coloring_parameters.get(attribute_name, {})
        
    def color_by_attribute(self, attribute_name):
        params = self._attribute_coloring_parameters.get(attribute_name)
        if params is None:
            return

        mset = self.mutation_set_for_attribute(attribute_name)
        if mset is None:
            return

        chains = mset.associated_chains()
        from chimerax.atomic import concise_chain_spec
        chain_spec = concise_chain_spec(chains)
        options = ' '.join(f'{opt_name} {opt_value}' for opt_name, opt_value in params.items())
        cmd_color = f'color byattribute r:{attribute_name} {chain_spec} {options}'

        from chimerax.core.commands import run
        self._ignore_color_command = True
        run(self._session, cmd_color)
        self._ignore_color_command = False
        
    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    def take_snapshot(self, session, flags):
        data = self._attribute_coloring_parameters.copy()
        data['version'] = 1
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        mch = _mutation_color_history(session)
        params = data.copy()
        params.pop('version', None)
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
        h.setText('Click a residue attribute name to color a structure using  a previous coloring.')
        layout.addWidget(h)
        
        from Qt.QtWidgets import QListWidget
        lw = QListWidget(parent)
        self._attribute_list = lw
        lw.setSortingEnabled(True)
        lw.setSelectionMode(lw.SingleSelection)
        lw.itemClicked.connect(self._attribute_clicked)
        layout.addWidget(lw)
        self._update_list()

        from chimerax.ui.widgets import button_row
        f, buttons = button_row(parent, [('Adjust colors', self._adjust_colors)],
                                spacing = 5, button_list = True)
        layout.addWidget(f)
                
        tw.manage(placement="side")

        session.triggers.add_handler('new mutation coloring', self._new_attribute)

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Mutation Coloring History', create=create)

    def _new_attribute(self, tname, value):
        if self.tool_window.tool_instance is None:
            return 'delete handler'	# GUI panel has been destroyed
        self._update_list()

    def _update_list(self):
        self._attribute_list.clear()
        attr_names = self._mutation_color_history.attribute_names()
        self._attribute_list.addItems(attr_names)

    def _attribute_clicked(self, item):
        attr_name = item.text()
        self._mutation_color_history.color_by_attribute(attr_name)
        
    def _adjust_colors(self):
        attr_names = [item.text() for item in self._attribute_list.selectedItems()]
        if len(attr_names) != 1:
            self.session.logger.error('Select exactly one attribute name in the list then press the "Adjust colors" button to show the Render by Attribute panel for adjusting the colors and color levels.')
            return

        attribute_name = attr_names[0]
        mch = self._mutation_color_history
        mset = mch.mutation_set_for_attribute(attribute_name)
        if mset is None:
            self.session.logger.error(f'No mutation set has an attribute "{attribute_name}".')
            return
        options = mch.options(attribute_name)
        if 'palette' in options:
            palette = []
            for thresh_color in options['palette'].split(':'):
                thresh, color = thresh_color.split(',')
                palette.append((float(thresh), color))
        else:
            palette = None
        no_value_color = options.get('noValueColor')

        from .ms_scatter_plot import _show_render_by_attribute_panel
        _show_render_by_attribute_panel(self.session, mset, attribute_name,
                                        palette = palette, no_value_color = no_value_color)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(
        required = [('attribute_name', StringArg)],
        keyword = [],
        synopsis = 'Color structure residues as they were last colored with the specified attribute name.'
    )
    register('mutationscores color', desc, mutation_scores_color, logger=logger)
